import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import pypdf
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import shutil
from streamlit_local_storage import LocalStorage

# --- Configuration ---
load_dotenv()
PDF_DIR = "data"
PERSIST_DIR = "chroma_db_bio"
COLLECTION_NAME = "bioinformatics_docs"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_GENERATIVE_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# --- Helper Functions ---

@st.cache_data
def load_and_process_pdfs_v2():
    """Loads PDFs from global PDF_DIR, extracts text, and splits into chunks."""
    pdf_directory = PDF_DIR
    all_texts = []
    if not os.path.exists(pdf_directory):
        st.error(f"PDF directory '{pdf_directory}' not found. Please create it and add your PDFs.")
        return None, None

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.warning(f"No PDF files found in '{pdf_directory}'.")
        return None, None

    total_files = len(pdf_files)

    for i, filename in enumerate(pdf_files):
        filepath = os.path.join(pdf_directory, filename)
        try:
            reader = pypdf.PdfReader(filepath)
            file_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    file_text += page_text + "\n"
            if file_text:
                all_texts.append({"filename": filename, "content": file_text})
            else:
                st.warning(f"Could not extract text from '{filename}'. Skipping.")
        except Exception as e:
            st.error(f"Error reading '{filename}': {e}")

    if not all_texts:
        st.error("No text could be extracted from any PDF files.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    doc_ids = []
    metadatas = []
    chunk_id_counter = 0

    for doc in all_texts:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            doc_ids.append(f"doc_{doc['filename']}_chunk_{chunk_id_counter}")
            metadatas.append({"source": doc["filename"]})
            chunk_id_counter += 1

    return all_chunks, doc_ids, metadatas

@st.cache_resource
def setup_vector_store(chunks, doc_ids, metadatas, api_key):
    """Sets up ChromaDB, embeds chunks, and adds them to the collection."""
    if not chunks:
        st.error("No text chunks available to create vector store.")
        return None

    try:
        genai.configure(api_key=api_key)

        client = chromadb.PersistentClient(path=PERSIST_DIR)

        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            if collection.count() > 0:
                return collection
        except Exception:
            pass

        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        if collection.count() == 0:
            embeddings = []
            batch_size = 100
            total_chunks = len(chunks)

            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i+batch_size]
                try:
                    result = genai.embed_content(
                        model=GEMINI_EMBEDDING_MODEL,
                        content=batch_chunks,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(result['embedding'])
                except Exception as e:
                    st.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                    return None
                time.sleep(1)

            if len(embeddings) != len(chunks):
                 st.error(f"Mismatch in number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}). Aborting.")
                 return None

            chroma_batch_size = 5000
            for i in range(0, len(chunks), chroma_batch_size):
                batch_end = min(i + chroma_batch_size, len(chunks))
                collection.add(
                    embeddings=embeddings[i:batch_end],
                    documents=chunks[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=doc_ids[i:batch_end]
                )

        return collection

    except Exception as e:
        st.error(f"Failed to setup vector store: {e}")
        return None

def get_relevant_context(query, collection, api_key, n_results=5):
    """Embeds query and retrieves relevant context from ChromaDB."""
    try:
        genai.configure(api_key=api_key)
        query_embedding_result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_result['embedding']

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        return results['documents'][0], results['metadatas'][0]
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return [], []

def generate_response(query, context_docs, context_metadatas, api_key):
    """Generates response using Gemini based on query and context."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_GENERATIVE_MODEL)

        context_string = ""
        sources = set()
        for doc, meta in zip(context_docs, context_metadatas):
            source_file = meta.get('source', 'Unknown source')
            context_string += f"Source: {source_file}\nContent:\n{doc}\n\n---\n\n"
            sources.add(source_file)

        prompt = f"""You are a helpful assistant knowledgeable in bioinformatics, answering questions based on the provided text snippets.
Directly answer the question using *only* the information available in the text snippets below.
Do not use any prior knowledge.
If the answer is not found in the snippets, state that the information is not available in the provided documents.

Context from documents:
{context_string}

Question:
{query}

Answer:
"""

        response = model.generate_content(prompt)

        response_text = response.text
        if sources:
            response_text += "\n\n*Sources:* " + ", ".join(sorted(list(sources)))

        return response_text

    except Exception as e:
        st.error(f"Error generating response with Gemini: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- Streamlit App ---

st.set_page_config(page_title="Bioinformatics Chat", page_icon="🧬")
st.title("🧬 Bioinformatics Chatbot")
st.caption("Ask questions about bioinformatics.")

localS = LocalStorage()

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    api_key_from_storage = localS.getItem("gemini_api_key")
    api_key_input = st.text_input(
        "Enter Google AI API Key",
        type="password",
        key="api_key_input",
        help="Get your key from Google AI Studio. Stored locally in your browser.",
        value=api_key_from_storage if api_key_from_storage else ""
    )

    if api_key_input:
        localS.setItem("gemini_api_key", api_key_input)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

    st.divider()
    st.subheader("Database Management")
    if st.button("Re-create Knowledge Base", help=f"Deletes the current database ({PERSIST_DIR}) and re-processes PDFs from '{PDF_DIR}'. This requires a restart."):
        with st.spinner("Attempting to delete existing database and clear cache..."):
            delete_success = False
            if os.path.exists(PERSIST_DIR):
                try:
                    shutil.rmtree(PERSIST_DIR)
                    st.success(f"Deleted database directory: {PERSIST_DIR}")
                    delete_success = True
                except PermissionError as pe:
                     st.error(f"Error deleting directory {PERSIST_DIR}: File is likely still in use by the app. Please RESTART the Streamlit app completely. The database will be rebuilt after restart. Details: {pe}")
                except Exception as e:
                    st.error(f"Error deleting directory {PERSIST_DIR}: {e}")
            else:
                st.info("Database directory not found, nothing to delete.")
                delete_success = True

            if delete_success:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cleared Streamlit cache. Please refresh the page or restart the app to rebuild the knowledge base.")
            else:
                 st.warning("Could not delete database directory. Cache not cleared. Please restart the app manually.")

# --- API Key Handling & Initialization ---

api_key = api_key_input

if not api_key:
    st.info("Please enter your Google AI API Key in the sidebar to begin.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    list(genai.list_models())
except Exception as e:
    st.error(f"Invalid API Key or configuration error. Please check the key in the sidebar. Error: {e}")
    st.stop()

# --- Initialization ---
init_start_time = time.time()

initialization_successful = False
chroma_collection = None

db_exists = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

if db_exists:
     try:
         client = chromadb.PersistentClient(path=PERSIST_DIR)
         collection_maybe = client.get_collection(name=COLLECTION_NAME)
         if collection_maybe.count() > 0:
             chroma_collection = collection_maybe
             initialization_successful = True
     except Exception as e:
         pass

if not initialization_successful:
    with st.spinner("Initializing knowledge base... This may take a few minutes the first time."):
        chunks, doc_ids, metadatas = load_and_process_pdfs_v2()

        if chunks and doc_ids and metadatas:
            chroma_collection = setup_vector_store(chunks, doc_ids, metadatas, api_key)
            if chroma_collection:
                initialization_successful = True
        else:
            st.error("Failed to load or process PDFs. Cannot initialize chatbot.")

if not initialization_successful or not chroma_collection:
    st.error("Knowledge base initialization failed. Please check PDF files, API key, and logs.")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about bioinformatics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            context_docs, context_metadatas = get_relevant_context(prompt, chroma_collection, api_key)

            if not context_docs:
                full_response = "I couldn't find relevant information in the documents to answer your question."
            else:
                full_response = generate_response(prompt, context_docs, context_metadatas, api_key)

            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})