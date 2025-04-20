import streamlit as st
import google.generativeai as genai
import pypdf
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from streamlit_local_storage import LocalStorage

# Import LangChain components for FAISS and Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

# --- Configuration ---
load_dotenv()
PDF_DIR = "data"
FAISS_INDEX_PATH = "faiss_index_bio"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_GENERATIVE_MODEL = "gemini-1.5-flash"
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
    metadatas = []
    chunk_id_counter = 0

    for doc in all_texts:
        chunks = text_splitter.split_text(doc["content"])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({"source": doc["filename"], "chunk": chunk_idx})
            chunk_id_counter += 1

    return all_chunks, metadatas

@st.cache_resource(ttl=3600)
def setup_faiss_vector_store(chunks, metadatas, api_key):
    """Creates or loads a FAISS vector store with Gemini embeddings."""
    if not chunks or not metadatas:
        st.error("No text chunks available to create vector store.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=api_key)

        if os.path.exists(FAISS_INDEX_PATH):
            try:
                vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                return vector_store
            except Exception as load_e:
                st.warning(f"Failed to load existing FAISS index ({load_e}). Rebuilding...")

        vector_store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
        vector_store.save_local(FAISS_INDEX_PATH)
        
        return vector_store

    except Exception as e:
        st.error(f"Failed to setup FAISS vector store: {e}")
        return None

def get_relevant_context(query, vector_store, api_key, n_results=5):
    """Retrieves relevant context from FAISS vector store."""
    if not vector_store:
        st.error("Vector store not initialized.")
        return [], []
    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=n_results)
        context_docs = [doc.page_content for doc, score in results_with_scores]
        context_metadatas = [doc.metadata for doc, score in results_with_scores]
        return context_docs, context_metadatas
    except Exception as e:
        st.error(f"Error retrieving context from FAISS: {e}")
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
    if st.button("Re-create Knowledge Base", help=f"Deletes the current FAISS index ({FAISS_INDEX_PATH}) and re-processes PDFs from '{PDF_DIR}'. This requires a restart."):
        with st.spinner("Attempting to delete existing FAISS index and clear cache..."):
            delete_success = False
            if os.path.exists(FAISS_INDEX_PATH):
                try:
                    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
                    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")
                    if os.path.exists(index_file):
                        os.remove(index_file)
                    if os.path.exists(pkl_file):
                        os.remove(pkl_file)
                    try:
                        os.rmdir(FAISS_INDEX_PATH)
                    except OSError:
                         pass
                    st.success(f"Deleted FAISS index files in: {FAISS_INDEX_PATH}")
                    delete_success = True
                except Exception as e:
                    st.error(f"Error deleting FAISS index files in {FAISS_INDEX_PATH}: {e}. Manual deletion might be required.")
            else:
                st.info("FAISS index not found, nothing to delete.")
                delete_success = True

            if delete_success:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cleared Streamlit cache. Please refresh the page or restart the app to rebuild the knowledge base.")
            else:
                 st.warning("Could not delete FAISS index. Cache not cleared. Please restart the app manually.")

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
faiss_vector_store = None

with st.spinner("Initializing knowledge base... This may take a few minutes the first time."):
    chunks, metadatas = load_and_process_pdfs_v2()

    if chunks and metadatas:
        faiss_vector_store = setup_faiss_vector_store(chunks, metadatas, api_key)
        if faiss_vector_store:
            initialization_successful = True
    else:
        st.error("Failed to load or process PDFs. Cannot initialize chatbot.")

if not initialization_successful or not faiss_vector_store:
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
            context_docs, context_metadatas = get_relevant_context(prompt, faiss_vector_store, api_key)

            if not context_docs:
                full_response = "I couldn't find relevant information in the documents to answer your question."
            else:
                full_response = generate_response(prompt, context_docs, context_metadatas, api_key)

            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})