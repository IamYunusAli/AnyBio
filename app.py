import streamlit as st
import google.generativeai as genai
import pypdf
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from streamlit_local_storage import LocalStorage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

# --- Configuration ---
load_dotenv()
PDF_DIR = "data"
FAISS_INDEX_PATH = "vectordb"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_GENERATIVE_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# --- Language Configuration ---
LANGUAGES = {
    "en": "English",
    "es": "Español",
    "am": "አማርኛ", 
    "ar": "العربية"  
}

TEXTS = {
    "en": {
        "page_title": "Bioinformatics Chat",
        "page_icon": "🧬",
        "title": "🧬 Know more About Bioinformatics",
        "caption": "Ask me any questions about bioinformatics.",
        "config_header": "Configuration",
        "api_key_label": "Enter Google AI API Key",
        "api_key_help": "Get your key from Google AI Studio. Stored locally in your browser.",
        "clear_history_button": "Clear Chat History",
        "history_cleared_success": "Chat history cleared.",
        "db_management_header": "Database Management",
        "recreate_db_button": "Re-create Knowledge Base",
        "recreate_db_help": "Deletes the current FAISS index ({}) and re-processes PDFs from '{}'. This requires a restart.",
        "deleting_db_spinner": "Attempting to delete existing FAISS index and clear cache...",
        "deleted_db_success": "Deleted FAISS index files in: {}",
        "delete_db_error": "Error deleting FAISS index files in {}: {}. Manual deletion might be required.",
        "db_not_found_info": "FAISS index not found, nothing to delete.",
        "cleared_cache_success": "Cleared Streamlit cache. Please refresh the page or restart the app to rebuild the knowledge base.",
        "delete_db_warning": "Could not delete FAISS index. Cache not cleared. Please restart the app manually.",
        "api_key_needed_info": "Please enter your Google AI API Key in the sidebar to begin.",
        "invalid_api_key_error": "Invalid API Key or configuration error. Please check the key in the sidebar. Error: {}",
        "init_spinner": "Initializing knowledge base... This may take a few minutes the first time.",
        "pdf_load_error": "Failed to load or process PDFs. Cannot initialize chatbot.",
        "init_fail_error": "Knowledge base initialization failed. Please check PDF files, API key, and logs.",
        "chat_input_placeholder": "Ask a question about bioinformatics...",
        "thinking_spinner": "Thinking...",
        "no_context_response": "I couldn't find relevant information in the documents to answer your question.",
        "response_error": "Sorry, I encountered an error while generating the response.",
        "language_header": "Language / Idioma / ቋንቋ / اللغة",
        "sources": "Sources",
        "settings_button": "⚙️ Settings"
    },
    "es": {
        "page_title": "Chat de Bioinformática",
        "page_icon": "🧬",
        "title": "🧬 Aprende más sobre Bioinformática",
        "caption": "Hazme cualquier pregunta sobre bioinformática.",
        "config_header": "Configuración",
        "api_key_label": "Introduce la Clave API de Google AI",
        "api_key_help": "Obtén tu clave de Google AI Studio. Se guarda localmente en tu navegador.",
        "clear_history_button": "Limpiar Historial de Chat",
        "history_cleared_success": "Historial de chat limpiado.",
        "db_management_header": "Gestión de Base de Datos",
        "recreate_db_button": "Recrear Base de Conocimiento",
        "recreate_db_help": "Elimina el índice FAISS actual ({}) y reprocesa los PDFs de '{}'. Requiere reiniciar.",
        "deleting_db_spinner": "Intentando eliminar el índice FAISS existente y limpiar caché...",
        "deleted_db_success": "Archivos de índice FAISS eliminados en: {}",
        "delete_db_error": "Error al eliminar archivos de índice FAISS en {}: {}. Puede requerir eliminación manual.",
        "db_not_found_info": "Índice FAISS no encontrado, nada que eliminar.",
        "cleared_cache_success": "Caché de Streamlit limpiada. Por favor, actualiza la página o reinicia la app para reconstruir la base de conocimiento.",
        "delete_db_warning": "No se pudo eliminar el índice FAISS. Caché no limpiada. Por favor, reinicia la app manualmente.",
        "api_key_needed_info": "Por favor, introduce tu Clave API de Google AI en la barra lateral para comenzar.",
        "invalid_api_key_error": "Clave API inválida o error de configuración. Por favor, revisa la clave en la barra lateral. Error: {}",
        "init_spinner": "Inicializando base de conocimiento... Esto puede tardar unos minutos la primera vez.",
        "pdf_load_error": "Fallo al cargar o procesar PDFs. No se puede inicializar el chatbot.",
        "init_fail_error": "Fallo en la inicialización de la base de conocimiento. Por favor, revisa los archivos PDF, la clave API y los logs.",
        "chat_input_placeholder": "Haz una pregunta sobre bioinformática...",
        "thinking_spinner": "Pensando...",
        "no_context_response": "No pude encontrar información relevante en los documentos para responder tu pregunta.",
        "response_error": "Lo siento, encontré un error al generar la respuesta.",
        "language_header": "Idioma / Language / ቋንቋ / اللغة",
        "sources": "Fuentes",
        "settings_button": "⚙️ Configuración"
    },
    "am": {
        "page_title": "ባዮኢንፎርማቲክስ ውይይት",
        "page_icon": "🧬",
        "title": "🧬 ስለ ባዮኢንፎርማቲክስ የበለጠ ይወቁ",
        "caption": "ስለ ባዮኢንፎርማቲክስ ማንኛውንም ጥያቄ ይጠይቁኝ።",
        "config_header": "ማዋቀር",
        "api_key_label": "የGoogle AI ኤፒአይ ቁልፍ ያስገቡ",
        "api_key_help": "ቁልፍዎን ከGoogle AI Studio ያግኙ። በአሳሽዎ ውስጥ በአካባቢው ይቀመጣል።",
        "clear_history_button": "የውይይት ታሪክ አጽዳ",
        "history_cleared_success": "የውይይት ታሪክ ጸድቷል።",
        "db_management_header": "የውሂብ ጎታ አስተዳደር",
        "recreate_db_button": "የእውቀት መሰረትን እንደገና ፍጠር",
        "recreate_db_help": "የአሁኑን የFAISS መረጃ ጠቋሚ ({}) ይሰርዛል እና ፒዲኤፎችን ከ '{}' እንደገና ያስኬዳል። እንደገና ማስጀመር ያስፈልገዋል።",
        "deleting_db_spinner": "ያለውን የFAISS መረጃ ጠቋሚ ለመሰረዝ እና መሸጎጫ ለማጽዳት በመሞከር ላይ...",
        "deleted_db_success": "የFAISS መረጃ ጠቋሚ ፋይሎች ተሰርዘዋል በ: {}",
        "delete_db_error": "የFAISS መረጃ ጠቋሚ ፋይሎችን በ {} ለመሰረዝ ስህተት፡ {}. በእጅ መሰረዝ ሊያስፈልግ ይችላል።",
        "db_not_found_info": "የFAISS መረጃ ጠቋሚ አልተገኘም፣ የሚሰረዝ ምንም ነገር የለም።",
        "cleared_cache_success": "የStreamlit መሸጎጫ ጸድቷል። እባክዎ የእውቀት መሰረቱን እንደገና ለመገንባት ገጹን ያድሱ ወይም መተግበሪያውን እንደገና ያስጀምሩ።",
        "delete_db_warning": "የFAISS መረጃ ጠቋሚን መሰረዝ አልተቻለም። መሸጎጫ አልጸዳም። እባክዎ መተግበሪያውን በእጅ እንደገና ያስጀምሩ።",
        "api_key_needed_info": "ለመጀመር እባክዎ የGoogle AI ኤፒአይ ቁልፍዎን በጎን አሞሌው ውስጥ ያስገቡ።",
        "invalid_api_key_error": "ልክ ያልሆነ የኤፒአይ ቁልፍ ወይም የማዋቀር ስህተት። እባክዎ በጎን አሞሌው ውስጥ ያለውን ቁልፍ ያረጋግጡ። ስህተት፡ {}",
        "init_spinner": "የእውቀት መሰረትን በማስጀመር ላይ... ይህ ለመጀመሪያ ጊዜ ጥቂት ደቂቃዎችን ሊወስድ ይችላል።",
        "pdf_load_error": "ፒዲኤፎችን መጫን ወይም ማስኬድ አልተሳካም። ቻትቦትን ማስጀመር አይቻልም።",
        "init_fail_error": "የእውቀት መሰረት ማስጀመር አልተሳካም። እባክዎ የፒዲኤፍ ፋይሎችን፣ የኤፒአይ ቁልፍን እና ምዝግብ ማስታወሻዎችን ያረጋግጡ።",
        "chat_input_placeholder": "ስለ ባዮኢንፎርማቲክስ ጥያቄ ይጠይቁ...",
        "thinking_spinner": "በማሰብ ላይ...",
        "no_context_response": "ለጥያቄዎ መልስ ለመስጠት በሰነዶቹ ውስጥ ተዛማጅ መረጃ ማግኘት አልቻልኩም።",
        "response_error": "ይቅርታ፣ ምላሹን በማመንጨት ላይ ሳለ ስህተት አጋጥሞኛል።",
        "language_header": "ቋንቋ / Language / Idioma / اللغة",
        "sources": "ምንጮች",
        "settings_button": "⚙️ ማዋቀር"
    },
    "ar": {
        "page_title": "دردشة المعلوماتية الحيوية",
        "page_icon": "🧬",
        "title": "🧬 اعرف المزيد عن المعلوماتية الحيوية",
        "caption": "اسألني أي أسئلة حول المعلوماتية الحيوية.",
        "config_header": "الإعدادات",
        "api_key_label": "أدخل مفتاح Google AI API",
        "api_key_help": "احصل على مفتاحك من Google AI Studio. يتم تخزينه محليًا في متصفحك.",
        "clear_history_button": "مسح سجل الدردشة",
        "history_cleared_success": "تم مسح سجل الدردشة.",
        "db_management_header": "إدارة قاعدة البيانات",
        "recreate_db_button": "إعادة إنشاء قاعدة المعرفة",
        "recreate_db_help": "يحذف فهرس FAISS الحالي ({}) ويعيد معالجة ملفات PDF من '{}'. يتطلب إعادة التشغيل.",
        "deleting_db_spinner": "جاري محاولة حذف فهرس FAISS الحالي ومسح ذاكرة التخزين المؤقت...",
        "deleted_db_success": "تم حذف ملفات فهرس FAISS في: {}",
        "delete_db_error": "خطأ في حذف ملفات فهرس FAISS في {}: {}. قد يتطلب الحذف اليدوي.",
        "db_not_found_info": "لم يتم العثور على فهرس FAISS، لا يوجد شيء لحذفه.",
        "cleared_cache_success": "تم مسح ذاكرة التخزين المؤقت لـ Streamlit. يرجى تحديث الصفحة أو إعادة تشغيل التطبيق لإعادة بناء قاعدة المعرفة.",
        "delete_db_warning": "تعذر حذف فهرس FAISS. لم يتم مسح ذاكرة التخزين المؤقت. يرجى إعادة تشغيل التطبيق يدويًا.",
        "api_key_needed_info": "الرجاء إدخال مفتاح Google AI API الخاص بك في الشريط الجانبي للبدء.",
        "invalid_api_key_error": "مفتاح API غير صالح أو خطأ في التكوين. يرجى التحقق من المفتاح في الشريط الجانبي. الخطأ: {}",
        "init_spinner": "جاري تهيئة قاعدة المعرفة... قد يستغرق هذا بضع دقائق في المرة الأولى.",
        "pdf_load_error": "فشل تحميل أو معالجة ملفات PDF. لا يمكن تهيئة روبوت الدردشة.",
        "init_fail_error": "فشل تهيئة قاعدة المعرفة. يرجى التحقق من ملفات PDF ومفتاح API والسجلات.",
        "chat_input_placeholder": "اطرح سؤالاً حول المعلوماتية الحيوية...",
        "thinking_spinner": "جارٍ التفكير...",
        "no_context_response": "لم أتمكن من العثور على معلومات ذات صلة في المستندات للإجابة على سؤالك.",
        "response_error": "عذرًا، واجهت خطأ أثناء إنشاء الرد.",
        "language_header": "اللغة / Language / Idioma / ቋንቋ",
        "sources": "المصادر",
        "settings_button": "⚙️ الإعدادات"
    }
}

# --- Helper Functions ---

@st.cache_data
def load_and_process_pdfs():
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

def generate_response(query, context_docs, context_metadatas, api_key, language="en"):
    """Generates response using Gemini based on query, context, and language."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_GENERATIVE_MODEL)

        context_string = ""
        sources = set()
        for doc, meta in zip(context_docs, context_metadatas):
            source_file = meta.get('source', 'Unknown source')
            context_string += f"Source: {source_file}\nContent:\n{doc}\n\n---\n\n"
            sources.add(source_file)

        language_name = LANGUAGES.get(language, "English") 
        prompt = f"""You are a helpful assistant knowledgeable in bioinformatics, answering questions based on the provided text snippets.
Directly answer the question using *only* the information available and summarize it with a good way to help people understand it better from the text snippets below.
Do not use any prior knowledge But you can use prior knowledge to support your summerization.
If the answer is not found in the snippets, state that the what you are asking for or mentioning is not in my knowledge databse.
**Please provide the answer in {language_name}.**

Context from documents:
{context_string}

Question:
{query}

Answer ({language_name}):
"""

        response = model.generate_content(prompt)
        response_text = response.text
        return response_text

    except Exception as e:
        st.error(f"{TEXTS[language]['response_error']}: {e}")
        return TEXTS[language]['response_error']
    

# --- Streamlit App ---

# Initialize session state for language and settings visibility
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# Get current language texts
current_texts = TEXTS[st.session_state.language]

# Use language strings for page config, title, caption
st.set_page_config(page_title=current_texts["page_title"], page_icon=current_texts["page_icon"])
st.title(current_texts["title"])
st.caption(current_texts["caption"])

localS = LocalStorage()

# --- Sidebar ---
with st.sidebar:

    if st.button(current_texts["settings_button"]):
        st.session_state.show_settings = not st.session_state.show_settings

    # Conditionally display settings
    if st.session_state.show_settings:
    
        st.header(current_texts["language_header"])
        selected_lang_code = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda code: LANGUAGES[code],
            key="lang_select",
            index=list(LANGUAGES.keys()).index(st.session_state.language)
        )
        # Update session state if language changed
        if selected_lang_code != st.session_state.language:
            st.session_state.language = selected_lang_code
            st.rerun()

        st.divider()

        # API Key Configuration
        st.header(current_texts["config_header"])
        api_key_from_storage = localS.getItem("gemini_api_key")
        api_key_input = st.text_input(
            current_texts["api_key_label"],
            type="password",
            key="api_key_input_field",
            help=current_texts["api_key_help"],
            value=api_key_from_storage if api_key_from_storage else ""
        )
        if api_key_input:
            localS.setItem("gemini_api_key", api_key_input)
        api_key = api_key_input if api_key_input else (api_key_from_storage if api_key_from_storage else None)

        st.divider()

    # --- Always Visible Sidebar Items ---

    # Retrieve API key if settings are hidden (needed for initialization check later)
    if not st.session_state.show_settings:
         api_key_from_storage = localS.getItem("gemini_api_key")
         api_key = api_key_from_storage if api_key_from_storage else None

    # Clear Chat History Button
    if st.button(current_texts["clear_history_button"]):
        st.session_state.messages = []
        st.success(current_texts["history_cleared_success"])
    
    # Database Management Section
    st.subheader(current_texts["db_management_header"])
    if st.button(current_texts["recreate_db_button"], help=current_texts["recreate_db_help"].format(FAISS_INDEX_PATH, PDF_DIR)):
        with st.spinner(current_texts["deleting_db_spinner"]):
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
                    st.success(current_texts["deleted_db_success"].format(FAISS_INDEX_PATH))
                    delete_success = True
                except Exception as e:
                    st.error(current_texts["delete_db_error"].format(FAISS_INDEX_PATH, e))
            else:
                st.info(current_texts["db_not_found_info"])
                delete_success = True

            if delete_success:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(current_texts["cleared_cache_success"])
            else:
                 st.warning(current_texts["delete_db_warning"])


# --- API Key Handling & Initialization ---

if not api_key:
    st.info(current_texts["api_key_needed_info"])
    st.info("Click the ⚙️ Settings button in the sidebar to enter your API key.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    list(genai.list_models())
except Exception as e:
    st.error(current_texts["invalid_api_key_error"].format(e))
    st.stop()

# --- Initialization ---
init_start_time = time.time()

initialization_successful = False
faiss_vector_store = None


with st.spinner(current_texts["init_spinner"]):
    chunks, metadatas = load_and_process_pdfs()
    if chunks and metadatas:
        faiss_vector_store = setup_faiss_vector_store(chunks, metadatas, api_key)
        if faiss_vector_store:
            initialization_successful = True
    else:
        st.error(current_texts["pdf_load_error"])


# Use translated error message
if not initialization_successful or not faiss_vector_store:
    st.error(current_texts["init_fail_error"])
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Use translated placeholder
if prompt := st.chat_input(current_texts["chat_input_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(current_texts["thinking_spinner"]):
            current_lang = st.session_state.language
            context_docs, context_metadatas = get_relevant_context(prompt, faiss_vector_store, api_key)

            if not context_docs:
                # Use translated response
                full_response = current_texts["no_context_response"]
            else:
                # Pass language to generate_response
                full_response = generate_response(prompt, context_docs, context_metadatas, api_key, current_lang)

            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})