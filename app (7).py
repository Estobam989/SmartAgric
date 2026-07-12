
import sys
import os
import streamlit as st
import tempfile
import shutil 

from langchain_huggingface import ChatHuggingFace 
                            import HuggingFaceEmbeddings
                            import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --- 1. Environment Variables ---
# Prefer Streamlit secrets for production deployment for HUGGINGFACE_API_KEY
# For local testing, it falls back to config.py
try:
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError:
    HUGGINGFACE_API_KEY = config.HUGGINGFACE_API_KEY
    st.warning("HUGGINGFACE_API_KEY not found in Streamlit secrets. Using config.py. Please set it in Streamlit secrets for deployment.")

MODEL_NAME = config.MODEL_NAME
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS

if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY == "hf_YOUR_HUGGINGFACE_API_TOKEN":
    st.error("HUGGINGFACE_API_KEY not found or is still a placeholder. Please set it in your Streamlit secrets or config.py.")
    st.stop()

# Set HF_TOKEN environment variable for HuggingFaceEndpoint
os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY

# --- 2. LLM Initialization ---
hf_llm = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    # huggingface_api_token=HUGGINGFACE_API_KEY, # Removed direct token passing, relying on HF_TOKEN env var
    temperature=TEMPERATURE,
    max_new_tokens=MAX_TOKENS # Use max_new_tokens for HuggingFaceEndpoint
)
llm = ChatHuggingFace(llm=hf_llm)

# --- 3. Embeddings Initialization ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. Define Tools ---
# Farm Profit Calculator
def calculate_profit(cost, revenue):
    return revenue - cost

# Web Search Tool
search = DuckDuckGoSearchRun()

# --- 5. Document Loading, Splitting, and Vector Store (RAG Setup) ---
VECTOR_STORE_PATH = "faiss_index"
VECTOR_STORE_NAME = "index"

# Initialize vectorstore and retriever
vectorstore = None
retriever = None

# Streamlit UI for Sidebar and PDF upload
with st.sidebar:
    st.title("🌾 AgriSmart AI")
    st.markdown("---")
    st.write("✔ AI Chat")
    st.write("✔ PDF Knowledge")
    st.write("✔ Calculator")
    st.write("✔ Web Search")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Agricultural PDF",
        type="pdf"
    )

documents = []
docs = []

if uploaded_file is not None:
    # Save uploaded file to a temporary location for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        st.sidebar.success(f"Loaded {len(documents)} pages from uploaded PDF.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded PDF: {e}")
    finally:
        os.remove(temp_file_path) # Clean up temporary file

if documents: # If documents were loaded from uploaded PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    if docs:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(folder_path=VECTOR_STORE_PATH, index_name=VECTOR_STORE_NAME)
        st.sidebar.success("Vector store created from uploaded PDF.")
        retriever = vectorstore.as_retriever()
    else:
        st.sidebar.warning("No chunks created from uploaded PDF.")
elif os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH): # Load existing vector store
    try:
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            index_name=VECTOR_STORE_NAME,
            embeddings=embeddings,
            allow_dangerous_deserialization=True # Required for FAISS.load_local with custom embeddings
        )
        st.sidebar.info("Loaded existing FAISS vector store.")
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.sidebar.error(f"Error loading existing FAISS vector store: {e}")
else: # Fallback to mock document if no PDF uploaded and no existing vector store
    mock_content = """
Agriculture is the backbone of many economies, providing food, fiber, and raw materials.
Sustainable agriculture practices are crucial for long-term ecological balance and food security.
These practices include crop rotation, organic farming, water conservation, and reduced pesticide use.
Precision agriculture, leveraging technologies like IoT and AI, is transforming farming by optimizing resource allocation and improving yields.
Key crops include wheat, corn, rice, and soybeans, which are staple foods globally.s
Livestock farming also plays a significant role, contributing to dairy, meat, and wool production.
Challenges in agriculture include climate change, soil degradation, and pest resistance, necessitating continuous innovation and research.
    """
    mock_doc = Document(page_content=mock_content, metadata={'source': 'mock_document.txt', 'page': 1})
    documents = [mock_doc]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    if docs:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(folder_path=VECTOR_STORE_PATH, index_name=VECTOR_STORE_NAME)
        st.sidebar.info("Using mock data to create vector store as no PDF was uploaded or or existing vector store was found.")
        retriever = vectorstore.as_retriever()
    else:
        st.sidebar.error("Failed to create vector store even with mock data.")

# --- 6. Conversation Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 7. Chat Interface ---
st.title("🌱 AgriSmart AI")
prompt = st.chat_input("Ask me anything about farming...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tool Routing Logic
    response_content = ""
    if "price" in prompt.lower():
        response_content = search.run(prompt)
    elif "profit" in prompt.lower():
        # Example hardcoded profit calculation, adjust inputs as needed
        response_content = f"The estimated profit is: ${calculate_profit(500000, 850000):,.2f}"
    elif retriever and ("agriculture" in prompt.lower() or "farm" in prompt.lower() or "crop" in prompt.lower() or "soil" in prompt.lower() or "irrigation" in prompt.lower()):
        # Use RAG if retriever is available and prompt is related to agricultural topics
        retrieved_docs = retriever.invoke(prompt)
        context = " ".join([d.page_content for d in retrieved_docs])
        # A simple prompt template for RAG
        rag_prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        response_content = llm.invoke(rag_prompt).content
    else:
        # Default to LLM without specific tools/RAG
        response_content = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": response_content})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
