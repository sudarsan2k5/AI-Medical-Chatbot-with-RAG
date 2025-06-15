from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load Raw PDF(f)
DATA_PATH = 'data/'
# ---------- Load PDF --------
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader
                             )
    documents = loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
print(len(documents))
# ---------- Load PDF --------


# Step 2: Create Chunks

# ------------- PDF Chunking -----------------
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

chunks = create_chunks(documents)
print(len(chunks))

# ------------- PDF Chunking -----------------


# Step 3: Create Vector Embeddings

def get_emmbedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-mpnet-base-v2'
    )
    return embedding_model
embedding_model = get_emmbedding_model()
# Step 4: Store Embedding in FAISS
