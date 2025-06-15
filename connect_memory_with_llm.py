from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Setup LLM (Mistral with HuggingFace)
huggingface_repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        model_kwargs = {
            "token": HF_TOKEN,
            "max_length": "517"
            }
    )
    return llm
# Step 2: Connect LLm With FAISS and create Chain