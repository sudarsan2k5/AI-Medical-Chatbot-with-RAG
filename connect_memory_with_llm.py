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
DB_FAISS_PATH = 'vectorstore/bd_faiss'

custom_prompt_templet = """
    Use the pieces of information provided in the context to answer user's question.
    If you don'd know the answer, just say that you don't know, don't try u[ an answer.
    Don't provide anything out of the given context

    contect: {context}
    Question: {question}

    Start the answer directly. no small talk please
"""

def set_custom_prompt(custom_prompt_templet):
    prompt = PromptTemplate(template = custom_prompt_templet, input_variables = ["context", "question"])
    return prompt