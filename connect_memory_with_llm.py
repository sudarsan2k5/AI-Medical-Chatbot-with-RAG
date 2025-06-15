from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

# HF_TOKEN = os.getenv('HF_TOKEN')

load_dotenv()

# Step 1: Setup LLM (Mistral with HuggingFace)
# huggingface_repo_id = 'gpt2'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_llm():
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        temperature=0.5,
        
        max_tokens=517
    )

# Step 2: Connect LLm With FAISS and create Chain

system_prompt = (
    "Use the pieces of information provided in the context to answer user's question. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Don't provide anything out of the given context.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# load from the database
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(
    model='sentence-transformers/all-mpnet-base-v2'
)
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
    )

question_answer_chain = create_stuff_documents_chain(
    load_llm(),
    prompt
)

chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={'k': 4}),
    question_answer_chain
)

# Now invoke with a single query
user_query = input('Write Query Here: ')

response = chain.invoke({"input": user_query})
print(response["answer"])