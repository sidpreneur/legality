import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_openai import ChatOpenAI


app = FastAPI()
load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')



embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})



chatModel = ChatOpenAI(
    model="openai/gpt-oss-safeguard-20b:groq",  # Model from your snippet
    api_key=HF_TOKEN,                         # Your HF token
    base_url="https://router.huggingface.co/v1" # The HF router
)
# ---------------------------------

# --- Define the prompt ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- Define the formatting function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Your LCEL Chain (This stays exactly the same) ---
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)
# ---------------------------------

# --- Define API ---
class ChatRequest(BaseModel):
    msg: str

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    answer = rag_chain.invoke(request.msg)
    return {"answer": answer}