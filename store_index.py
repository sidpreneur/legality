from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone  
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("Loading PDF data...")
extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)
print(f"Loaded {len(text_chunks)} text chunks.")

print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check if index exists
if index_name not in pc.list_indexes().names():
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # This MUST match your 'all-MiniLM-L6-v2' model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists.")

print("Uploading data to Pinecone... This may take a few minutes.")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Data upload complete.")