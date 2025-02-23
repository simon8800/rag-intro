import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Loading environment variables from .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Embedding function that allows us to create embeddings once we chunk our data
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# resp = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is human life expectancy in the United States?"}
#     ],
# )

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print(f"=== Loading documents from {directory_path} ===")
    documents = []
    for filename in os.listdir(directory_path):
        print(f"Loading {filename}")
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    print("=== Finished loading documents ===")
    return documents

# Function to split text into chunks
# Move overlap, more context kept; less overlap, less context kept
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the directory
directory_path = "./data/news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

# Split documents into chunks
chunked_documents = []

for doc in documents:
    chunks = split_text(doc["text"])
    print("=== Splitting docs into chunks ===")
    for i, chunk in enumerate(chunks): 
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")
    
# Function to generate embeddings using OpenAI API:
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("=== Generating embeddings ===")
    return embedding