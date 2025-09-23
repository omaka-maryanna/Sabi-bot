import os

# 🚑 Disable Chroma telemetry & system restrictions for Streamlit
os.environ["CHROMA_API_IMPL"] = "chromadb.api.local.LocalAPI"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from chromadb.utils import embedding_functions

# ✅ In-memory only (safe for Streamlit Cloud)
chroma_client = chromadb.Client()

embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_func
)

def add_documents(documents):
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)

def unified_search(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results