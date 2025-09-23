import chromadb
from chromadb.utils import embedding_functions

# ✅ Use /tmp for persistence (works on Streamlit Cloud)
chroma_client = chromadb.PersistentClient(path="/tmp/chroma_db")

# Example embedding function (you can adjust if you’re using another one)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# Create or load your collection
collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_func
)

def add_documents(documents):
    """
    Add a list of documents to the ChromaDB collection.
    Each document should be a dict with keys: 'id', 'text', and optional 'metadata'.
    """
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)

def unified_search(query, n_results=3):
    """
    Search for the most relevant documents.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    return results