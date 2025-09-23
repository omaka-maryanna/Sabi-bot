import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="sabi_biz_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

def reset_database():
    all_docs = collection.get()
    if all_docs and "ids" in all_docs and all_docs["ids"]:
        collection.delete(ids=all_docs["ids"])
        print(f"Deleted {len(all_docs['ids'])} documents from ChromaDB")
    else:
        print("ChromaDB collection is already empty")

def load_excel_to_chroma(file_paths):
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_excel(path)
        docs = df.astype(str).apply(lambda row: " | ".join(row).strip().lower(), axis=1).tolist()
        ids = [f"doc_{i}_{j}" for j in range(len(docs))]
        collection.add(documents=docs, ids=ids)
        print(f"Loaded {len(docs)} docs from {path}")

def unified_search(query, n_results=3):
    query = query.strip().lower()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results.get("documents", [[]])[0]

def get_all_documents(limit=5):
    docs = collection.get()
    if "documents" in docs and docs["documents"]:
        return docs["documents"][:limit]
    return []