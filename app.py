import os
import streamlit as st
import requests
from dotenv import load_dotenv
from embed import unified_search

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

st.title("Sabi Biz Chat")

# User input
user_query = st.text_input("Ask me about 7Up:")

if st.button("Send") and user_query:
    # Normalize user query for better matching
    user_query = user_query.strip().lower()

    # Retrieve documents from ChromaDB (both Book1 and Book2)
    retrieved_docs = unified_search(user_query, n_results=10)

    # Combine retrieved documents into context
    context = "\n".join(retrieved_docs)

    prompt = f"""
    Answer the question based ONLY on the following context:
    {context}

    Question: {user_query}
    """

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    # Call Groq LLM
    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        st.success(answer)
    else:
        st.error(f"Error: {response.text}")