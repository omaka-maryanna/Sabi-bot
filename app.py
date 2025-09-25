import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# -------- Load Excel data --------
def load_data(file_paths):
    dfs = []
    for path in file_paths:
        df = pd.read_excel(path)
        dfs.append(df)
    return dfs

# Convert DataFrame to readable text
def dataframe_to_text(df, title=""):
    text = f"\n{title}\n"
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        text += f"- {row_text}\n"
    return text

# -------- Groq Chat Function --------
def chat_with_groq(user_query, knowledge_text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""
You are a business assistant for Sabi Biz.
Use the following company data to answer questions.

📌 STRICT RULES:
- Only use driver phone numbers that exist in the company data below.
- Do not guess or invent phone numbers under any circumstances.
- If the data does not contain a phone number, say: 
  "The phone number for this driver is not available."
- Use general knowledge only when the question is outside the company data,
  but clearly mention when it is not in the company data.

Company Data:
{knowledge_text}
"""

    payload = {
        "model": "llama-3.1-8b-instant",  # ✅ Groq available model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    }

    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    data = resp.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    return data["choices"][0]["message"]["content"]

# -------- Streamlit UI --------
st.set_page_config(page_title="Sabi Biz Chat", page_icon="🟢", layout="wide")

# Background with 7up logo (replace URL with your logo if needed)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://upload.wikimedia.org/wikipedia/en/4/49/7-up_logo.svg");
        background-attachment: fixed;
        background-size: 200px;
        background-position: top right;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sabi Biz Chat")

# Load Excel files
book1, book2 = load_data(["Book1.xlsx", "Book2.xlsx"])

# Turn into text knowledge base
knowledge_text = dataframe_to_text(book1, "Book 1 Data") + dataframe_to_text(book2, "Book 2 Data")

# Chat input and send button
user_query = st.text_input("Ask about stock, products, or delivery:")

if st.button("Send") and user_query.strip():
    with st.spinner("Sabi Biz Chat is thinking..."):
        answer = chat_with_groq(user_query, knowledge_text)
    st.success(answer)