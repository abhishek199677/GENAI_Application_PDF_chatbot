import pdfplumber
import streamlit as st
import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import requests
import os
from dotenv import load_dotenv


# Load environment variables first
load_dotenv()  # Add this line

EURI_API_KEY = os.getenv("EURI_API_KEY")
EURI_CHAT_URL= "https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_EMBED_URL= "https://api.euron.one/api/v1/euri/alpha/embeddings"

conversation_memory = []

#extracting the ext from the pdfs
def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


#splitting the text into chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start= 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# passing the entire chunks into euri embedding api

def get_euri_embeddings(texts):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": texts,
        "model": "text-embedding-3-small"
    }
    res = requests.post(EURI_EMBED_URL, headers=headers, json=payload)
    response_json = res.json()
    print("API Response:", response_json)  # Debug
    if 'data' not in response_json:
        raise RuntimeError(f"Embedding API error: {response_json.get('error', 'Unknown error')}")
    # Access 'embedding' key instead of 'embeddings'
    return np.array([d["embedding"] for d in response_json["data"]])


#storing data into vectordb
def build_vector_store(chunks):
    embeddings = get_euri_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Corrected line
    index.add(embeddings)
    return index, embeddings

#converting the questions into numericals and retrieving the context 
def retrieve_context(question, chunks, index, embeddings, top_k=3):
    q_embed = get_euri_embeddings([question])[0]
    D, I = index.search(np.array([q_embed]), top_k)
    return "\n\n".join([chunks[i] for i in I[0]])


def ask_euri_with_context(question, context, memory=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions from a document."}
    ]
    if memory:
        messages.extend(memory)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    reply = res.json()['choices'][0]['message']['content']
    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": reply})
    return reply

# Streamlit UI
st.title("📄 PDF Knowledge Extraction RAG Bot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_question = st.text_input("Ask a question about the document")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    full_text = extract_text_from_pdf("temp.pdf")
    chunks = split_text(full_text)
    index, embeddings = build_vector_store(chunks)

    st.success("PDF loaded and indexed.")

    if user_question:
        context = retrieve_context(user_question, chunks, index, embeddings)
        response = ask_euri_with_context(user_question, context, conversation_memory)
        print("🧠 Conversation Memory:", conversation_memory)
        st.markdown("### ✅ Answer:")
        st.write(response)
        