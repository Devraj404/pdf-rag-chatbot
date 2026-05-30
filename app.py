# import streamlit as st

# st.set_page_config(page_title="PDF RAG Chatbot")

# st.title("📚 PDF RAG Chatbot")

# st.markdown(
#     """
#     Ask questions from your uploaded PDF documents.
#     Answers are generated using local Qwen + FAISS retrieval.
#     """
# )

# uploaded_files = st.file_uploader(
#     "Upload PDF files",
#     type="pdf",
#     accept_multiple_files=True
# )

# if uploaded_files:
#     st.success(f"{len(uploaded_files)} PDF(s) uploaded")
#     for file in uploaded_files:
#         st.write("📄", file.name)
    
#     question = st.chat_input("Ask a question about your PDFs...")

import streamlit as st
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from ollama import chat

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📚 PDF RAG Chatbot")

st.markdown("""
Ask questions from your uploaded PDF documents.
Answers are generated using local Qwen + FAISS retrieval.
""")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDF files to get started.")
    st.stop()

st.success(f"{len(uploaded_files)} PDF(s) uploaded")
for file in uploaded_files:
    st.write("📄", file.name)

# =========================
# LOAD + CHUNK
# =========================

chunk_sources = []
chunks = []

for uploaded_file in uploaded_files:
    reader = PdfReader(uploaded_file)
    document_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            document_text += page_text + "\n"

    chunk_size = 500
    overlap = 100
    for i in range(0, len(document_text), chunk_size - overlap):
        chunk = document_text[i:i + chunk_size]
        chunks.append(chunk)
        chunk_sources.append(uploaded_file.name)

# =========================
# EMBEDDINGS + FAISS
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
print("Chunks:", len(chunks))
print("Embeddings:", len(embeddings))
index.add(np.array(embeddings))

# =========================
# CHAT
# =========================

question = st.chat_input("Ask a question about your PDFs...")

if question:
    st.write("**You asked:**", question)

    query_embedding = model.encode([question])
    D, I = index.search(query_embedding, k=5)

    context = ""
    sources = set()
    for idx in I[0]:
        context += chunks[idx] + "\n\n"
        sources.add(chunk_sources[idx])

    prompt = f"""
Answer using ONLY the information present in the context.
Do not use outside knowledge.
If the answer is not present in the context, say:
I could not find the answer in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner("Generating answer..."):
        response = chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}]
        )

    with st.chat_message("assistant"):
        st.write(response["message"]["content"])

    st.subheader("📚 Sources")
    for source in sources:
        st.write(f"📄 {source}")