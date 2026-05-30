# from sentence_transformers import SentenceTransformer
# from pypdf import PdfReader
# import faiss
# import numpy as np
# from ollama import chat

# reader = PdfReader("CSDF NOTES.pdf")

# text = ""

# for page in reader.pages:
#     page_text = page.extract_text()

#     if page_text:
#         text += page_text

# chunk_size = 500

# chunks = []

# for i in range(0, len(text), chunk_size):
#     chunks.append(text[i:i + chunk_size])

# print("Chunks:", len(chunks))
# print("\nFirst Chunk:\n")
# print(chunks[0])

# model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = model.encode(chunks)

# print(embeddings.shape)


# dimension = embeddings.shape[1]

# index = faiss.IndexFlatL2(dimension)

# index.add(np.array(embeddings))

# print("Stored vectors:", index.ntotal)

# question = input("\nAsk Question: ")
# query_embedding = model.encode([question])

# D, I = index.search(query_embedding, k=3)

# print("\nRetrieved Chunks:", I)


# context = ""

# for idx in I[0]:
#     context += chunks[idx] + "\n\n" 

# print("\nContext Preview:\n")
# print(context[:1000])

# prompt = f"""
# Answer only from the context below.

# If the answer is not present, say:
# 'I could not find the answer in the document.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """


# print("Sending prompt to Qwen...")


# response = chat(
#     model="qwen2.5:3b",
#     messages=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )
# print("Response received!")

# print("\n" + "="*50)
# print("ANSWER")
# print("="*50)

# print(response["message"]["content"])
















































# import streamlit as st


# st.set_page_config(page_title="PDF RAG Chatbot")

# st.title("📚 PDF RAG Chatbot")

# st.markdown("""
# Ask questions from your uploaded PDF documents.

# **Powered by:**
# - Qwen 2.5 3B
# - FAISS
# - Sentence Transformers
# """)

# with st.sidebar:

#     st.header("📋 Project Info")

#     st.write("Local RAG Chatbot")

#     st.write("Model: Qwen 2.5 3B")
#     st.write("Vector DB: FAISS")

#     st.divider()

#     st.write("Upload one or more PDFs and ask questions.")


# from pypdf import PdfReader
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from ollama import chat


# # =========================
# # LOAD PDF
# # =========================

# uploaded_files = st.file_uploader(
#     "Upload PDF files",
#     type="pdf",
#     accept_multiple_files=True
# )

# if not uploaded_files:
#     st.stop()

# chunk_sources = []
# raw_documents = []

# for uploaded_file in uploaded_files:

#     reader = PdfReader(uploaded_file)

#     document_text = ""

#     for page in reader.pages:

#         page_text = page.extract_text()

#         if page_text:
#             document_text += page_text + "\n"

#     raw_documents.append(
#         {
#             "filename": uploaded_file.name,
#             "text": document_text
#         }
#     )


# # =========================
# # CHUNKING
# # =========================

# # =========================
# # CHUNKING WITH OVERLAP
# # =========================

# chunk_size = 500
# overlap = 100

# chunks = []

# for doc in raw_documents:

#     for i in range(0, len(doc["text"]), chunk_size - overlap):

#         chunk = doc["text"][i:i + chunk_size]

#         chunks.append(chunk)

#         chunk_sources.append(doc["filename"])

# # =========================
# # EMBEDDINGS
# # =========================

# model = SentenceTransformer("all-MiniLM-L6-v2")

# import os

# if os.path.exists("embeddings.npy"):
#     print("📂 Loading saved embeddings...")
#     embeddings = np.load("embeddings.npy")
# else:
#     print("🧠 Generating embeddings...")
#     embeddings = model.encode(chunks)
#     np.save("embeddings.npy", embeddings)

# np.save("embeddings.npy", embeddings)
# # =========================
# # VECTOR DATABASE (FAISS)
# # =========================

# dimension = embeddings.shape[1]

# index = faiss.IndexFlatL2(dimension)

# index.add(np.array(embeddings))


# # =========================
# # SYSTEM INFO
# # =========================

# print("\n" + "=" * 70)
# print("📄 PDF LOADED")
# print("=" * 70)

# print(f"Characters : {len(text)}")
# print(f"Chunks     : {len(chunks)}")
# print(f"Vectors    : {index.ntotal}")


# # =========================
# # USER QUESTION
# # =========================

# # =========================
# # CHAT LOOP
# # =========================

# question = st.chat_input("Ask a question...")
# if question:
#     st.write("Question received:", question)

# if question:

#     query_embedding = model.encode([question])

#     D, I = index.search(query_embedding, k=5)

#     context = ""

#     for idx in I[0]:
#         context += chunks[idx] + "\n\n"

# sources = set()

# for idx in I[0]:
#     sources.add(chunk_sources[idx])

# prompt = f"""

# Answer using ONLY the information present in the context.

# Do not use outside knowledge.

# If the answer is not present in the context, say:
# I could not find the answer in the document.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# with st.spinner("Searching and generating answer..."):

#     response = chat(
#         model="qwen2.5:3b",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

# with st.chat_message("assistant"):
#     st.write(response["message"]["content"])

# st.subheader("Sources")

# for source in sources:
#     st.write(f"📄 {source}")









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

**Powered by:**
- Qwen 2.5 3B
- FAISS
- Sentence Transformers
""")

with st.sidebar:
    st.header("📋 Project Info")
    st.write("Local RAG Chatbot")
    st.write("Model: Qwen 2.5 3B")
    st.write("Vector DB: FAISS")
    st.divider()
    st.write("Upload one or more PDFs and ask questions.")

# =========================
# LOAD PDF
# =========================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDF files to get started.")
    st.stop()

chunk_sources = []
raw_documents = []
total_chars = 0  # FIX: track total characters properly

for uploaded_file in uploaded_files:
    reader = PdfReader(uploaded_file)
    document_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            document_text += page_text + "\n"
    total_chars += len(document_text)  # FIX: accumulate char count
    raw_documents.append({
        "filename": uploaded_file.name,
        "text": document_text
    })

# =========================
# CHUNKING WITH OVERLAP
# =========================

chunk_size = 500
overlap = 100
chunks = []

for doc in raw_documents:
    for i in range(0, len(doc["text"]), chunk_size - overlap):
        chunk = doc["text"][i:i + chunk_size]
        chunks.append(chunk)
        chunk_sources.append(doc["filename"])

# =========================
# EMBEDDINGS
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists("embeddings.npy"):
    print("📂 Loading saved embeddings...")
    embeddings = np.load("embeddings.npy")
else:
    print("🧠 Generating embeddings...")
    embeddings = model.encode(chunks)
    np.save("embeddings.npy", embeddings)  # FIX: only save when freshly generated

# =========================
# VECTOR DATABASE (FAISS)
# =========================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =========================
# SYSTEM INFO
# =========================

print("\n" + "=" * 70)
print("📄 PDF LOADED")
print("=" * 70)
print(f"Characters : {total_chars}")  # FIX: use total_chars instead of undefined `text`
print(f"Chunks     : {len(chunks)}")
print(f"Vectors    : {index.ntotal}")

# =========================
# CHAT LOOP
# =========================

question = st.chat_input("Ask a question...")

if question:  # FIX: ALL logic below is inside this block
    st.write("**You asked:**", question)

    query_embedding = model.encode([question])
    D, I = index.search(query_embedding, k=5)

    context = ""
    for idx in I[0]:
        context += chunks[idx] + "\n\n"

    sources = set()  # FIX: moved inside `if question` block
    for idx in I[0]:
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

    with st.spinner("Searching and generating answer..."):
        response = chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}]
        )

    with st.chat_message("assistant"):
        st.write(response["message"]["content"])

    st.subheader("Sources")
    for source in sources:
        st.write(f"📄 {source}")