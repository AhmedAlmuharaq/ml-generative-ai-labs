"""
TP3 - Step 7: FAISS vector store with HuggingFace embeddings
TP3 - Step 9: Text splitting before indexation
Exercises 7.1, 7.2, 9.1, 9.2

NOTE: Place the file 'Guyeux_2024.pdf' inside an 'images/' subfolder
      next to this script before running:
      IA générative Ahmed/images/Guyeux_2024.pdf
"""

import os
import time
import warnings
from pathlib import Path
from textwrap import fill, shorten

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.simplefilter("ignore")
load_dotenv(override=True)

# ---------- Path to PDF ----------
PDF_PATH = Path(__file__).parent / "images" / "Guyeux_2024.pdf"
if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found at '{PDF_PATH}'.\n"
        "Please place 'Guyeux_2024.pdf' in the 'images/' folder next to this script."
    )

# ---------- Embeddings model ----------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =============================================================================
# STEP 7 — FAISS vector store (raw pages, no splitting)
# =============================================================================

print("=== Step 7: FAISS with raw pages ===")
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load_and_split()
print(f"  Loaded {len(pages)} pages from PDF.")

t0 = time.time()
faiss_index = FAISS.from_documents(pages, embeddings_model)
print(f"  Indexation time (raw pages): {time.time() - t0:.2f}s")

# Base query
query_base = "Is there a lineage 10 in M.tuberculosis?"
docs = faiss_index.similarity_search(query_base, k=2)

print(f"\nQuery: '{query_base}'")
for doc in docs:
    print(f"  Page {doc.metadata.get('page', '?')}: {fill(shorten(doc.page_content, 500), 80)}\n")

# ---------- Ex 7.1: geographic location of a lineage ----------
query_geo = "What is the geographic location of lineage 2 of M.tuberculosis?"
docs_geo = faiss_index.similarity_search(query_geo, k=2)

print(f"Ex 7.1 — Query: '{query_geo}'")
for doc in docs_geo:
    print(f"  Page {doc.metadata.get('page', '?')}: {fill(shorten(doc.page_content, 500), 80)}\n")

# ---------- Ex 7.2: full RAG pipeline (retrieve → format prompt → LLM) ----------
print("=== Ex 7.2: Full RAG pipeline ===")

llm = ChatMistralAI(model="mistral-large-latest")

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a scientific expert in mycobacteriology. "
            "Answer the question using ONLY the context provided. "
            "If the information is not in the context, reply: 'I don't know.'"
        ),
    ),
    ("human", "Question: {query}\n\nContext:\n{context}"),
])

rag_chain = rag_prompt | llm


def rag_answer(query: str, index: FAISS, k: int = 3) -> str:
    retrieved = index.similarity_search(query, k=k)
    context = "\n\n".join(d.page_content for d in retrieved)
    return rag_chain.invoke({"query": query, "context": context}).content


answer = rag_answer(query_base, faiss_index)
print(f"Query : {query_base}")
print(f"Answer: {answer}\n")

# =============================================================================
# STEP 9 — Text splitting before indexation
# =============================================================================

print("=== Step 9: Text splitting ===")

# Ex 9.1: test different chunk_size / chunk_overlap pairs
SPLIT_CONFIGS = [
    {"chunk_size": 1000, "chunk_overlap": 200},  # default — balanced
    {"chunk_size": 500,  "chunk_overlap": 100},  # smaller chunks — more granular retrieval
    {"chunk_size": 300,  "chunk_overlap": 50},   # very small — risk of cutting mid-sentence
]

for cfg in SPLIT_CONFIGS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        keep_separator=False,
        separators=["\n\n", "\n", ". "],
    )
    chunks = splitter.split_documents(pages)
    print(f"  chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']} → {len(chunks)} chunks")

print()

# Ex 9.2: re-index FAISS with split chunks (chunk_size=500 is a good default)
print("=== Ex 9.2: FAISS with split chunks ===")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    keep_separator=False,
    separators=["\n\n", "\n", ". "],
)
split_docs = splitter.split_documents(pages)
print(f"  {len(split_docs)} chunks created from {len(pages)} pages.")

t0 = time.time()
faiss_split_index = FAISS.from_documents(split_docs, embeddings_model)
print(f"  Indexation time (split chunks): {time.time() - t0:.2f}s")

answer_split = rag_answer(query_base, faiss_split_index)
print(f"\nQuery (split index) : {query_base}")
print(f"Answer              : {answer_split}")
print(
    "\nObservation: smaller chunks improve retrieval precision (less irrelevant text\n"
    "per chunk) but may lose context if a concept spans multiple chunks. Overlapping\n"
    "mitigates boundary cuts. Tune based on your document structure.\n"
)
