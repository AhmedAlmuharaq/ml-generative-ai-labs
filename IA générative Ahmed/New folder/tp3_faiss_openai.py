"""
TP3 - Step 8: FAISS with OpenAI embeddings + local cache
Exercises 8.1 and 8.2

NOTE: OpenAI API calls are billed. Set a usage quota in your dashboard.
      Requires OPENAI_API_KEY in .env.
      Place 'Guyeux_2024.pdf' in the 'images/' folder next to this script.
"""

import os
import time
import warnings
from pathlib import Path
from textwrap import fill, shorten

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Ex 8.2: LangChain in-memory cache to avoid redundant API calls
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

warnings.simplefilter("ignore")
load_dotenv(override=True)

# ---------- Ex 8.2: enable SQLite cache (avoids re-embedding identical texts) ----------
CACHE_PATH = Path(__file__).parent / ".langchain_cache.db"
set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))
print(f"Cache enabled → {CACHE_PATH}")

# ---------- PDF ----------
PDF_PATH = Path(__file__).parent / "images" / "Guyeux_2024.pdf"
if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found at '{PDF_PATH}'.\n"
        "Please place 'Guyeux_2024.pdf' in the 'images/' folder next to this script."
    )

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load_and_split()
print(f"Loaded {len(pages)} pages.\n")

query = "Is there a lineage 10 in M.tuberculosis?"

# =============================================================================
# Ex 8.1 — Compare indexation time: HuggingFace vs OpenAI
# =============================================================================

print("=== Ex 8.1: Indexation time comparison ===")

# HuggingFace (local, free)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
t0 = time.perf_counter()
faiss_hf = FAISS.from_documents(pages, hf_embeddings)
t_hf = time.perf_counter() - t0
print(f"  HuggingFace (local) : {t_hf:.2f}s")

# OpenAI (remote, billed)
openai_embeddings = OpenAIEmbeddings()  # uses text-embedding-ada-002 by default
t0 = time.perf_counter()
faiss_openai = FAISS.from_documents(pages, openai_embeddings)
t_openai = time.perf_counter() - t0
print(f"  OpenAI (remote)     : {t_openai:.2f}s")
print(
    "  Observation: HuggingFace is typically faster for small corpora (no network\n"
    "  overhead). OpenAI may be faster for large batches due to parallelized server\n"
    "  inference, but adds cost and latency variability.\n"
)

# =============================================================================
# STEP 8 — Similarity search with OpenAI FAISS index
# =============================================================================

print("=== Step 8: OpenAI FAISS similarity search ===")
docs = faiss_openai.similarity_search(query, k=2)
for doc in docs:
    print(f"Page {doc.metadata.get('page', '?')}: {fill(shorten(doc.page_content, 500), 80)}\n")

# =============================================================================
# Ex 8.2 — Cache demonstration
# =============================================================================
print("=== Ex 8.2: Cache in action ===")
print("Re-indexing with OpenAI (second call — should be faster if embeddings are cached)...")
t0 = time.perf_counter()
faiss_openai_2 = FAISS.from_documents(pages, openai_embeddings)
t_cached = time.perf_counter() - t0
print(f"  Second indexation time: {t_cached:.2f}s")
print(
    "  Note: LangChain's SQLiteCache caches LLM completions, not embedding calls.\n"
    "  To cache embeddings specifically, use CacheBackedEmbeddings from langchain.\n"
    "  See: langchain_community.embeddings.CacheBackedEmbeddings\n"
)
