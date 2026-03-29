# TP3 — RAG Systems and Vector Databases

## Objectives
- Represent text as vectors (BoW, Transformers, OpenAI embeddings).
- Measure semantic similarity with cosine similarity.
- Build a RAG pipeline with FAISS and LangChain.
- Apply text splitting and explore document loaders.

## File
| File | Description |
|------|-------------|
| `tp3_rag_vectors.ipynb` | All steps and exercises in one notebook |

## Setup
```bash
pip install scikit-learn numpy sentence-transformers langchain langchain-community
pip install langchain-mistralai langchain-huggingface langchain-openai faiss-cpu
pip install pypdf
```

Place `Guyeux_2024.pdf` in an `images/` folder next to the notebook to run Steps 7–9.

## Steps covered
| Step | Topic | Exercises |
|------|-------|-----------|
| 2 | Bag of Words | 2.1 (synonym issue), 2.2 (limitations) |
| 3 | Sentence Transformers | 3.1 (model comparison), 3.2 (device=cpu) |
| 4 | Cosine similarity | 4.1 (different sentences), 4.2 (nearest neighbour) |
| 5 | OpenAI embeddings | 5.1 (comparison), 5.2 (dimension reduction) |
| 6 | Basic RAG | 6.1 (French), 6.2 (out-of-context guard) |
| 7+9 | FAISS + text splitting | 7.1 (geo query), 7.2 (full RAG), 9.1–9.2 (chunks) |
| 10 | Loaders & vector stores | 10.1 (loaders), 10.2 (FAISS vs Chroma vs Milvus) |
