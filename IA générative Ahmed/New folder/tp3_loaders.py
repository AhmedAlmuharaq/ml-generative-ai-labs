"""
TP3 - Step 10: Other document loaders and vector store comparison
Exercises 10.1 and 10.2

NOTE: YoutubeLoader requires 'youtube-transcript-api' and 'pytube' installed.
      pip install youtube-transcript-api pytube
"""

from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# STEP 10 — YouTube loader example
# =============================================================================

print("=== Step 10: YouTube Loader ===")
try:
    from langchain_community.document_loaders import YoutubeLoader

    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=YcIbZGTRMjI",
        language=["fr"],
        add_video_info=False,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from YouTube.")
    if docs:
        print(f"First 300 chars: {docs[0].page_content[:300]}")
except Exception as e:
    print(f"  YouTube loader unavailable: {e}")

print()

# =============================================================================
# Ex 10.1 — Other useful loaders
# =============================================================================

print("=== Ex 10.1: Other useful loaders ===")

LOADER_CATALOG = [
    {
        "name": "UnstructuredMarkdownLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Load .md files (docs, READMEs, wikis)",
        "example": 'UnstructuredMarkdownLoader("README.md")',
    },
    {
        "name": "BSHTMLLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Parse HTML pages (web scraping, internal tools)",
        "example": 'BSHTMLLoader("page.html")',
    },
    {
        "name": "SQLDatabaseLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Load rows from SQL databases as documents",
        "example": 'SQLDatabaseLoader(query="SELECT ...", db=db)',
    },
    {
        "name": "CSVLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Load tabular CSV data (each row becomes a document)",
        "example": 'CSVLoader("data.csv")',
    },
    {
        "name": "NotionDirectoryLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Load exported Notion pages for a knowledge base",
        "example": 'NotionDirectoryLoader("notion_export/")',
    },
    {
        "name": "GitLoader",
        "import": "langchain_community.document_loaders",
        "use_case": "Index source code from a Git repository",
        "example": 'GitLoader(repo_path="./my_repo", branch="main")',
    },
]

for loader_info in LOADER_CATALOG:
    print(f"  {loader_info['name']}")
    print(f"    Use case : {loader_info['use_case']}")
    print(f"    Example  : {loader_info['example']}")
    print()

# =============================================================================
# Ex 10.2 — FAISS vs other vector stores
# =============================================================================

print("=== Ex 10.2: Vector store comparison ===")

VECTOR_STORE_COMPARISON = [
    {
        "name": "FAISS",
        "strengths": [
            "Extremely fast local similarity search",
            "Zero infrastructure — runs in-process",
            "Well-suited for prototyping and offline pipelines",
        ],
        "limitations": [
            "No persistence out-of-the-box (must serialize manually with save_local)",
            "No metadata filtering",
            "Single machine only — not distributed",
        ],
        "when_to_use": "Prototyping, small corpora (<1M docs), offline/batch RAG",
    },
    {
        "name": "Chroma",
        "strengths": [
            "Persistent by default (DuckDB + Parquet)",
            "Metadata filtering support",
            "Easy local dev with a client-server mode",
        ],
        "limitations": [
            "Slower than FAISS for pure ANN search",
            "Still single-node for most deployments",
        ],
        "when_to_use": "Local apps that need persistence and filtering without DevOps overhead",
    },
    {
        "name": "Milvus",
        "strengths": [
            "Distributed, horizontally scalable",
            "Rich filtering (scalar + vector hybrid search)",
            "Multiple index types (IVF, HNSW, DiskANN...)",
            "Multi-vector / multi-embedding support",
        ],
        "limitations": [
            "Requires Kubernetes or Docker for production",
            "More complex to set up and maintain",
        ],
        "when_to_use": "Production systems with millions of vectors, multi-tenant SaaS, strict SLAs",
    },
    {
        "name": "Weaviate",
        "strengths": [
            "GraphQL API with semantic + keyword hybrid search",
            "Built-in vectorizer modules (OpenAI, Cohere, HuggingFace)",
            "Strong multi-tenancy and auth support",
        ],
        "limitations": [
            "Heavier infrastructure than FAISS/Chroma",
            "Managed cloud adds cost",
        ],
        "when_to_use": "Enterprise search, hybrid keyword+semantic use cases",
    },
]

for vs in VECTOR_STORE_COMPARISON:
    print(f"  [{vs['name']}]")
    print(f"    When to use : {vs['when_to_use']}")
    print(f"    Strengths   : {'; '.join(vs['strengths'])}")
    print(f"    Limitations : {'; '.join(vs['limitations'])}")
    print()
