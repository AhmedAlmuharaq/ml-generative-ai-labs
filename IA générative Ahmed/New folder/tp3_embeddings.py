"""
TP3 - Step 3: Contextual embeddings with Sentence Transformers
Exercises 3.1 and 3.2
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# STEP 3 — Sentence Transformers embeddings
# =============================================================================

sentences = [
    "This is an example sentence.",
    "Each sentence is converted into a fixed-sized vector.",
]

# ---------- Base model ----------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embeddings = model.encode(sentences)

print("=== Step 3: Sentence Transformers ===")
for sentence, embedding in zip(sentences, embeddings):
    print(f"{sentence!r}")
    print(f"  → first 3 dims: {embedding[:3]}  |  vector size: {len(embedding)}")
print()

# ---------- Ex 3.1: compare several models and their vector sizes ----------
# Ex 3.1: uncomment a model below to test it (downloads on first run)
MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",    # 384 dims  — fast, small
    "sentence-transformers/all-MiniLM-L12-v2",   # 384 dims  — deeper, same output size
    "sentence-transformers/all-distilroberta-v1", # 768 dims  — larger representation
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384 dims — multilingual
]

print("=== Ex 3.1: Model comparison ===")
for model_name in MODELS_TO_COMPARE:
    m = SentenceTransformer(model_name, device="cpu")
    emb = m.encode(sentences[0])
    print(f"  {model_name}")
    print(f"    Vector size : {len(emb)}")
    print(f"    First 3 dims: {emb[:3]}")
print()

# ---------- Ex 3.2: why force device="cpu"? ----------
# On machines without a compatible CUDA GPU (most student laptops), PyTorch will
# raise a RuntimeError or silently fall back if you don't specify device="cpu".
# By setting device="cpu" explicitly:
#   - The code is reproducible across any machine regardless of GPU availability.
#   - You avoid cryptic CUDA-related errors (CUDA not available, cuDNN mismatch...).
# On a machine WITH a GPU you would use device="cuda" to accelerate inference.
print("=== Ex 3.2: Why device='cpu'? ===")
print(
    "  Explicitly setting device='cpu' prevents RuntimeError on machines without a\n"
    "  CUDA-compatible GPU. Without this flag, SentenceTransformer tries 'cuda' first\n"
    "  and crashes if no NVIDIA GPU / CUDA driver is found.\n"
    "  On a GPU machine you would set device='cuda' for faster encoding.\n"
)

# Uncomment below to observe the warning/error on a CPU-only machine:
# model_no_device = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# print(model_no_device.encode(sentences[0])[:3])

# =============================================================================
# Cosine similarity helper (reused in other TP3 files)
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Quick sanity check on the two base embeddings
sim = cosine_similarity(embeddings[0], embeddings[1])
print(f"Cosine similarity between the two example sentences: {sim:.4f}")
