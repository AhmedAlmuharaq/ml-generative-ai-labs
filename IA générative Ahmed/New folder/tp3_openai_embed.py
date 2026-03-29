"""
TP3 - Step 5: OpenAI embeddings + cosine comparison
Exercises 5.1 and 5.2

NOTE: OpenAI API calls are billed. Set a usage quota in your dashboard before running.
      Requires OPENAI_API_KEY in your .env file.
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

# =============================================================================
# Shared cosine similarity helper
# =============================================================================

def cosine_similarity(a: np.ndarray | list, b: np.ndarray | list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# STEP 5 — OpenAI embeddings
# =============================================================================

client = OpenAI()

SENTENCE_A = "What is Mycobacterium kansasii?"
SENTENCE_B = (
    "To sum up, we have presented a case of Mycobacterium kansasii monoarthritis "
    "in an immunocompetent patient successfully treated with a combination therapy."
)


def embed_openai(text: str, model: str = "text-embedding-3-large", dimensions: int = 3072) -> list[float]:
    """Return an OpenAI embedding vector for the given text."""
    return (
        client.embeddings.create(input=[text], model=model, dimensions=dimensions)
        .data[0]
        .embedding
    )


print("=== Step 5: OpenAI Embeddings ===")

# Full-dimension embedding (3072)
vec_a_3072 = embed_openai(SENTENCE_A, dimensions=3072)
vec_b_3072 = embed_openai(SENTENCE_B, dimensions=3072)
sim_openai_full = cosine_similarity(vec_a_3072, vec_b_3072)

print(f"Sentence A: '{SENTENCE_A}'")
print(f"Sentence B: '{SENTENCE_B[:60]}...'")
print(f"Cosine similarity (OpenAI 3072 dims): {sim_openai_full:.4f}")
print()

# ---------- Ex 5.1: compare with sentence-transformers ----------
print("=== Ex 5.1: Comparison with sentence-transformers ===")

st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb_a_st = st_model.encode(SENTENCE_A)
emb_b_st = st_model.encode(SENTENCE_B)
sim_st = cosine_similarity(emb_a_st, emb_b_st)

print(f"  sentence-transformers (384 dims): {sim_st:.4f}")
print(f"  OpenAI text-embedding-3-large (3072 dims): {sim_openai_full:.4f}")
print(
    "  Observation: OpenAI's large model typically scores higher on domain-specific\n"
    "  texts because it was trained on a much larger and more diverse corpus,\n"
    "  and its higher dimensionality captures finer semantic nuances.\n"
)

# ---------- Ex 5.2: reduce dimensions to 256 ----------
print("=== Ex 5.2: Reduced dimensions (256) ===")

vec_a_256 = embed_openai(SENTENCE_A, dimensions=256)
vec_b_256 = embed_openai(SENTENCE_B, dimensions=256)
sim_openai_256 = cosine_similarity(vec_a_256, vec_b_256)

print(f"  Cosine similarity (OpenAI 256 dims) : {sim_openai_256:.4f}")
print(f"  Cosine similarity (OpenAI 3072 dims): {sim_openai_full:.4f}")
print(
    "  Observation: reducing to 256 dimensions via Matryoshka representation\n"
    "  (the technique OpenAI uses) preserves most of the semantic signal for\n"
    "  common sentence pairs. Quality degrades mainly on edge cases requiring\n"
    "  fine-grained distinction. Smaller vectors also cut storage and retrieval cost.\n"
)
