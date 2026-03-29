"""
TP3 - Step 2: Bag of Words representation
TP3 - Step 4: Cosine similarity
Exercises 2.1, 2.2, 4.1, 4.2
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# =============================================================================
# STEP 2 — Bag of Words
# =============================================================================

corpus = [
    "Demonstration text, first document",
    "Demo text, and here's a second document.",
    "And finally, this is the third document.",
    # Ex 2.1: added a fourth document using a synonym of "demo" → "showcase"
    # Observation: 'showcase' appears as a NEW column — BoW does NOT know it means
    # the same thing as 'demo'. The matrix grows by one column without capturing
    # any semantic link between the two words.
    "This showcase illustrates a fourth document example.",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("=== Step 2: Bag of Words ===")
print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW matrix:\n", X.toarray())
print()

# Ex 2.2 — Two weaknesses of BoW for capturing semantics:
# 1. No synonym awareness: "demo" and "showcase" are different columns even though
#    they share the same meaning. Semantic proximity is invisible to BoW.
# 2. No word order / context: "dog bites man" and "man bites dog" produce identical
#    BoW vectors, yet their meanings are opposite.
print("Ex 2.2 — BoW limitations:")
print("  1. Synonyms treated as unrelated (e.g. 'demo' ≠ 'showcase').")
print("  2. Word order ignored ('dog bites man' == 'man bites dog' in BoW).")
print()


# =============================================================================
# STEP 4 — Cosine similarity
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Use dense BoW vectors to illustrate cosine similarity
vectors = X.toarray().astype(float)

print("=== Step 4: Cosine Similarity (BoW vectors) ===")

# Similarity between doc 0 and doc 1 (share 'demo'/'demonstration' tokens)
sim_01 = cosine_similarity(vectors[0], vectors[1])
print(f"Similarity(doc0, doc1) = {sim_01:.4f}  ← share some vocabulary")

# Ex 4.1: similarity between two very different sentences
# doc 0 ("demonstration text first document") vs doc 3 ("showcase illustrates fourth...")
sim_03 = cosine_similarity(vectors[0], vectors[3])
print(f"Similarity(doc0, doc3) = {sim_03:.4f}  ← very different vocabulary → low score")
print(
    "  Observation: when two documents share almost no words the cosine approaches 0,\n"
    "  showing no semantic relationship is captured for unrelated vocabulary.\n"
)


# Ex 4.2 — find the most similar document to a query
def find_most_similar(query: str, corpus_vectors: np.ndarray, vocab_vectorizer: CountVectorizer) -> int:
    """Return the index of the corpus document most similar to query."""
    query_vec = vocab_vectorizer.transform([query]).toarray().astype(float)[0]
    similarities = [cosine_similarity(query_vec, doc_vec) for doc_vec in corpus_vectors]
    return int(np.argmax(similarities)), similarities


query = "first demonstration document"
best_idx, scores = find_most_similar(query, vectors, vectorizer)
print(f"Ex 4.2 — Query: '{query}'")
for i, s in enumerate(scores):
    marker = " ← best match" if i == best_idx else ""
    print(f"  doc{i}: {s:.4f}{marker}")
