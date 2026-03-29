"""
TP3 - Step 6: Basic RAG pipeline (manual context injection)
Exercises 6.1 and 6.2
"""

import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

llm = ChatMistralAI(model="mistral-large-latest")

# ---------- Data ----------
query = "What is Mycobacterium kansasii?"
context = (
    "To sum up, we have presented a case of Mycobacterium kansasii monoarthritis "
    "in an immunocompetent patient successfully treated with a combination therapy "
    "including rifampicin, ethambutol and clarithromycin for 18 months. "
    "Mycobacterium kansasii is a slow-growing, photochromogenic non-tuberculous "
    "mycobacterium that can cause pulmonary and extra-pulmonary infections."
)

# =============================================================================
# STEP 6 — Basic RAG: manual context + prompt
# =============================================================================

# ---------- Base prompt (English) ----------
base_prompt_messages = [
    (
        "system",
        (
            "You are an expert in the Mycobacterium field. "
            "Answer the question using ONLY the context provided. "
            "If the context does not contain the answer, reply exactly: 'I don't know.'"  # Ex 6.2
        ),
    ),
    (
        "human",
        "Question: {query}\n\nContext:\n{context}",
    ),
]

base_prompt = ChatPromptTemplate.from_messages(base_prompt_messages)
base_chain = base_prompt | llm

print("=== Step 6: Basic RAG (English) ===")
response = base_chain.invoke({"query": query, "context": context})
print(response.content)
print()

# ---------- Ex 6.1: force French answer ----------
french_prompt_messages = [
    (
        "system",
        (
            "Tu es un expert dans le domaine des Mycobactéries. "
            "Réponds à la question en utilisant UNIQUEMENT le contexte fourni, et en français. "
            "Si le contexte ne contient pas l'information, réponds exactement : 'Je ne sais pas.'"
        ),
    ),
    (
        "human",
        "Question : {query}\n\nContexte :\n{context}",
    ),
]

french_prompt = ChatPromptTemplate.from_messages(french_prompt_messages)
french_chain = french_prompt | llm

print("=== Ex 6.1: Forced French answer ===")
response_fr = french_chain.invoke({"query": query, "context": context})
print(response_fr.content)
print()

# ---------- Ex 6.2: validation — out-of-context question ----------
out_of_scope_query = "What is the GDP of France?"
print("=== Ex 6.2: Out-of-context query (validation) ===")
print(f"Query: '{out_of_scope_query}'")
response_oos = base_chain.invoke({"query": out_of_scope_query, "context": context})
print(f"Answer: {response_oos.content}")
print(
    "\nObservation: when the context does not contain the answer the model should\n"
    "reply 'I don't know.' rather than hallucinating — the system prompt enforces this.\n"
)
