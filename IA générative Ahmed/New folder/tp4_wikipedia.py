"""
TP4 - Step 2: First tool — querying Wikipedia
Exercises 2.1 and 2.2
"""

from dotenv import load_dotenv
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv(override=True)

# =============================================================================
# STEP 2 — Wikipedia tool
# =============================================================================

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Basic test
print("=== Step 2: Wikipedia query ===")
result = wikipedia.run("Alan Turing")
print(result[:800])
print()

# ---------- Ex 2.1: summarise_article function (truncated to 500 chars) ----------
def summarise_article(topic: str) -> str:
    """Query Wikipedia for the given topic and return the first 500 characters."""
    raw = wikipedia.run(topic)
    return raw[:500].rstrip() + "..." if len(raw) > 500 else raw


print("=== Ex 2.1: summarise_article ===")
for topic in ["Artificial Intelligence", "FAISS vector database", "Mistral AI"]:
    print(f"Topic: {topic}")
    print(summarise_article(topic))
    print()

# ---------- Ex 2.2: advantages & limitations of a single-tool agent ----------
NOTES = """
# Single-tool agent — notes (Ex 2.2)

## Advantages
- Simple to build and debug: one clear responsibility, easy to trace failures.
- Low latency: no routing overhead, every query goes straight to the tool.
- Cost-effective: no unnecessary LLM calls to decide between tools.

## Limitations
- Can only handle tasks the single tool covers (Wikipedia ≠ real-time data).
- No fallback: if Wikipedia has no article, the agent fails silently.
- Not adaptable to multi-step tasks (research → compute → summarise).

## When to use
Use a single-tool agent for narrow, well-scoped tasks (e.g. "always look up
Wikipedia"). As soon as the user's need crosses two knowledge domains, add a
second tool or compose specialised agents.
"""

NOTES_PATH = "notes_tp4.md"
with open(NOTES_PATH, "w", encoding="utf-8") as f:
    f.write(NOTES)
print(f"Ex 2.2: notes saved to '{NOTES_PATH}'")
