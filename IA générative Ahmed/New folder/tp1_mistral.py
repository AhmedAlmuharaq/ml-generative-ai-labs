import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# TP1 - Step 3: First Mistral invocation
# - Load API key from .env
# - Ask a factual question with temperature=0 (deterministic)
# - Print response + usage metadata (Ex 3.1)
# - Compare with temperature=0.7 for a creative prompt (Ex 3.2)

if __name__ == "__main__":
    load_dotenv(override=True)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Missing MISTRAL_API_KEY. Put it in .env or set it as an environment variable.")

    # ----- Step 3: Basic invocation -----
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=api_key)

    questions = [
        "Quelle est la capitale de l'Albanie ?",
        "Explique TCP vs UDP en 3 phrases simples.",
        "Donne un résumé en 2 phrases de la Tour Eiffel."
    ]

    for q in questions:
        resp = llm.invoke([HumanMessage(content=q)])
        print("Q:", q)
        print("A:", resp.content)
        print("usage_metadata:", getattr(resp, "usage_metadata", None))  # Ex 3.1
        print("-" * 60)

    # ----- Ex 3.2: Temperature impact (creative prompt) -----
    llm_creative = ChatMistralAI(model="mistral-large-latest", temperature=0.7, api_key=api_key)
    creative_prompt = "Écris une mini-histoire drôle (5 lignes) sur un robot qui apprend à cuisiner."
    resp2 = llm_creative.invoke([HumanMessage(content=creative_prompt)])

    print("Creative prompt:", creative_prompt)
    print("Answer:", resp2.content)
    print("usage_metadata:", getattr(resp2, "usage_metadata", None))