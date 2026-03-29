import os
import json
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# TP1 - Step 4: Chains and structured prompts
# - Build a chain: prompt -> model -> parser
# Ex 4.1: dynamic topics + prevent empty
# Ex 4.2: JSON output variant (setup + punchline)
# Ex 4.3: compare temperature impact

if __name__ == "__main__":
    load_dotenv(override=True)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Missing MISTRAL_API_KEY. Put it in .env or set it as an environment variable.")

    model = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=api_key)
    parser = StrOutputParser()

    # ----- Base chain -----
    prompt = ChatPromptTemplate.from_template("Fais-moi une blague sur le sujet : {sujet}")
    chain = prompt | model | parser

    # ----- Ex 4.1: Dynamic topics + block empty -----
    raw = input("Enter topics (comma-separated): ").strip()
    topics = [t.strip() for t in raw.split(",") if t.strip()]

    if not topics:
        raise ValueError("Empty input is not allowed. Please type at least one topic.")

    print("\n--- Ex 4.1: Standard jokes ---")
    for sujet in topics:
        print(chain.invoke({"sujet": sujet}))
        print("-" * 10)

    # ----- Ex 4.2: JSON variant -----
    json_prompt = ChatPromptTemplate.from_template(
        'Retourne STRICTEMENT un JSON (sans texte autour, sans markdown). '
        'Le JSON doit contenir exactement deux clés: "setup" et "punchline". '
        'Sujet: {sujet}'
    )
    json_chain = json_prompt | model | parser

    print("\n--- Ex 4.2: JSON jokes ---")
    for sujet in topics:
        raw_json = json_chain.invoke({"sujet": sujet}).strip()
        # Clean up potential Markdown code blocks (common with LLMs)
        # We look for the first '{' and the last '}' to handle cases where the model adds text around the JSON
        start_idx = raw_json.find("{")
        end_idx = raw_json.rfind("}")

        if start_idx != -1 and end_idx != -1:
            raw_json = raw_json[start_idx : end_idx + 1]

        try:
            data = json.loads(raw_json)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print("WARNING: Invalid JSON returned. Raw output:")
            print(raw_json)
        print("-" * 10)

    # ----- Ex 4.3: Temperature comparison -----
    def joke_with_temperature(sujet: str, temperature: float) -> str:
        temp_model = ChatMistralAI(model="mistral-large-latest", temperature=temperature, api_key=api_key)
        temp_chain = prompt | temp_model | parser
        return temp_chain.invoke({"sujet": sujet})

    print("\n--- Ex 4.3: Temperature comparison (0 vs 0.7) ---")
    for sujet in topics:
        print(f"Sujet={sujet} | temperature=0")
        print(joke_with_temperature(sujet, 0.0))
        print()
        print(f"Sujet={sujet} | temperature=0.7")
        print(joke_with_temperature(sujet, 0.7))
        print("-" * 30)