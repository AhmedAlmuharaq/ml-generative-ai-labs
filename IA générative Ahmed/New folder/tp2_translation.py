"""
TP2 - Step 5: Enforcing a usable JSON format via structured output
Exercises 5.1 and 5.2
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv(override=True)

# ---------- Supported language pairs ----------
# Ex 5.2: define a set of supported languages to validate against
SUPPORTED_LANGUAGES = {
    "english", "french", "spanish", "german", "italian",
    "portuguese", "dutch", "arabic", "chinese", "japanese",
}

# ---------- Schema ----------
class Translation(BaseModel):
    original_text: str = Field(..., description="The original text before translation")
    original_language: str = Field(..., description="The language of the original text")
    translated_text: str = Field(..., description="The translated text")
    translated_language: str = Field(..., description="The language into which the text was translated")


# ---------- Translation function ----------
def translate(
    text: str,
    source_language: str = "french",
    target_language: str = "english",
    return_full_object: bool = False,
) -> str | Translation:
    """
    Translate text from source_language to target_language.

    Ex 5.1: set return_full_object=True to get the full Translation object
            instead of just the translated string.
    Ex 5.2: raises ValueError if target_language is not in SUPPORTED_LANGUAGES.
    """
    # Ex 5.2: validate language pair
    if target_language.lower() not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported target language: '{target_language}'. "
            f"Supported languages are: {sorted(SUPPORTED_LANGUAGES)}"
        )
    if source_language.lower() not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported source language: '{source_language}'. "
            f"Supported languages are: {sorted(SUPPORTED_LANGUAGES)}"
        )

    llm = ChatMistralAI(model="mistral-medium-latest")
    prompt = ChatPromptTemplate.from_template(
        "Please translate the following text from {source_language} to {target_language}. "
        "Your translation must be accurate, fluent, and natural, perfectly preserving the original meaning.\n\n"
        "Text to translate:\n----\n{text}"
    )

    # Ex 5.1: without RunnableLambda we get the full Translation object
    structured_chain = prompt | llm.with_structured_output(Translation)

    if return_full_object:
        # Ex 5.1: returns the complete Translation object — all fields are accessible
        return structured_chain.invoke({
            "source_language": source_language,
            "target_language": target_language,
            "text": text,
        })

    extract_translation = RunnableLambda(lambda t: t.translated_text)
    chain = structured_chain | extract_translation
    return chain.invoke({
        "source_language": source_language,
        "target_language": target_language,
        "text": text,
    })


# ---------- Demo ----------
print("=== Basic translation (string output) ===")
print(translate("What is the capital of Albania?", source_language="english", target_language="french"))
print()

print("=== Ex 5.1: Full Translation object (no RunnableLambda) ===")
full = translate("What is the capital of Albania?", source_language="english", target_language="french", return_full_object=True)
print(f"Original   : {full.original_text}")
print(f"From       : {full.original_language}")
print(f"Translated : {full.translated_text}")
print(f"To         : {full.translated_language}")
print()

print("=== Ex 5.2: Other language pairs ===")
pairs = [
    ("The sun rises in the east.", "english", "spanish"),
    ("Bonjour tout le monde.", "french", "german"),
]
for text, src, tgt in pairs:
    result = translate(text, source_language=src, target_language=tgt)
    print(f"[{src} → {tgt}] {text}")
    print(f"           → {result}")
    print()

print("=== Ex 5.2: Unsupported language raises ValueError ===")
try:
    translate("Hello", source_language="english", target_language="klingon")
except ValueError as e:
    print(f"ValueError caught: {e}")
