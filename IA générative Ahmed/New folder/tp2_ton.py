"""
TP2 - Step 4: Integer score + comment + aggregated average
Exercises 4.1 and 4.2
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv(override=True)

# ---------- Schema ----------
# Ex 4.1: added the comment field
class MessageTone(BaseModel):
    """Evaluation of the tone of the user's message."""

    tone_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score assigned to the message tone: 1 = neutral/cold, 5 = very friendly",
    )
    comment: str = Field(  # Ex 4.1
        ...,
        description="Brief justification for the assigned score (1 to 2 sentences maximum)",
    )


# ---------- Prompt & chain ----------
# Ex 4.1: system prompt updated to request the comment field
prompt_message = [
    (
        "system",
        (
            "You are an assistant responsible for evaluating the tone of a message provided by the user. "
            "Assign a score from 1 to 5 to the message tone, where 1 means neutral/cold and 5 means very friendly. "
            "Also provide a short comment (1 to 2 sentences) justifying your score."
        ),
    ),
    ("human", "{text}"),
]

prompt = ChatPromptTemplate.from_messages(prompt_message)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt | llm.with_structured_output(schema=MessageTone)

# ---------- Messages to evaluate ----------
messages = [
    "Hello, could you help me please?",
    "I need this immediately.",
    "Thank you so much for your precious help!",
    "What is this bug again?",              # additional — frustrated tone
    "Great work, truly impressive!",        # very positive
]

# ---------- Evaluation + data collection for aggregation (Ex 4.2) ----------
scores = []

print("=== Tone Evaluation ===")
for text in messages:
    result = chain.invoke({"text": text})
    scores.append(result.tone_score)
    print(f"Message : {text}")
    print(f"Score   : {result.tone_score}/5")
    print(f"Comment : {result.comment}")
    print()

# ---------- Ex 4.2: aggregated metric ----------
average = sum(scores) / len(scores)
print("=== Aggregated Metric (Ex 4.2) ===")
print(f"Individual scores : {scores}")
print(f"Average tone_score: {average:.2f} / 5")
