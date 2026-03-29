"""
TP2 - Step 3: Choosing a predefined action
Exercises 3.1 and 3.2
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv(override=True)

# ---------- Available actions ----------
# Ex 3.1: added a third action
tasks = [
    "Answer a new question",
    "Provide more details about the previous question",
    "Ask for clarification",  # Ex 3.1
]


# ---------- Schema ----------
class NextTask(BaseModel):
    """Always use this tool to structure your response to the user."""

    action: str = Field(
        ...,
        enum=tasks,
        description="The next action to take",
    )


# ---------- Prompt & chain ----------
prompt_message = [
    (
        "system",
        (
            "You are an assistant responsible for classifying a user's request into one of a small set of "
            "actions to take as a chatbot. You must determine the next action to perform. "
            "If the user's intent is unclear or lacks context, choose 'Ask for clarification'."
        ),
    ),
    ("human", "{text}"),
]

prompt = ChatPromptTemplate.from_messages(prompt_message)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt | llm.with_structured_output(schema=NextTask)

# ---------- Tests ----------
messages = [
    "Can you tell me more?",                              # → Provide more details
    "What are PPVs?",                                     # → Answer a new question
    "I'm not sure... it's something I saw somewhere",     # Ex 3.1 → Ask for clarification
]

print("=== Action classification ===")
for text in messages:
    result = chain.invoke({"text": text})
    print(f"Message : {text}")
    print(f"Action  : {result.action}")
    print()

# ---------- Ex 3.2: robustness vs. free-text response ----------
print("=== Discussion Ex 3.2 ===")
print(
    "With a structured schema, the LLM is FORCED to pick from the defined actions.\n"
    "Hallucinations (e.g. 'Transfer to a human agent') are impossible because Pydantic\n"
    "validates that the value is in the enum — any out-of-list value raises a ValidationError.\n"
    "With a free-text response, the model could invent any phrasing,\n"
    "making parsing brittle and application code unstable."
)
