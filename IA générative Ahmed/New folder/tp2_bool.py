"""
TP2 - Step 2: Boolean structured output
Exercises 2.1 and 2.2
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv(override=True)

# ---------- Schema ----------
class Answer(BaseModel):
    answer: bool

# ---------- Prompt & chain ----------
# Ex 2.1: strengthened prompt — explicitly asks the model to pick a side even on ambiguous questions
prompt_answer = [
    (
        "system",
        (
            "You are an assistant that answers only True or False to the user's question. "
            "If the question is ambiguous, answer based on the most literal interpretation. "
            "Provide no explanation, only the boolean value."
        ),
    ),
    ("human", "{question}"),
]

prompt_answer_template = ChatPromptTemplate.from_messages(prompt_answer)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt_answer_template | llm.with_structured_output(schema=Answer)


def answer_question(question: str) -> bool:
    return chain.invoke({"question": question}).answer


# ---------- Questions ----------
questions = [
    "Christmas is in winter",                         # clearly True (northern hemisphere)
    "It rains when it doesn't rain",                  # clearly False (contradiction)
    "A square is a rectangle",                        # Ex 2.1 — ambiguous depending on math rigor
]

for question in questions:
    response = answer_question(question)
    # Ex 2.2: verifying the Python type — bool enables direct comparisons (if response:)
    print(f"Question : {question}")
    print(f"Answer   : {response}  (type: {type(response).__name__})")
    print()

# Ex 2.2 — showing how the bool type is used directly in application code
print("=== Application-level usage (Ex 2.2) ===")
for q in questions:
    if answer_question(q):
        print(f"[TRUE]   → {q}")
    else:
        print(f"[FALSE]  → {q}")
