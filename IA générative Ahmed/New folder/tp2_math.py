"""
TP2 - Step 6: Structuring step-by-step reasoning
Exercises 6.1 and 6.2
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv(override=True)

# ---------- Schema ----------
class Step(BaseModel):
    explanation: str
    output: str


# Ex 6.1: added verifications field to force the model to check its own solution
class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str
    verifications: list[str]  # Ex 6.1 — e.g. "Substituting x=-3.625 back: 8*(-3.625)+31 = 2 ✓"


# ---------- Prompt & chain ----------
# Ex 6.1: system prompt updated to request verifications
prompt_answer = [
    (
        "system",
        (
            "You are a very pedagogical mathematics teacher. "
            "When solving an equation, show each step clearly. "
            "After reaching the final answer, also provide a list of verifications "
            "that confirm your solution is correct (e.g. substitute back into the equation)."
        ),
    ),
    ("human", "{exercise}"),
]

prompt_answer_template = ChatPromptTemplate.from_messages(prompt_answer)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt_answer_template | llm.with_structured_output(schema=MathResponse)


def solve(exercise: str) -> None:
    result = chain.invoke({"exercise": exercise})

    print(f"Exercise: {exercise}")
    print("--- Steps ---")
    for step in result.steps:
        print(f"  • {step.explanation}")
        print(f"    Result: {step.output}")

    print(f"\nFinal answer: {result.final_answer}")

    # Ex 6.1 — display verifications
    print("--- Verifications (Ex 6.1) ---")
    for v in result.verifications:
        print(f"  ✓ {v}")
    print()


# ---------- Exercises ----------
exercises = [
    "Solve: 8x + 31 = 2",                     # original exercise
    "Solve: 3x² - 12 = 0",                    # Ex 6.2 — non-linear (quadratic)
    "Solve: x/2 + 3/4 = 7/4",                 # Ex 6.2 — fractions
]

for ex in exercises:
    solve(ex)

# ---------- Ex 6.2: Discussion ----------
print("=== Limitations identified (Ex 6.2) ===")
print(
    "1. Non-linear equations (e.g. quadratics): the model may produce incomplete or\n"
    "   imprecise steps, especially when symbolic simplification is needed.\n"
    "2. Complex fractions: the model can make arithmetic errors with nested fractions.\n"
    "3. The schema enforces structure but cannot enforce mathematical correctness —\n"
    "   the verifications field helps, but the model may still verify incorrectly.\n"
    "4. For reliable math, a dedicated symbolic engine (e.g. SymPy) should be used\n"
    "   alongside the LLM rather than relying on it exclusively."
)
