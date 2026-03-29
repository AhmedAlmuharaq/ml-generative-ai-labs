"""
TP4 - Step 8: Designing your own tools
Exercises 8.1, 8.2, 8.3

Requires: OPENAI_API_KEY in .env
"""

import datetime
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)

LOG_FILE = Path(__file__).parent / "operations.log"

# =============================================================================
# STEP 8 — Custom arithmetic tools
# =============================================================================

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers and return the result."""
    # Ex 8.1: safe int conversion
    result = int(first_int) * int(second_int)
    _log_operation("multiply", f"{first_int} * {second_int} = {result}")
    return result


@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers and return the result."""
    result = int(first_int) + int(second_int)
    _log_operation("add", f"{first_int} + {second_int} = {result}")
    return result


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Raise an integer base to an integer exponent and return the result."""
    result = int(base) ** int(exponent)
    _log_operation("exponentiate", f"{base} ^ {exponent} = {result}")
    return result


# ---------- Ex 8.2: log_to_file tool ----------
def _log_operation(op_name: str, detail: str) -> None:
    """Append an operation record to the log file."""
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {op_name}: {detail}\n")


@tool
def log_to_file(operation_description: str) -> str:
    """
    Log a custom operation description to the operations.log file.
    Use this tool to record any important step or result during a computation.
    Input: a plain-English description of the operation performed.
    """
    _log_operation("manual_log", operation_description)
    return f"Logged: '{operation_description}'"


# =============================================================================
# STEP 8 — Agent with custom tools
# =============================================================================

llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)
tools = [multiply, add, exponentiate, log_to_file]

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== Step 8: Custom tools agent ===")
executor.invoke({
    "input": (
        "Raise 3 to the power of 5, then multiply the result by the sum of "
        "twelve and three, then square the whole thing."
    )
})
print(f"\nOperations logged to: {LOG_FILE}")
print()

# ---------- Ex 8.1: type validation demo ----------
print("=== Ex 8.1: Type validation ===")
# Direct tool invocations with string inputs — int() conversion handles these
print("multiply('4', '5')   →", multiply.invoke({"first_int": "4", "second_int": "5"}))
print("exponentiate('2', '8') →", exponentiate.invoke({"base": "2", "exponent": "8"}))

# Edge case: non-numeric string
try:
    multiply.invoke({"first_int": "four", "second_int": 3})
except (ValueError, TypeError) as e:
    print(f"Invalid input error caught: {e}")
print()

# ---------- Ex 8.3 (optional): StructuredTool with explicit JSON schema ----------
print("=== Ex 8.3: StructuredTool ===")

class DivideInput(BaseModel):
    numerator: float = Field(..., description="The number to be divided")
    denominator: float = Field(..., description="The divisor (must not be zero)")


def _divide(numerator: float, denominator: float) -> float | str:
    if denominator == 0:
        return "Error: division by zero is undefined."
    result = numerator / denominator
    _log_operation("divide", f"{numerator} / {denominator} = {result}")
    return result


divide = StructuredTool.from_function(
    func=_divide,
    name="divide",
    description="Divide two numbers. Returns an error message if denominator is zero.",
    args_schema=DivideInput,
)

tools_with_divide = tools + [divide]
agent_full = create_tool_calling_agent(llm, tools_with_divide, prompt)
executor_full = AgentExecutor(agent=agent_full, tools=tools_with_divide, verbose=True)

executor_full.invoke({
    "input": "What is 144 divided by 12? Then multiply the result by 7."
})
print(f"\nAll operations logged to: {LOG_FILE}")

# Show log tail
print("\n--- Last 5 log entries ---")
with open(LOG_FILE, encoding="utf-8") as f:
    lines = f.readlines()
for line in lines[-5:]:
    print(line.rstrip())
