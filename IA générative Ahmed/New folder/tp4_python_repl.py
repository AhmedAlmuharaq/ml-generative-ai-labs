"""
TP4 - Step 6: Developer agent — Python REPL
Exercises 6.1 and 6.2

Requires: OPENAI_API_KEY in .env
          pip install langchain-experimental
"""

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# =============================================================================
# STEP 6 — Python REPL agent
# =============================================================================

tools = [PythonREPLTool()]

INSTRUCTIONS = """\
You are an agent designed to write and execute Python code to answer questions.
You have access to a Python REPL, which you can use to execute Python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question.
You might know the answer without running any code, but you should still run
the code to get the answer.
If it does not seem like you can write code to answer the question, return
"I don't know" as the answer.
"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=INSTRUCTIONS)

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# ---------- Base executor ----------
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== Step 6: Python REPL — 1000th Fibonacci number ===")
response = executor.invoke({"input": "What is the 1000th Fibonacci number?"})
print("\nAnswer:", response["output"][:300])
print()

# ---------- Ex 6.1: timeout ----------
print("=== Ex 6.1: Executor with timeout ===")
executor_with_timeout = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_execution_time=10,   # seconds — agent stops if it exceeds this
    early_stopping_method="generate",  # let the LLM produce a partial answer
)

response_timeout = executor_with_timeout.invoke({
    "input": (
        "Compute the sum of all prime numbers below 10,000,000. "
        "This might take a while."
    )
})
print("\nAnswer (with timeout):", response_timeout["output"][:300])
print(
    "\nObservation: with max_execution_time set the executor stops after the limit\n"
    "and returns whatever the agent managed to compute. Without it the agent would\n"
    "run indefinitely on a sufficiently expensive computation.\n"
)

# ---------- Ex 6.2: guard-rail — truncate output > 200 chars ----------
MAX_DISPLAY_CHARS = 200

def safe_invoke(executor: AgentExecutor, user_input: str) -> str:
    """Invoke the executor and truncate the output if it exceeds MAX_DISPLAY_CHARS."""
    result = executor.invoke({"input": user_input})
    output = result["output"]
    if len(output) > MAX_DISPLAY_CHARS:
        output = output[:MAX_DISPLAY_CHARS].rstrip() + f"... [truncated to {MAX_DISPLAY_CHARS} chars]"
    return output


print("=== Ex 6.2: Guard-rail — output truncation ===")
long_output_query = (
    "List the first 50 prime numbers, one per line, with their index."
)
answer = safe_invoke(executor, long_output_query)
print(f"Query : {long_output_query}")
print(f"Answer: {answer}")
