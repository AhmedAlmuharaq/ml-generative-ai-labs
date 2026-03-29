"""
TP4 - Step 4: OpenAI ReAct agent variant + cost comparison
Exercises 4.1 and 4.2

Requires: OPENAI_API_KEY and TAVILY_API_KEY in .env
"""

import time
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

search = TavilySearchResults(max_results=2)
tools  = [search]

QUERIES = [
    "Should I bring an umbrella? I am travelling to Belfort today and tomorrow.",
    "What are the latest breakthroughs in quantum computing?",
    "Who won the last FIFA World Cup and what was the final score?",
]

# =============================================================================
# STEP 4 — OpenAI ReAct variant
# =============================================================================

llm_openai = ChatOpenAI(model_name="gpt-4.1", temperature=0)
prompt_openai = hub.pull("hwchase17/react")

agent_openai = create_react_agent(llm_openai, tools, prompt_openai)
executor_openai = AgentExecutor(
    agent=agent_openai,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,  # Ex 4.1
)

print("=== Step 4: OpenAI GPT-4.1 ReAct agent ===")
response = executor_openai.invoke({"input": QUERIES[0]})
print("\nFinal answer:", response["output"])

# ---------- Ex 4.1: inspect intermediate steps (Tavily calls) ----------
print("\n=== Ex 4.1: Intermediate steps (Tavily requests) ===")
for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
    print(f"Step {i} | Tool: {action.tool} | Input: {action.tool_input}")
    print(f"  Tavily result (first 300 chars): {str(observation)[:300]}...")
print()

# ---------- Ex 4.2: cost / usage comparison table ----------
print("=== Ex 4.2: Usage metadata comparison (Mistral vs OpenAI) ===")

llm_mistral = ChatMistralAI(model="mistral-large-latest", temperature=0)
prompt_mistral = hub.pull("amalnuaimi/react-mistral")
agent_mistral = create_react_agent(llm_mistral, tools, prompt_mistral)
executor_mistral = AgentExecutor(
    agent=agent_mistral,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)

results = []

for query in QUERIES:
    row = {"query": query[:60] + "..."}

    # OpenAI run
    t0 = time.perf_counter()
    resp_oai = executor_openai.invoke({"input": query})
    row["openai_time_s"] = round(time.perf_counter() - t0, 2)
    # usage_metadata lives on the raw LLM response inside intermediate_steps
    row["openai_output_len"] = len(resp_oai["output"])

    # Mistral run
    t0 = time.perf_counter()
    resp_mis = executor_mistral.invoke({"input": query, "chat_history": []})
    row["mistral_time_s"] = round(time.perf_counter() - t0, 2)
    row["mistral_output_len"] = len(resp_mis["output"])

    results.append(row)

# Print comparison table
header = f"{'Query':<65} {'OAI time':>9} {'OAI len':>8} {'MIS time':>9} {'MIS len':>8}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['query']:<65} {r['openai_time_s']:>9.2f} {r['openai_output_len']:>8} "
        f"{r['mistral_time_s']:>9.2f} {r['mistral_output_len']:>8}"
    )
print(
    "\nNote: precise token costs require reading usage_metadata from each LLM call.\n"
    "OpenAI: resp.usage_metadata['input_tokens'] / ['output_tokens']\n"
    "Mistral: same field name — check their pricing pages for $/1M token rates.\n"
)
