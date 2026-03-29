"""
TP4 - Step 3: ReAct agent with Tavily web search
Exercises 3.1, 3.2, 3.3

Requires: TAVILY_API_KEY and MISTRAL_API_KEY in .env
"""

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv(override=True)

# =============================================================================
# STEP 3 — ReAct agent with Tavily
# =============================================================================

tavily_search = TavilySearchResults(max_results=2)
wikipedia   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# Pull the Mistral-compatible ReAct prompt from LangChain Hub
prompt = hub.pull("amalnuaimi/react-mistral")

# ---------- Base agent (Tavily only) ----------
tools_base = [tavily_search]
agent_base  = create_react_agent(llm, tools_base, prompt)
executor_base = AgentExecutor(
    agent=agent_base,
    tools=tools_base,
    verbose=True,
    return_intermediate_steps=True,  # needed for Ex 3.1 / 4.1
)

WEATHER_QUERY = (
    "Should I bring an umbrella? I am travelling to Belfort today and tomorrow."
)

print("=== Step 3: ReAct agent (Tavily only) ===")
response = executor_base.invoke({
    "input": WEATHER_QUERY,
    "chat_history": [],
})
print("\nFinal answer:", response["output"])
print()

# ---------- Ex 3.1: add Wikipedia as a second tool ----------
print("=== Ex 3.1: ReAct agent (Tavily + Wikipedia) ===")
tools_ex31 = [tavily_search, wikipedia]
agent_ex31 = create_react_agent(llm, tools_ex31, prompt)
executor_ex31 = AgentExecutor(
    agent=agent_ex31,
    tools=tools_ex31,
    verbose=True,
    return_intermediate_steps=True,
)

response_ex31 = executor_ex31.invoke({
    "input": WEATHER_QUERY,
    "chat_history": [],
})

print("\nFinal answer:", response_ex31["output"])
print("\n--- Intermediate steps (Ex 3.1 analysis) ---")
for i, (action, observation) in enumerate(response_ex31["intermediate_steps"], 1):
    print(f"Step {i}:")
    print(f"  Tool  : {action.tool}")
    print(f"  Input : {action.tool_input}")
    print(f"  Output: {str(observation)[:200]}...")
print()

# ---------- Ex 3.2: force formal English + cite sources ----------
print("=== Ex 3.2: Formal English + source citations ===")

FORMAL_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: You must always respond in formal English. "
    "At the end of your answer, list every source URL you consulted under a "
    "'Sources:' heading. If no URL is available, write 'Source: Wikipedia'."
)

response_ex32 = executor_ex31.invoke({
    "input": WEATHER_QUERY + FORMAL_PROMPT_SUFFIX,
    "chat_history": [],
})
print("\nFormal answer with sources:\n", response_ex32["output"])
print()

# ---------- Ex 3.3 (optional): swap to OpenAI GPT-4.1 ----------
# Uncomment to test — requires OPENAI_API_KEY
# from langchain_openai import ChatOpenAI
# llm_oai = ChatOpenAI(model_name="gpt-4.1", temperature=0)
# prompt_oai = hub.pull("hwchase17/react")
# agent_oai = create_react_agent(llm_oai, tools_ex31, prompt_oai)
# executor_oai = AgentExecutor(agent=agent_oai, tools=tools_ex31, verbose=True,
#                              return_intermediate_steps=True)
# resp_oai = executor_oai.invoke({"input": WEATHER_QUERY})
# print("GPT-4.1 answer:", resp_oai["output"])
