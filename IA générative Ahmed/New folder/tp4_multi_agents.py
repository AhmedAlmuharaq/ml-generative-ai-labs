"""
TP4 - Step 9: Multi-agent orchestration
Exercises 9.1, 9.2, 9.3

Requires: OPENAI_API_KEY, MISTRAL_API_KEY, TAVILY_API_KEY in .env
          pip install arxiv

Architecture:
  [Researcher agent]  ← Tavily + arXiv  →  raw documents
        ↓
  [Analyst agent]     ← Python REPL + llm-math  →  structured answer
        ↓
  [Orchestrator]      → final report

Sequential diagram (Ex 9.1):
  User ──► Orchestrator
             │
             ├──► Researcher.invoke(query)
             │         │
             │     (Tavily / arXiv calls)
             │         │
             │    researcher_output
             │
             ├──► Analyst.invoke(researcher_output)
             │         │
             │     (Python REPL / math)
             │         │
             │    analyst_output
             │
             └──► Final report (time, cost, quality comparison)
"""

import time
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# =============================================================================
# AGENT 1 — Researcher (Tavily + arXiv)
# Specialised in finding and retrieving information
# =============================================================================

researcher_llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
researcher_tools = [TavilySearchResults(max_results=3)] + load_tools(["arxiv"])
researcher_prompt = hub.pull("hwchase17/react")

researcher_agent = create_react_agent(researcher_llm, researcher_tools, researcher_prompt)
researcher_executor = AgentExecutor(
    agent=researcher_agent,
    tools=researcher_tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_execution_time=60,
)

# =============================================================================
# AGENT 2 — Analyst (Python REPL + llm-math)
# Specialised in computation and structured summarisation
# =============================================================================

analyst_llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)
analyst_tools = [PythonREPLTool()] + load_tools(["llm-math"], llm=analyst_llm)
analyst_prompt = hub.pull("hwchase17/react")

analyst_agent = create_react_agent(analyst_llm, analyst_tools, analyst_prompt)
analyst_executor = AgentExecutor(
    agent=analyst_agent,
    tools=analyst_tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_execution_time=60,
)

# =============================================================================
# ORCHESTRATOR — chains the two agents and compares results
# =============================================================================

def orchestrate(query: str) -> dict:
    """
    Run the researcher then feed its output to the analyst.
    Returns a report dict with timing, step counts, and final answer.
    """
    report = {"query": query}

    # --- Researcher phase ---
    print("\n" + "=" * 60)
    print("ORCHESTRATOR → launching Researcher agent")
    print("=" * 60)
    t0 = time.perf_counter()
    research_result = researcher_executor.invoke({
        "input": (
            f"Research the following topic thoroughly and return all key facts, "
            f"figures, and source references: {query}"
        ),
        "chat_history": [],
    })
    report["researcher_time_s"] = round(time.perf_counter() - t0, 2)
    report["researcher_steps"] = len(research_result["intermediate_steps"])
    researcher_output = research_result["output"]
    print(f"\nResearcher finished in {report['researcher_time_s']}s "
          f"({report['researcher_steps']} steps)")

    # --- Analyst phase ---
    print("\n" + "=" * 60)
    print("ORCHESTRATOR → launching Analyst agent")
    print("=" * 60)
    t0 = time.perf_counter()
    analyst_result = analyst_executor.invoke({
        "input": (
            "You are given the following research notes. Analyse them, extract "
            "the three most important numerical facts, and present a concise "
            "structured summary (3 bullet points max).\n\n"
            f"Research notes:\n{researcher_output}"
        ),
    })
    report["analyst_time_s"] = round(time.perf_counter() - t0, 2)
    report["analyst_steps"] = len(analyst_result["intermediate_steps"])
    report["final_answer"] = analyst_result["output"]
    print(f"\nAnalyst finished in {report['analyst_time_s']}s "
          f"({report['analyst_steps']} steps)")

    return report


# =============================================================================
# Ex 9.2 — Run the pipeline and compare two queries
# =============================================================================

QUERIES = [
    "What are the most recent advances in large language model quantisation?",
    "What is the current state of quantum error correction research?",
]

reports = []
for q in QUERIES:
    rep = orchestrate(q)
    reports.append(rep)

# Comparison table
print("\n" + "=" * 60)
print("COMPARISON TABLE (Ex 9.2)")
print("=" * 60)
header = f"{'Query':<55} {'R.time':>7} {'R.steps':>8} {'A.time':>7} {'A.steps':>8}"
print(header)
print("-" * len(header))
for r in reports:
    print(
        f"{r['query'][:55]:<55} {r['researcher_time_s']:>7.2f} "
        f"{r['researcher_steps']:>8} {r['analyst_time_s']:>7.2f} {r['analyst_steps']:>8}"
    )

print("\n--- Final answers ---")
for r in reports:
    print(f"\nQ: {r['query']}")
    print(f"A: {r['final_answer']}")

# =============================================================================
# Ex 9.3 (optional) — LangGraph orchestration sketch
# =============================================================================
# LangGraph lets you model this as a directed graph instead of a script.
# Uncomment after: pip install langgraph
#
# from langgraph.graph import StateGraph, END
# from typing import TypedDict
#
# class AgentState(TypedDict):
#     query: str
#     research_output: str
#     final_answer: str
#
# def researcher_node(state: AgentState) -> AgentState:
#     result = researcher_executor.invoke({"input": state["query"], "chat_history": []})
#     return {**state, "research_output": result["output"]}
#
# def analyst_node(state: AgentState) -> AgentState:
#     result = analyst_executor.invoke({
#         "input": f"Analyse and summarise:\n{state['research_output']}"
#     })
#     return {**state, "final_answer": result["output"]}
#
# graph = StateGraph(AgentState)
# graph.add_node("researcher", researcher_node)
# graph.add_node("analyst", analyst_node)
# graph.set_entry_point("researcher")
# graph.add_edge("researcher", "analyst")
# graph.add_edge("analyst", END)
# app = graph.compile()
#
# result = app.invoke({"query": QUERIES[0], "research_output": "", "final_answer": ""})
# print("LangGraph final answer:", result["final_answer"])
