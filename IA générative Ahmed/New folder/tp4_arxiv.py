"""
TP4 - Step 5: Specialised agent for scientific literature (arXiv)
Exercises 5.1, 5.2, 5.3

Requires: MISTRAL_API_KEY in .env
          pip install arxiv
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
tools = load_tools(["arxiv"])

# =============================================================================
# STEP 5 — arXiv agent (base)
# =============================================================================

prompt_base = hub.pull("hwchase17/react")
agent_base  = create_react_agent(llm, tools, prompt_base)
executor_base = AgentExecutor(agent=agent_base, tools=tools, verbose=True)

print("=== Step 5: arXiv agent ===")
executor_base.invoke({"input": "Summarise the paper 1605.08386 in English."})

# ---------- Ex 5.1: document the tool ----------
print("\n=== Ex 5.1: Tool documentation ===")
arxiv_tool = tools[0]
print(f"Tool name       : {arxiv_tool.name}")
print(f"Tool description: {arxiv_tool.description}")
print(
    "\nREADME excerpt:\n"
    "  Tool    : arxiv\n"
    f"  Name    : {arxiv_tool.name}\n"
    f"  Purpose : {arxiv_tool.description}\n"
    "  Usage   : Pass an arXiv paper ID (e.g. '1605.08386') or a keyword query.\n"
    "  Returns : Title, authors, abstract, and publication date of matching papers.\n"
)

# ---------- Ex 5.2: structured summary prompt ----------
print("\n=== Ex 5.2: Structured summary prompt ===")

STRUCTURED_TEMPLATE = """\
You are a scientific research assistant. Answer the following question using the tools available.
When summarising a paper, you MUST structure your response exactly as:

**Summary**: <2-3 sentence overview of the paper>
**Key Points**:
  - <point 1>
  - <point 2>
  - <point 3>
**Potential Applications**:
  - <application 1>
  - <application 2>

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

structured_prompt = PromptTemplate.from_template(STRUCTURED_TEMPLATE)
agent_structured  = create_react_agent(llm, tools, structured_prompt)
executor_structured = AgentExecutor(agent=agent_structured, tools=tools, verbose=True)

response = executor_structured.invoke({
    "input": "Summarise the paper 1605.08386 in English."
})
print("\nStructured answer:\n", response["output"])

# ---------- Ex 5.3 (optional): download the PDF from arXiv ----------
# Uncomment to enable — requires: pip install arxiv requests
#
# import arxiv
# import requests
#
# def download_arxiv_pdf(paper_id: str, output_dir: str = ".") -> Path:
#     """Download the PDF for the given arXiv paper ID and return its path."""
#     client = arxiv.Client()
#     search = arxiv.Search(id_list=[paper_id])
#     paper  = next(client.results(search))
#     out_path = Path(output_dir) / f"{paper_id.replace('/', '_')}.pdf"
#     paper.download_pdf(filename=str(out_path))
#     print(f"PDF saved to: {out_path}")
#     return out_path
#
# pdf_path = download_arxiv_pdf("1605.08386")
# # Then feed pdf_path into your FAISS RAG pipeline (see tp3_faiss.py)
