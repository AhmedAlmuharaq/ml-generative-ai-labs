# TP4 — Agent Orchestration with LangChain

## Objectives
- Understand LLM-driven agents and their executor feedback loop.
- Install and configure tools: Wikipedia, Tavily, arXiv, Python REPL.
- Create and orchestrate specialised agents for real use cases.
- Design your own tools and compose multi-agent pipelines.

## File
| File | Description |
|------|-------------|
| `tp4_agents.ipynb` | Complete notebook: all steps and exercises |

## Setup
```bash
pip install langchain langchain-community langchain-openai langchain-mistralai
pip install wikipedia langchain-tavily arxiv python-dotenv langchain-experimental
```

Add to `../.env`:
```
TAVILY_API_KEY=your_key
MISTRAL_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Steps covered
| Step | Topic | Exercises |
|------|-------|-----------|
| 2 | Wikipedia tool | 2.1 (summarise_article), 2.2 (mono-tool limits) |
| 3 | ReAct + Tavily | 3.1 (Wikipedia second tool), 3.2 (formal prompt + sources) |
| 4 | OpenAI ReAct | 4.1 (intermediate steps), 4.2 (cost comparison table) |
| 5 | arXiv agent | 5.1 (tool docs), 5.2 (structured summary) |
| 6 | Python REPL | 6.1 (timeout), 6.2 (output guard-rail) |
| 7 | Multi-tools | 7.1 (currency converter), 7.2 (handle_parsing_errors) |
| 8 | Custom tools | 8.1 (type validation), 8.2 (log file), 8.3 (StructuredTool) |
| 9 | Multi-agent | 9.1–9.2 (researcher + analyst pipeline) |
