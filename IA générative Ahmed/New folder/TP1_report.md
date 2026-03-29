# TP1 — First steps with text generation (LangChain + Mistral)



Student: Ahmed Al-Muharaq  

Course: Machine Learning — Generative AI  

Date: 11/03/2026  

Tools: Python 3.x, LangChain, langchain-mistralai, python-dotenv



---



## 1) Understanding: LangChain & Mistral (What and why)



LangChain is a framework that helps orchestrate LLM applications by structuring prompts, chaining steps, and integrating tools (parsers, memory, retrieval, agents). Instead of writing “one prompt = one API call” everywhere, you build reusable pipelines (prompt → model → parser) and can extend to multi-step workflows (agents, RAG, LangGraph).



Mistral AI provides language models accessible via an API. In this TP, I used a Mistral chat model (mistral-large-latest) through LangChain to generate responses, measure token usage, and observe how parameters like temperature affect output variability.



---



## 2) Setup performed (Environment & security)



- Created/used a Python virtual environment (.venv) and installed required packages:

  langchain, langchain-core, langchain-mistralai, python-dotenv.

- Stored the API key in a .env file and loaded it using load_dotenv() to avoid hardcoding secrets in code.



Security note: API keys must never be pushed to GitHub or shared publicly.



---



## 3) Exercises



### Exercise 2.1 — Concrete use case for LangChain



Use case (example): An “IoT incident assistant” for a smart warehouse:

- Input: sensor alerts/logs (CO₂ spikes, vibration anomalies, leak detection).

- Chain: summarize logs → classify incident severity → propose troubleshooting steps → generate a short incident report.



LangChain helps by chaining prompt templates, parsing structured output, and later integrating retrieval (past incidents / documentation).



### Exercise 2.2 — Example from official documentation to test later



Chosen feature to test later:

- Tool calling (function calling): configuring the model to decide when to call external functions (e.g., get_weather or query_database) before answering.



Resource/page: LangChain documentation — Tool calling (tools)



---



## 4) Experiments & observations



### Step 3 — First invocation (tp1_mistral.py)



- I invoked ChatMistralAI with temperature=0 for stable, factual answers.

- I changed the question multiple times to verify consistency and correctness.



Observation: With temperature=0, answers were more deterministic. With higher temperature, outputs became more creative and variable.



### Exercise 3.2 — Temperature impact



- Prompt tested: “Write a funny 5-line story about a robot learning to cook.”

- temperature=0 → predictable structure, logical flow, standard vocabulary.

- temperature=0.7 → more diversity in wording, creative analogies, and unexpected plot twists.



### Step 4 — Chains & structured prompts (tp1_chain.py)



- Built a chain: ChatPromptTemplate → ChatMistralAI → StrOutputParser.

- Added:

  - Ex 4.1: dynamic topics + block empty strings.

  - Ex 4.2: JSON output with keys setup and punchline (+ JSON parsing).

  - Ex 4.3: compared multiple temperatures by instantiating models with different temperature values.



### Step 5 — System prompt + usage metadata



- Used a system message (role/style instruction) + user input.

- Tested style changes (concise vs humorous vs detailed) and recorded token usage.



---



## 5) Token usage & cost estimation



Pricing used (Mistral Large estimates):

- Input price per 1K tokens: $0.002 ($2 per 1M)

- Output price per 1K tokens: $0.006 ($6 per 1M)

- Source: Mistral pricing page (date checked: 11/03/2026)



Cost formula:

`cost = (input_tokens/1000) * input_price + (output_tokens/1000) * output_price`



### Exercise 5.2 — Summary table (3 requests)



| Request (short description)         | input_tokens | output_tokens | Estimated cost ($) |

|-------------------------------------|--------------|---------------|--------------------|

| 1) Simple greeting (Hello)          | 15           | 10            | 0.00009            |

| 2) Robot story (5 lines)            | 45           | 120           | 0.00081            |

| 3) JSON joke (system instruction)   | 180          | 60            | 0.00072            |



Observation: More detailed system instructions increase input_tokens, while longer requested text increases output_tokens and cost.



---



## 6) Conclusion & future improvements



This TP showed how to call Mistral models using LangChain, build reusable chains, and control output style via prompts and temperature. Next steps include:

- Using stricter structured output parsers (e.g., PydanticOutputParser) for reliable JSON validation.

- Adding RAG (vector store + embeddings) to ground answers on trusted documents.

- Exploring agents/LangGraph for multi-step reasoning and tool usage.



---



## Deliverables



- tp1_mistral.py

- tp1_chain.py (+ JSON variant)

- 1-page report
