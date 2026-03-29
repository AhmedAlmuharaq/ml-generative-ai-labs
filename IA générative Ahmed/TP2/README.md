# TP2 — Structured Outputs with LangChain and Mistral

## Objectives
- Understand why structured outputs make LLM responses reliable.
- Define Pydantic schemas and use `with_structured_output`.
- Explore formats: boolean, enum action, integer score, translation, step-by-step reasoning.

## File
| File | Description |
|------|-------------|
| `tp2_structured_outputs.ipynb` | All steps and exercises in one notebook |

## Setup
```bash
pip install langchain langchain-mistralai pydantic python-dotenv
```
Place `MISTRAL_API_KEY=your_key` in `../.env`.

## Steps covered
| Step | Topic | Exercises |
|------|-------|-----------|
| 2 | Boolean output | 2.1 (ambiguous question), 2.2 (type verification) |
| 3 | Enum action selection | 3.1 (third action), 3.2 (hallucination robustness) |
| 4 | Integer score + comment | 4.1 (comment field), 4.2 (aggregated average) |
| 5 | Structured translation | 5.1 (full object), 5.2 (language validation) |
| 6 | Math step-by-step | 6.1 (verifications), 6.2 (non-linear limits) |
