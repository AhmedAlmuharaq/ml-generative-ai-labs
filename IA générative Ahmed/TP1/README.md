# TP1 — Getting Started with Text Generation

## Objectives
- Set up LangChain with the Mistral AI API.
- Send your first prompt and read the response.
- Build simple prompt chains using `ChatPromptTemplate` and `StrOutputParser`.
- Understand the effect of `temperature` on outputs.

## File
| File | Description |
|------|-------------|
| `tp1_mistral.ipynb` | Complete notebook: API setup, first invocation, chains, temperature comparison |

## Setup
1. Create a `.env` file in the **parent folder** (`IA générative Ahmed/`) with:
   ```
   MISTRAL_API_KEY=your_key_here
   ```
2. Install dependencies:
   ```bash
   pip install langchain langchain-mistralai python-dotenv
   ```
3. Open `tp1_mistral.ipynb` in Jupyter and run cells top to bottom.

## Key Concepts
| Concept | Description |
|---------|-------------|
| `ChatMistralAI` | LangChain wrapper for the Mistral chat API |
| `ChatPromptTemplate` | Structured prompt with variables |
| `StrOutputParser` | Converts model output to a plain string |
| `temperature` | Controls randomness: 0 = deterministic, 1 = creative |
| `usage_metadata` | Token counts returned per API call |
