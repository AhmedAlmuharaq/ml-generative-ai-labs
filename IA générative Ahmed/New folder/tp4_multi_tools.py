"""
TP4 - Step 7: Combining multiple tools (llm-math + Wikipedia + custom currency converter)
Exercises 7.1 and 7.2

Requires: OPENAI_API_KEY in .env
"""

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_core.tools import tool
from langchain_openai import OpenAI

load_dotenv(override=True)

llm = OpenAI(temperature=0)

# =============================================================================
# STEP 7 — llm-math + Wikipedia
# =============================================================================

tools_base = load_tools(["llm-math", "wikipedia"], llm=llm)
prompt = hub.pull("hwchase17/react")

agent_base = create_react_agent(llm, tools_base, prompt)
executor_base = AgentExecutor(
    agent=agent_base,
    tools=tools_base,
    handle_parsing_errors=True,
    verbose=True,
)

print("=== Step 7: llm-math + Wikipedia ===")
executor_base.invoke({"input": "What is 25% of 300?"})
print()

# =============================================================================
# Ex 7.1 — Custom currency converter tool
# =============================================================================

# Static exchange rates (relative to EUR)
EXCHANGE_RATES: dict[str, float] = {
    "EUR": 1.0,
    "USD": 1.09,
    "GBP": 0.86,
    "JPY": 163.5,
    "CHF": 0.97,
    "CAD": 1.48,
    "AUD": 1.65,
    "CNY": 7.88,
}


@tool
def convert_currency(amount_and_pair: str) -> str:
    """
    Convert a monetary amount between two currencies.

    Input format: "<amount> <FROM_CURRENCY> to <TO_CURRENCY>"
    Example: "100 USD to EUR"

    Supported currencies: EUR, USD, GBP, JPY, CHF, CAD, AUD, CNY.
    Exchange rates are static (for demonstration purposes).
    """
    try:
        parts = amount_and_pair.upper().split()
        if len(parts) != 4 or parts[2] != "TO":
            return (
                "Invalid format. Use: '<amount> <FROM> to <TO>' "
                "(e.g. '100 USD to EUR')"
            )
        amount = float(parts[0])
        from_cur = parts[1]
        to_cur = parts[3]

        if from_cur not in EXCHANGE_RATES:
            return f"Unsupported currency: {from_cur}. Supported: {list(EXCHANGE_RATES)}"
        if to_cur not in EXCHANGE_RATES:
            return f"Unsupported currency: {to_cur}. Supported: {list(EXCHANGE_RATES)}"

        amount_in_eur = amount / EXCHANGE_RATES[from_cur]
        converted = amount_in_eur * EXCHANGE_RATES[to_cur]
        return f"{amount:.2f} {from_cur} = {converted:.2f} {to_cur} (static rates)"
    except (ValueError, IndexError) as e:
        return f"Error parsing input: {e}"


# Test the tool directly
print("=== Ex 7.1: Currency converter (direct test) ===")
print(convert_currency.invoke("100 USD to EUR"))
print(convert_currency.invoke("250 GBP to JPY"))
print(convert_currency.invoke("500 USD to BTC"))  # unsupported → error
print()

# Integrate into the agent
tools_ex71 = load_tools(["llm-math", "wikipedia"], llm=llm) + [convert_currency]

# System message that forces the agent to prefer the currency tool for conversions
SYSTEM_HINT = (
    " When the question involves currency conversion, ALWAYS use the "
    "convert_currency tool instead of computing it yourself."
)

agent_ex71 = create_react_agent(llm, tools_ex71, prompt)
executor_ex71 = AgentExecutor(
    agent=agent_ex71,
    tools=tools_ex71,
    handle_parsing_errors=True,
    verbose=True,
)

print("=== Ex 7.1: Agent with currency converter ===")
executor_ex71.invoke({
    "input": (
        "Convert 500 USD to EUR using the currency tool, "
        "then tell me what 15% of that EUR amount is." + SYSTEM_HINT
    )
})

# ---------- Ex 7.2: handle_parsing_errors memo ----------
print("\n=== Ex 7.2: handle_parsing_errors ===")
print(
    "When handle_parsing_errors=True:\n"
    "  - The executor catches OutputParserException (malformed Action/Observation)\n"
    "    and feeds the error message back to the LLM so it can self-correct.\n"
    "  - This prevents the agent from crashing on a single bad output format.\n"
    "\nWhen to activate:\n"
    "  - Always in production: LLMs occasionally produce malformed ReAct lines.\n"
    "  - During development: keep it ON to see how often the model fails to format\n"
    "    correctly — high frequency signals a weak prompt or wrong model.\n"
    "\nWhen to deactivate (temporarily):\n"
    "  - During debugging: turning it OFF surfaces the raw parsing error so you\n"
    "    can diagnose and fix the prompt or the tool description.\n"
)

# Demonstrate the crash when disabled
executor_no_handle = AgentExecutor(
    agent=agent_base,
    tools=tools_base,
    handle_parsing_errors=False,
    verbose=False,
)
try:
    # Deliberately ambiguous query likely to cause a parsing issue
    executor_no_handle.invoke({"input": "???"})
except Exception as e:
    print(f"Error without handle_parsing_errors: {type(e).__name__}: {e}")
