"""A LangChain agent loop over a small CRM, with Code Mode instead of direct tools.

    pip install 'astromesh-codemode[langchain]' langchain-openai
    export OPENAI_API_KEY=...
    python examples/langchain_crm.py

The model gets two tools, ``crm_search`` and ``crm_run``. It searches for what it needs,
then writes one program that fetches the orders, picks the top 3 and looks up their
customers in the sandbox; only that answer comes back into the conversation. This calls
a paid API (a handful of requests to ``gpt-4o-mini``).
"""

import asyncio
import os
import sys

from astromesh_codemode import CodeMode, Limits
from astromesh_codemode.adapters.langchain import as_tools
from astromesh_codemode.sources import from_function

CUSTOMERS = {i: f"Customer {i}" for i in range(1, 51)}


def list_orders(limit: int = 10) -> list[dict]:
    """List the most recent orders. Each order is a dict with keys id, amount, customer_id."""
    return [{"id": i, "amount": 10 + i, "customer_id": i % 50 + 1} for i in range(1, limit + 1)]


def get_customer(customer_id: int) -> dict:
    """Get one customer by id. The customer is a dict with keys id and name."""
    return {"id": customer_id, "name": CUSTOMERS[customer_id]}


def list_products(category: str | None = None) -> list[dict]:
    """List products, optionally of one category."""
    return [{"sku": "A1", "category": "tools"}, {"sku": "B2", "category": "toys"}]


def delete_order(order_id: int) -> str:
    """Delete an order (a write: hidden from the model, since allow_writes is off)."""
    return f"deleted {order_id}"


TASK = (
    "Look at the last 500 orders, take the 3 with the highest amount and tell me, for each, "
    "the order id, the amount and the customer's name."
)


async def main() -> int:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("needs: pip install 'astromesh-codemode[langchain]' langchain-openai")
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("needs OPENAI_API_KEY (this example calls a real model)")
        return 2

    tools = [
        from_function(list_orders),
        from_function(get_customer),
        from_function(list_products),
        from_function(delete_order, writes=True),
    ]
    async with CodeMode(tools, name="crm", limits=Limits(timeout_s=20, max_calls=20)) as cm:
        lc_tools = as_tools(cm)
        by_name = {t.name: t for t in lc_tools}
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(lc_tools)
        messages = [
            SystemMessage("You are a CRM assistant. Use the tools to answer."),
            HumanMessage(TASK),
        ]
        for _ in range(8):  # at most 8 model calls
            ai = await model.ainvoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                print(ai.content)
                return 0
            for call in ai.tool_calls:
                print(f"--> {call['name']}({call['args']})")
                # Invoking a tool with the ToolCall returns the ToolMessage for the model.
                messages.append(await by_name[call["name"]].ainvoke(call))
                print(f"<-- {messages[-1].content[:300]}")
        print("no answer after 8 model calls")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
