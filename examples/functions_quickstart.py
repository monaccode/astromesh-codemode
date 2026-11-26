"""Code Mode over plain Python functions: no framework, no model, no network.

    uv run python examples/functions_quickstart.py

It plays the model's part by hand: search for tools, then send one program to ``run``.
The 500 orders stay in the sandbox; only the top 3 come back as text.
"""

import asyncio

from astromesh_codemode import CodeMode, Limits
from astromesh_codemode.sources import from_function

CUSTOMERS = {i: f"Customer {i}" for i in range(1, 51)}


def list_orders(limit: int = 10) -> list[dict]:
    """List the most recent orders, each with id, amount and customer_id."""
    return [{"id": i, "amount": 10 + i, "customer_id": i % 50 + 1} for i in range(1, limit + 1)]


async def get_customer(customer_id: int) -> dict:
    """Get one customer by id: id and name."""
    return {"id": customer_id, "name": CUSTOMERS[customer_id]}


def delete_order(order_id: int) -> str:
    """Delete an order. A write: hidden unless the CodeMode is built with allow_writes=True."""
    return f"deleted {order_id}"


PROGRAM = """
import asyncio
orders = await list_orders(limit=500)
top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:3]
customers = await asyncio.gather(*[get_customer(customer_id=o["customer_id"]) for o in top])
[{"order": o["id"], "amount": o["amount"], "customer": c["name"]} for o, c in zip(top, customers)]
"""


async def main() -> None:
    tools = [
        from_function(list_orders),
        from_function(get_customer),
        from_function(delete_order, writes=True),
    ]
    async with CodeMode(tools, limits=Limits(timeout_s=10, max_calls=20)) as cm:
        # The two tools a model would get: codemode_search and codemode_run.
        print([t.name for t in cm.tool_definitions()])

        # 1. The model searches; it gets typed Python stubs (delete_order is not shown).
        print(await cm.search("orders customer"))

        # 2. The model sends a program; the host runs it in the Monty sandbox.
        text, trace = await cm.run_traced(PROGRAM)
        print(text)
        print(f"{len(trace.calls)} tool calls in {trace.ms:.0f} ms, limit hit: {trace.limit_hit}")

        # 3. Mistakes come back as short text the model can read and fix.
        print(await cm.run("await delete_order(order_id=1)"))
        print(await cm.run("open('/etc/passwd').read()"))


if __name__ == "__main__":
    asyncio.run(main())
