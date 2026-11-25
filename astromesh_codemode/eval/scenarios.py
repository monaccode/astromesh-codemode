"""Cost eval scenarios: catalog size x data size, and the fixed task each one asks.

Every scenario asks the same thing: "from ``list_orders`` (5 or 500 rows) keep the 3
with the highest amount and fetch each one's customer with ``get_customer``". The
answer comes from the real synthetic handlers (``_synthetic.py``), where order ``i``
has amount ``10 + i``, so the top 3 never tie.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

from astromesh_codemode._synthetic import synthetic_catalog

__all__ = ["DATA_ROWS", "SCENARIOS", "SYSTEM_PROMPT", "Scenario", "expected_answer", "user_task"]

DATA_ROWS: dict[str, int] = {"small": 5, "large": 500}
TOP_K = 3

SYSTEM_PROMPT = (
    "You are an operations assistant. Use the available tools to answer. "
    "Reply with the final answer only."
)


@dataclass(frozen=True)
class Scenario:
    catalog_size: int
    data: Literal["small", "large"]

    def __post_init__(self) -> None:
        # list_orders and get_customer are the 2nd and 3rd tools of the synthetic catalog.
        if self.catalog_size < 3:
            raise ValueError(
                f"catalog_size must be >= 3 to hold the task's tools, got {self.catalog_size}"
            )
        if self.data not in DATA_ROWS:
            raise ValueError(f"data must be one of {sorted(DATA_ROWS)}, got {self.data!r}")

    @property
    def rows(self) -> int:
        return DATA_ROWS[self.data]


SCENARIOS: list[Scenario] = [
    Scenario(n, d) for n, d in itertools.product((8, 64, 512), ("small", "large"))
]


def user_task(scenario: Scenario) -> str:
    """The user message, identical for both strategies."""
    return (
        f"Look at the last {scenario.rows} orders (list_orders with limit={scenario.rows}). "
        f"Take the {TOP_K} with the highest amount and, for each, fetch its customer with "
        "get_customer. Answer with order id, amount and customer name for each."
    )


async def expected_answer(scenario: Scenario) -> list[dict]:
    """The right answer, computed straight from the catalog's handlers (no model)."""
    tools = {t.name: t for t in synthetic_catalog(scenario.catalog_size)}
    orders = await tools["list_orders"].handler(limit=scenario.rows)
    top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:TOP_K]
    out = []
    for o in top:
        customer = await tools["get_customer"].handler(customer_id=o["customer_id"])
        out.append({"order_id": o["id"], "amount": o["amount"], "customer": customer["name"]})
    return out
