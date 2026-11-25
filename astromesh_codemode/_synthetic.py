"""Deterministic synthetic tool catalogs, shared by tests and the cost eval."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.spec import ToolSpec


def _const(value: Any):
    async def handler(**_: Any) -> Any:
        return value

    return handler


async def _get_invoice(invoice_id: str, **_: Any) -> dict:
    return {"id": invoice_id, "total": 120.5, "currency": "USD", "lines": 3}


async def _list_orders(limit: int = 10, **_: Any) -> list[dict]:
    return [{"id": i, "customer_id": i % 7 + 1, "amount": 10 + i} for i in range(limit)]


async def _get_customer(customer_id: int, **_: Any) -> dict:
    return {"id": customer_id, "name": f"Customer {customer_id}", "tier": "gold"}


async def _find_contact(customer_email: str, **_: Any) -> dict:
    return {"id": 1, "found": True}


def _core() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="billing_get_invoice",
            description="Get an invoice by id, with its total, tax and line items.",
            input_schema={
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
            handler=_get_invoice,
            output_schema={"type": "object"},
        ),
        ToolSpec(
            name="list_orders",
            description="List the most recent orders.",
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
            handler=_list_orders,
            output_schema={"type": "array", "items": {"type": "object"}},
        ),
        ToolSpec(
            name="get_customer",
            description="Fetch one record by its numeric id.",
            input_schema={
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
            handler=_get_customer,
            output_schema={"type": "object"},
        ),
        ToolSpec(
            name="crm_find_contact",
            description="Look up a contact record by its address.",
            input_schema={
                "type": "object",
                "properties": {"customer_email": {"type": "string"}},
                "required": ["customer_email"],
            },
            handler=_find_contact,
            output_schema={"type": "object"},
        ),
    ]


_DOMAINS = {
    "billing": [
        "payment",
        "refund",
        "subscription",
        "plan",
        "coupon",
        "charge",
        "payout",
        "ledger",
    ],
    "crm": ["lead", "deal", "contact", "account", "note", "campaign", "pipeline", "task"],
    "inventory": ["product", "warehouse", "shipment", "supplier", "sku", "stock", "batch", "bin"],
    "support": ["ticket", "comment", "queue", "agent", "macro", "survey", "article", "tag"],
    "hr": ["employee", "leave", "payroll", "review", "position", "benefit", "timesheet", "team"],
    "calendar": ["event", "room", "reminder", "slot", "invite", "booking", "series", "feed"],
    "analytics": ["report", "metric", "funnel", "cohort", "segment", "dashboard", "alert", "goal"],
    "docs": ["page", "folder", "revision", "template", "comment_thread", "share", "label", "draft"],
}
_VERBS = [
    ("get", "Get one {n} by id.", False),
    ("list", "List {n} records, newest first.", False),
    ("search", "Search {n} records by free text.", False),
    ("create", "Create a new {n}.", True),
    ("update", "Update fields of an existing {n}.", True),
    ("delete", "Delete a {n} permanently.", True),
    ("archive", "Archive a {n} so it stops showing up.", True),
    ("export", "Export {n} records as rows.", False),
]


def _generated(domain: str, noun: str, verb: str, desc: str, writes: bool) -> ToolSpec:
    label = noun.replace("_", " ")
    props: dict[str, Any] = {}
    required: list[str] = []
    if verb in ("get", "update", "delete", "archive"):
        props["id"] = {"type": "integer"}
        required.append("id")
    if verb in ("list", "search", "export"):
        props["limit"] = {"type": "integer"}
        props["status"] = {"type": "string", "enum": ["open", "closed", "all"]}
    if verb == "search":
        props["query"] = {"type": "string"}
        required.append("query")
    if verb in ("create", "update"):
        props["name"] = {"type": "string"}
        props["notes"] = {"type": "string"}
        if verb == "create":
            required.append("name")
    result = {"domain": domain, "noun": noun, "verb": verb}
    return ToolSpec(
        name=f"{domain}_{verb}_{noun}",
        description=desc.format(n=label),
        input_schema={"type": "object", "properties": props, "required": required},
        handler=_const(result),
        output_schema={"type": "object"},
        writes=writes,
    )


def synthetic_catalog(n: int) -> list[ToolSpec]:
    """First ``n`` tools of a fixed catalog: the core tools, then generated ones."""
    out = _core()
    seen = {t.name for t in out}
    domains = list(_DOMAINS)
    for slot in range(8):
        for verb, desc, writes in _VERBS:
            for d in domains:
                spec = _generated(d, _DOMAINS[d][slot], verb, desc, writes)
                if spec.name not in seen:
                    seen.add(spec.name)
                    out.append(spec)
    return out[:n]
