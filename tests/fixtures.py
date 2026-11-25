from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from astromesh_codemode._synthetic import synthetic_catalog
from astromesh_codemode.spec import ToolError, ToolSpec

__all__ = ["STAMP", "Opaque", "make_tools", "synthetic_catalog"]

STAMP = datetime(2026, 11, 25, 10, 30, tzinfo=UTC)


class Opaque:
    """A host object Monty cannot convert into a sandbox value."""

    def __repr__(self) -> str:
        return "Opaque(42)"


def make_tools() -> tuple[list[ToolSpec], Counter[str]]:
    """Test tools plus a counter of how many times each handler actually ran."""
    calls: Counter[str] = Counter()

    async def get_rows(n: int) -> list[dict[str, Any]]:
        calls["get_rows"] += 1
        return [{"id": i, "value": i * 10} for i in range(1, n + 1)]

    async def boom() -> None:
        calls["boom"] += 1
        raise ToolError("crm caído")

    async def stamp() -> datetime:
        calls["stamp"] += 1
        return STAMP

    async def save(x: int) -> dict[str, Any]:
        calls["save"] += 1
        return {"saved": x}

    async def opaque() -> Any:
        calls["opaque"] += 1
        return {"thing": Opaque(), "raw": b"\x00\x01"}

    async def crash() -> None:
        calls["crash"] += 1
        raise KeyError("postgres://admin:secret@db/prod")

    def schema(props: dict, required: list[str] | None = None) -> dict:
        return {"type": "object", "properties": props, "required": required or []}

    specs = [
        ToolSpec(
            name="get_rows",
            description="Return n rows, each with an id and a value.",
            input_schema=schema({"n": {"type": "integer"}}, ["n"]),
            handler=get_rows,
            output_schema={"type": "array", "items": {"type": "object"}},
        ),
        ToolSpec("boom", "Always fails like a broken CRM.", schema({}), boom),
        ToolSpec("stamp", "Return the current timestamp.", schema({}), stamp),
        ToolSpec(
            "save",
            "Persist a value.",
            schema({"x": {"type": "integer"}}, ["x"]),
            save,
            writes=True,
        ),
        ToolSpec("opaque", "Return a host object.", schema({}), opaque),
        ToolSpec("crash", "Fail with an unexpected host error.", schema({}), crash),
    ]
    return specs, calls
