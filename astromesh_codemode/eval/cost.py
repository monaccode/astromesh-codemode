"""Cost eval, level 1: how much context each strategy carries, with a scripted model.

No LLM is called. A scripted model plays the same task twice per scenario
(``scenarios.py``), against the real tools:

Both strategies see the same tool set: the catalog's read-only tools (``writes=False``),
which is what ``CodeMode`` v0.1 binds. Write tools appear in neither.

- ``direct``: classic tool calling. Every read-only tool is in the ``tools`` array
  on every call. Call 1 asks for ``list_orders``; call 2 sees its real result and asks
  for ``get_customer`` three times in one turn (parallel tool calls); call 3 sees the
  three results and answers.
- ``codemode``: only ``codemode_search`` and ``codemode_run`` are in the ``tools`` array.
  Call 1 asks ``search``; call 2 sees the real ``CodeMode.search`` output and sends a
  fixed program to ``run``; call 3 sees the real ``run`` output and answers.

The scripted model reads its decisions off the conversation (it parses the tool results
it was given), so a wrong tool output would give a wrong answer, and the test checks
both strategies against the answer computed straight from the handlers.

Counting, with ``tiktoken`` ``o200k_base``:

- input tokens of one call = the JSON of the OpenAI-style ``tools`` array + every message
  sent so far (``MESSAGE_OVERHEAD`` tokens per message, plus its content, plus the name
  and arguments of each tool call) + ``REPLY_PRIMING``. ``context_tokens`` is the sum over
  all model calls: each call re-sends the whole conversation.
- output tokens = what the model emits: tool-call names and arguments JSON (the program,
  for ``run``), or the final answer.

See ``docs/eval.md`` ("Cost — level 1") for the limits of this measure.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, Literal

from astromesh_codemode._synthetic import synthetic_catalog
from astromesh_codemode.codemode import CodeMode
from astromesh_codemode.eval.scenarios import (
    SYSTEM_PROMPT,
    TOP_K,
    Scenario,
    user_task,
)
from astromesh_codemode.spec import ToolSpec

__all__ = [
    "ENCODING",
    "Level1Row",
    "format_markdown",
    "ratios",
    "run_level1",
    "to_json",
    "tool_set",
]

ENCODING = "o200k_base"
MESSAGE_OVERHEAD = 3
"""Tokens a chat template adds around each message (OpenAI's published estimate)."""
REPLY_PRIMING = 3
"""Tokens that open the assistant's reply on every call."""

SEARCH_QUERY = "list orders and get customer"
PROGRAM = """\
import asyncio
orders = await list_orders(limit={rows})
top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:{k}]
customers = await asyncio.gather(*[get_customer(customer_id=o["customer_id"]) for o in top])
[{{"order_id": o["id"], "amount": o["amount"], "customer": c["name"]}}
 for o, c in zip(top, customers)]
"""

Strategy = Literal["direct", "codemode"]


@dataclass(frozen=True)
class Level1Row:
    scenario: Scenario
    strategy: Strategy
    context_tokens: int
    """Input tokens summed over every model call."""
    output_tokens: int
    model_calls: int
    answer: list[dict]
    """What the scripted model answered: order id, amount and customer name, top first."""


@cache
def _encoder() -> Any:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The cost eval needs tiktoken: pip install 'astromesh-codemode[eval]'"
        ) from exc
    return tiktoken.get_encoding(ENCODING)


def _tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _openai_tools(defs: Iterable[tuple[str, str, dict]]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
        for n, d, p in defs
    ]


def _message_tokens(msg: dict) -> int:
    n = MESSAGE_OVERHEAD + _tokens(msg["role"]) + _tokens(msg.get("content") or "")
    for call in msg.get("tool_calls", []):
        n += _tokens(call["function"]["name"]) + _tokens(call["function"]["arguments"])
    return n


def _emitted_tokens(msg: dict) -> int:
    n = _tokens(msg.get("content") or "")
    for call in msg.get("tool_calls", []):
        n += _tokens(call["function"]["name"]) + _tokens(call["function"]["arguments"])
    return n


class _Conversation:
    """Counts what each model call receives and emits."""

    def __init__(self, tools: list[dict], scenario: Scenario):
        self.tools_tokens = _tokens(json.dumps(tools))
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_task(scenario)},
        ]
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def model_turn(self, reply: dict) -> None:
        """The model is called with everything so far and replies with ``reply``."""
        self.calls += 1
        self.input_tokens += (
            self.tools_tokens + sum(map(_message_tokens, self.messages)) + REPLY_PRIMING
        )
        self.output_tokens += _emitted_tokens(reply)
        self.messages.append({"role": "assistant", **reply})

    def tool_result(self, call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def tool_contents(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "tool"]


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _final_text(answer: list[dict]) -> str:
    return "\n".join(
        f"Order {a['order_id']}: amount {a['amount']}, customer {a['customer']}." for a in answer
    )


async def _direct(scenario: Scenario, catalog: list[ToolSpec]) -> Level1Row:
    by_name = {t.name: t for t in catalog}
    conv = _Conversation(
        _openai_tools((t.name, t.description, t.input_schema) for t in catalog), scenario
    )

    conv.model_turn(
        {"content": None, "tool_calls": [_call("c1", "list_orders", {"limit": scenario.rows})]}
    )
    orders = await by_name["list_orders"].handler(limit=scenario.rows)
    conv.tool_result("c1", json.dumps(orders))

    seen = json.loads(conv.tool_contents()[-1])
    top = sorted(seen, key=lambda o: o["amount"], reverse=True)[:TOP_K]
    calls = [
        _call(f"c{2 + i}", "get_customer", {"customer_id": o["customer_id"]})
        for i, o in enumerate(top)
    ]
    conv.model_turn({"content": None, "tool_calls": calls})
    for call in calls:
        args = json.loads(call["function"]["arguments"])
        conv.tool_result(call["id"], json.dumps(await by_name["get_customer"].handler(**args)))

    customers = [json.loads(c) for c in conv.tool_contents()[-TOP_K:]]
    answer = [
        {"order_id": o["id"], "amount": o["amount"], "customer": c["name"]}
        for o, c in zip(top, customers, strict=True)
    ]
    conv.model_turn({"content": _final_text(answer)})
    return Level1Row(scenario, "direct", conv.input_tokens, conv.output_tokens, conv.calls, answer)


def _parse_run_result(text: str) -> list[dict]:
    for line in text.splitlines():
        if line.startswith("result: "):
            return json.loads(line.removeprefix("result: "))
    raise ValueError(f"run output has no result line: {text[:200]!r}")


async def _codemode(scenario: Scenario, catalog: list[ToolSpec]) -> Level1Row:
    async with CodeMode(catalog) as cm:
        defs = {d.name: d for d in cm.tool_definitions()}
        conv = _Conversation(
            _openai_tools((d.name, d.description, d.parameters) for d in defs.values()), scenario
        )
        search, run = f"{cm.name}_search", f"{cm.name}_run"

        conv.model_turn(
            {"content": None, "tool_calls": [_call("c1", search, {"query": SEARCH_QUERY})]}
        )
        conv.tool_result("c1", await defs[search].handler(query=SEARCH_QUERY))

        code = PROGRAM.format(rows=scenario.rows, k=TOP_K)
        conv.model_turn({"content": None, "tool_calls": [_call("c2", run, {"code": code})]})
        conv.tool_result("c2", await defs[run].handler(code=code))

        answer = _parse_run_result(conv.tool_contents()[-1])
        conv.model_turn({"content": _final_text(answer)})
    return Level1Row(
        scenario, "codemode", conv.input_tokens, conv.output_tokens, conv.calls, answer
    )


def tool_set(scenario: Scenario) -> list[ToolSpec]:
    """The tools both strategies get: the catalog's read-only tools (what CodeMode v0.1 binds)."""
    return [t for t in synthetic_catalog(scenario.catalog_size) if not t.writes]


async def run_level1(scenarios: Iterable[Scenario]) -> list[Level1Row]:
    """For each scenario, a ``direct`` row then a ``codemode`` row, over the same tools."""
    rows: list[Level1Row] = []
    for s in scenarios:
        catalog = tool_set(s)
        rows.append(await _direct(s, catalog))
        rows.append(await _codemode(s, catalog))
    return rows


def ratios(rows: list[Level1Row]) -> dict[Scenario, float]:
    """``codemode.context_tokens / direct.context_tokens`` per scenario (<1: Code Mode wins)."""
    by = {(r.scenario, r.strategy): r for r in rows}
    return {
        s: by[(s, "codemode")].context_tokens / by[(s, "direct")].context_tokens
        for s, strategy in by
        if strategy == "direct" and (s, "codemode") in by
    }


def to_json(rows: list[Level1Row], date: str) -> dict:
    return {
        "level": 1,
        "date": date,
        "tokenizer": ENCODING,
        "model": "scripted",
        "rows": [
            {**asdict(r), "scenario": asdict(r.scenario) | {"rows": r.scenario.rows}} for r in rows
        ],
        "ratios": [
            {"catalog_size": s.catalog_size, "data": s.data, "codemode/direct": round(v, 4)}
            for s, v in ratios(rows).items()
        ],
    }


def format_markdown(rows: list[Level1Row], date: str | None = None) -> str:
    """The table: scenario, strategy, calls, context and output tokens, codemode/direct ratio."""
    rat = ratios(rows)
    title = "# Cost eval — level 1 (scripted model)"
    lines = [title, ""]
    if date:
        lines += [f"Date: {date}. Tokenizer: `{ENCODING}`. See `docs/eval.md`.", ""]
    lines += [
        "| catalog | data | strategy | model_calls | context_tokens | output_tokens "
        "| ratio codemode/direct |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = f"{rat[r.scenario]:.2f}" if r.strategy == "codemode" else ""
        lines.append(
            f"| {r.scenario.catalog_size} | {r.scenario.data} | {r.strategy} | {r.model_calls} "
            f"| {r.context_tokens} | {r.output_tokens} | {ratio} |"
        )
    lines += ["", "ratio < 1: Code Mode carries less context; > 1: it carries more.", ""]
    return "\n".join(lines)
