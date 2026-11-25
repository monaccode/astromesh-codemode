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

Level 2 (``run_level2``) runs the same scenarios with real models through
``litellm.acompletion``. Each attempt is a real agent loop: system prompt + task + tools
(``direct``: the shared read-only catalog; ``codemode``: only ``CodeMode``'s two tools);
every tool call the model makes runs against the real handlers or the real ``CodeMode``
and its result goes back, until the model answers without tool calls or after
``MAX_MODEL_CALLS``. Tokens come from each response's ``usage``; cost is ``usage`` x the
per-token prices of the cost map bundled with the installed litellm. Rows where every
attempt errored (a provider that rejects 256 tools) are ``not_runnable``, never $0.
See ``docs/eval.md`` ("Cost — level 2").
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import Any, Literal

from astromesh_codemode._synthetic import synthetic_catalog
from astromesh_codemode.codemode import CodeMode
from astromesh_codemode.eval.scenarios import (
    SYSTEM_PROMPT,
    TOP_K,
    Scenario,
    expected_answer,
    user_task,
)
from astromesh_codemode.spec import ToolSpec

__all__ = [
    "ENCODING",
    "MAX_MODEL_CALLS",
    "Level1Row",
    "Level2Row",
    "answer_matches",
    "format_markdown",
    "format_markdown_l2",
    "is_transient",
    "litellm_version",
    "max_model_requests",
    "missing_api_keys",
    "price_per_token",
    "ratios",
    "run_level1",
    "run_level2",
    "to_json",
    "to_json_l2",
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


# --------------------------------------------------------------------------- level 2

MAX_MODEL_CALLS = 12
"""Model calls per attempt; an attempt still calling tools at the cap fails."""
RETRY_BACKOFF_S = (2.0, 8.0)
"""Seconds to wait before each retry of a transient model error: 2 retries."""
_sleep = asyncio.sleep  # patched in tests

ToolHandler = Callable[..., Awaitable[Any]]
RowStatus = Literal["ok", "partial", "not_runnable"]


@dataclass(frozen=True)
class Level2Row:
    """One (model, scenario, strategy) over ``attempts`` repeats.

    ``success_rate`` counts every attempt (an errored one is a failure). The averages
    (calls, tokens, cost, latency) are over the attempts that did not error, so a provider
    that rejects the request never shows up as a cheap row: when every attempt errored,
    ``status`` is ``"not_runnable"`` and the averages are ``None``.
    """

    model: str
    scenario: Scenario
    strategy: Strategy
    success_rate: float
    """Fraction of all attempts whose final answer matched ``expected_answer``."""
    model_calls: float | None
    """Model calls that returned a response, per non-errored attempt."""
    tokens_in: float | None
    tokens_out: float | None
    cost_usd: float | None
    latency_s: float | None
    """Wall clock per non-errored attempt: model calls, tool runs and retry waits."""
    attempts: int
    status: RowStatus = "ok"
    """``ok``: no attempt errored; ``partial``: some did; ``not_runnable``: all did."""
    errored: int = 0
    """Attempts lost to a model error (after retries, when it was transient)."""
    errors: dict[str, int] = field(default_factory=dict)
    """``errored`` by exception type."""
    cost_per_success: float | None = None
    """Cost of the non-errored attempts divided by the successes; ``None`` with none."""
    capped: float | None = None
    """Fraction of non-errored attempts that reached ``MAX_MODEL_CALLS`` calling tools."""


@dataclass
class _Attempt:
    success: bool = False
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    capped: bool = False
    error: str | None = None


def _litellm() -> Any:
    """Import litellm with its bundled price map, so prices do not change under a run.

    By default litellm downloads the latest cost map at import time; the eval pins it to
    the copy shipped with the installed litellm, whose version goes into the results.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The level-2 cost eval needs litellm: pip install 'astromesh-codemode[eval]'"
        ) from exc
    return litellm


def litellm_version() -> str:
    _litellm()
    return importlib.metadata.version("litellm")


def price_per_token(model: str) -> tuple[float, float] | None:
    """``(input, output)`` USD per token from ``litellm.model_cost``, or None if unknown."""
    table = _litellm().model_cost
    for key in (model, model.split("/", 1)[-1]):
        entry = table.get(key) or {}
        if "input_cost_per_token" in entry and "output_cost_per_token" in entry:
            return float(entry["input_cost_per_token"]), float(entry["output_cost_per_token"])
    return None


def missing_api_keys(model: str) -> list[str]:
    """Env vars litellm says ``model``'s provider needs and are not set; [] when ready."""
    env = _litellm().validate_environment(model)
    return [] if env.get("keys_in_environment") else list(env.get("missing_keys") or [])


_NEVER_RETRY = (
    "BadRequestError",  # ContextWindowExceededError is a subclass
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ContextWindowExceededError",
)
_TRANSIENT = (
    "RateLimitError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
)


def _classes(names: Iterable[str]) -> tuple[type[BaseException], ...]:
    litellm = _litellm()
    return tuple(c for n in names if isinstance(c := getattr(litellm, n, None), type))


def is_transient(exc: BaseException) -> bool:
    """Whether a model error is worth retrying: rate limits, timeouts, connection, 5xx.

    A request the provider rejects (bad request, too many tools, context window, auth,
    unknown model) fails the same way every time, so it is never retried.
    """
    if isinstance(exc, _classes(_NEVER_RETRY)):
        return False
    if isinstance(exc, _classes(_TRANSIENT)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _number(n: float) -> re.Pattern[str]:
    return re.compile(rf"(?<![\d.]){re.escape(str(n))}(?:\.0+)?(?!\d)")


def _name_family(names: list[str]) -> re.Pattern[str] | None:
    """For names like ``Customer 3``, a pattern matching any ``Customer <n>``."""
    prefixes = {m.group(1) for n in names if (m := re.fullmatch(r"(.*\S)\s+\d+", n))}
    if len(prefixes) != 1:
        return None
    return re.compile(rf"(?<!\w){re.escape(prefixes.pop())}\s+\d+(?!\w)")


def answer_matches(text: str | None, expected: list[dict]) -> bool:
    """True when ``text`` gives exactly the rows of ``expected_answer``, in order.

    The customer names must appear in the expected order, each matched whole
    (``Customer 1`` is not found inside ``Customer 12``), and no other customer of the
    same family may be named (an answer listing the top 5 is wrong). Each row's order id
    and amount must appear between the previous row's name and the next row's name, so
    they can be written before or after the name ("order 499, 509 USD, Customer 3" or
    "Customer 3: #499, $509.00"). The wording is free.
    """
    if not text or not expected:
        return False
    names = [str(row["customer"]) for row in expected]
    family = _name_family(names)
    if family is not None:
        wanted = {re.sub(r"\s+", " ", n) for n in names}
        if any(re.sub(r"\s+", " ", m.group(0)) not in wanted for m in family.finditer(text)):
            return False
    spans: list[tuple[int, int]] = []
    pos = 0
    for name in names:
        m = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)").search(text, pos)
        if m is None:
            return False
        spans.append(m.span())
        pos = m.end()
    for i, row in enumerate(expected):
        lo = spans[i - 1][1] if i else 0
        hi = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        window = text[lo:hi]
        if not (_number(row["amount"]).search(window) and _number(row["order_id"]).search(window)):
            return False
    return True


def _call_dict(call: Any) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments or ""},
    }


async def _complete(model: str, messages: list[dict], tools: list[dict]) -> Any:
    """``litellm.acompletion``; a transient error is retried ``len(RETRY_BACKOFF_S)`` times."""
    litellm = _litellm()
    for wait in (*RETRY_BACKOFF_S, None):
        try:
            return await litellm.acompletion(model=model, messages=messages, tools=tools)
        except Exception as exc:
            if wait is None or not is_transient(exc):
                raise
            await _sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


async def _run_tool(handlers: dict[str, ToolHandler], call: dict) -> str:
    """Run one tool call; a failure goes back to the model as text, as frameworks do."""
    name = call["function"]["name"]
    if name not in handlers:
        return f"Error: unknown tool {name!r}. Available tools: {', '.join(handlers)}."
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
        if not isinstance(args, dict):
            raise TypeError("tool arguments must be a JSON object")
        result = await handlers[name](**args)
    except Exception as exc:  # noqa: BLE001 - the model sees the error, the eval goes on
        return f"Error: {type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else json.dumps(result, default=str)


async def _attempt(
    model: str,
    scenario: Scenario,
    tools: list[dict],
    handlers: dict[str, ToolHandler],
    expected: list[dict],
) -> _Attempt:
    out = _Attempt()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task(scenario)},
    ]
    start = time.perf_counter()
    try:
        while True:
            if out.calls >= MAX_MODEL_CALLS:
                out.capped = True
                break
            try:
                resp = await _complete(model, messages, tools)
            except Exception as exc:  # noqa: BLE001 - recorded with its type, the run goes on
                out.error = type(exc).__name__
                break
            out.calls += 1
            usage = getattr(resp, "usage", None)
            out.tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            out.tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
            msg = resp.choices[0].message
            calls = [_call_dict(c) for c in getattr(msg, "tool_calls", None) or []]
            if not calls:
                out.success = answer_matches(msg.content, expected)
                break
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": calls})
            # Parallel tool calls: every one runs and every result goes back, in order.
            for call in calls:
                content = await _run_tool(handlers, call)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    finally:
        out.latency_s = time.perf_counter() - start
    return out


def _row(
    model: str,
    scenario: Scenario,
    strategy: Strategy,
    attempts: list[_Attempt],
    prices: tuple[float, float],
) -> Level2Row:
    p_in, p_out = prices
    clean = [a for a in attempts if a.error is None]
    errored = len(attempts) - len(clean)
    costs = [a.tokens_in * p_in + a.tokens_out * p_out for a in clean]
    successes = sum(a.success for a in attempts)

    def mean(xs: Iterable[float]) -> float | None:
        return sum(xs) / len(clean) if clean else None

    status: RowStatus = "ok" if not errored else "not_runnable" if not clean else "partial"
    return Level2Row(
        model=model,
        scenario=scenario,
        strategy=strategy,
        success_rate=successes / len(attempts),
        model_calls=mean(a.calls for a in clean),
        tokens_in=mean(a.tokens_in for a in clean),
        tokens_out=mean(a.tokens_out for a in clean),
        cost_usd=mean(costs),
        latency_s=mean(a.latency_s for a in clean),
        attempts=len(attempts),
        status=status,
        errored=errored,
        errors=dict(Counter(a.error for a in attempts if a.error)),
        cost_per_success=sum(costs) / successes if successes else None,
        capped=mean(a.capped for a in clean),
    )


async def run_level2(
    scenarios: Iterable[Scenario], models: list[str], repeats: int = 3
) -> list[Level2Row]:
    """For each scenario and model, a ``direct`` row then a ``codemode`` row.

    Raises ``ValueError`` before any model call when a model has no price in
    ``litellm.model_cost``: the table would have no cost.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    prices: dict[str, tuple[float, float]] = {}
    for m in models:
        p = price_per_token(m)
        if p is None:
            raise ValueError(f"no price for model {m!r} in litellm.model_cost")
        prices[m] = p

    rows: list[Level2Row] = []
    for s in scenarios:
        catalog = tool_set(s)
        expected = await expected_answer(s)
        direct_tools = _openai_tools((t.name, t.description, t.input_schema) for t in catalog)
        direct_handlers: dict[str, ToolHandler] = {t.name: t.handler for t in catalog}
        async with CodeMode(catalog) as cm:
            defs = cm.tool_definitions()
            cm_tools = _openai_tools((d.name, d.description, d.parameters) for d in defs)
            cm_handlers: dict[str, ToolHandler] = {d.name: d.handler for d in defs}
            await cm.run("None")  # start the worker pool outside the timed attempts
            for m in models:
                strategies: list[tuple[Strategy, list[dict], dict[str, ToolHandler]]] = [
                    ("direct", direct_tools, direct_handlers),
                    ("codemode", cm_tools, cm_handlers),
                ]
                for strategy, tools, handlers in strategies:
                    attempts = [
                        await _attempt(m, s, tools, handlers, expected) for _ in range(repeats)
                    ]
                    rows.append(_row(m, s, strategy, attempts, prices[m]))
    return rows


def max_model_requests(n_models: int, n_scenarios: int, repeats: int) -> tuple[int, int]:
    """Upper bound of a level-2 run: ``(model calls, requests counting retries)``."""
    calls = n_models * n_scenarios * 2 * repeats * MAX_MODEL_CALLS
    return calls, calls * (len(RETRY_BACKOFF_S) + 1)


def _prices_of(rows: list[Level2Row]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for model in dict.fromkeys(r.model for r in rows):
        p = price_per_token(model)
        out[model] = {
            "input_per_token": p[0] if p else None,
            "output_per_token": p[1] if p else None,
        }
    return out


def to_json_l2(rows: list[Level2Row], date: str, repeats: int) -> dict:
    return {
        "level": 2,
        "date": date,
        "repeats": repeats,
        "max_model_calls": MAX_MODEL_CALLS,
        "litellm_version": litellm_version(),
        "cost": "response usage x per-token prices of litellm's bundled model_cost map",
        "prices_usd": _prices_of(rows),
        "rows": [
            {**asdict(r), "scenario": asdict(r.scenario) | {"rows": r.scenario.rows}} for r in rows
        ],
    }


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _per_million(per_token: float | None) -> str:
    return "n/a" if per_token is None else f"{per_token * 1e6:g}"


def format_markdown_l2(rows: list[Level2Row], date: str | None = None) -> str:
    """The table: model, scenario, strategy, success, calls, tokens, cost, latency, errors."""
    lines = ["# Cost eval — level 2 (real models)", ""]
    if date:
        repeats = rows[0].attempts if rows else 0
        lines += [
            f"Date: {date}. {repeats} attempts per row, at most {MAX_MODEL_CALLS} model calls "
            "each. Success counts every attempt; the other numbers are averages over the "
            "attempts that did not error. See `docs/eval.md`.",
            "",
            f"Prices: litellm {litellm_version()} bundled `model_cost`, USD per 1M tokens "
            "(input / output): "
            + "; ".join(
                f"`{m}` {_per_million(p['input_per_token'])} / "
                f"{_per_million(p['output_per_token'])}"
                for m, p in _prices_of(rows).items()
            )
            + ".",
            "",
        ]
    lines += [
        "| model | catalog | data | strategy | status | success | model_calls | tokens_in "
        "| tokens_out | cost_usd | cost_per_success | latency_s | capped | errors |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        errors = ", ".join(f"{k} x{v}" for k, v in sorted(r.errors.items()))
        head = (
            f"| {r.model} | {r.scenario.catalog_size} | {r.scenario.data} | {r.strategy} "
            f"| {r.status} | {r.success_rate:.0%} "
        )
        if r.status == "not_runnable":
            na = f"n/a ({', '.join(sorted(r.errors))})"
            lines.append(head + f"| {na} " * 7 + f"| {errors} |")
            continue
        lines.append(
            head + f"| {_fmt(r.model_calls, '.1f')} | {_fmt(r.tokens_in, '.0f')} "
            f"| {_fmt(r.tokens_out, '.0f')} | {_fmt(r.cost_usd, '.6f')} "
            f"| {_fmt(r.cost_per_success, '.6f')} | {_fmt(r.latency_s, '.2f')} "
            f"| {_fmt(r.capped, '.0%')} | {errors} |"
        )
    lines.append("")
    return "\n".join(lines)
