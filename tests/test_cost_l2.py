"""Cost eval, level 2: real agent loops against ``litellm.acompletion``, mocked here (no spend)."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("litellm")

import litellm

from astromesh_codemode.eval import __main__ as cli
from astromesh_codemode.eval import cost
from astromesh_codemode.eval.cost import (
    MAX_MODEL_CALLS,
    Level2Row,
    answer_matches,
    run_level2,
    tool_set,
)
from astromesh_codemode.eval.scenarios import SYSTEM_PROMPT, Scenario, expected_answer, user_task

MODEL = "gpt-4o-mini"
USAGE_IN, USAGE_OUT = 100, 10


# ---------------------------------------------------------------- scripted model plumbing


def _tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=USAGE_IN, completion_tokens=USAGE_OUT),
    )


def _tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


def _final(answer: list[dict]) -> str:
    return "\n".join(
        f"Order {a['order_id']}: amount {a['amount']}, customer {a['customer']}." for a in answer
    )


PROGRAM = """\
import asyncio
orders = await list_orders(limit={rows})
top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:3]
customers = await asyncio.gather(*[get_customer(customer_id=o["customer_id"]) for o in top])
[{{"order_id": o["id"], "amount": o["amount"], "customer": c["name"]}}
 for o, c in zip(top, customers)]
"""


def _rows_from_task(messages: list[dict]) -> int:
    return int(re.search(r"limit=(\d+)", messages[1]["content"]).group(1))


def scripted_model(messages: list[dict], tools: list[dict]) -> SimpleNamespace:
    """A model that plays the task right, reading only what it was sent."""
    names = _tool_names(tools)
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    if "codemode_run" in names:
        if not tool_msgs:
            return _response(tool_calls=[_tool_call("s1", "codemode_search", {"query": "orders"})])
        if len(tool_msgs) == 1:
            code = PROGRAM.format(rows=_rows_from_task(messages))
            return _response(tool_calls=[_tool_call("r1", "codemode_run", {"code": code})])
        line = next(x for x in tool_msgs[-1]["content"].splitlines() if x.startswith("result: "))
        return _response(content=_final(json.loads(line.removeprefix("result: "))))
    # direct
    if not tool_msgs:
        rows = _rows_from_task(messages)
        return _response(tool_calls=[_tool_call("c1", "list_orders", {"limit": rows})])
    if len(tool_msgs) == 1:
        orders = json.loads(tool_msgs[0]["content"])
        top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:3]
        calls = [  # parallel tool calls: three in one turn
            _tool_call(f"g{i}", "get_customer", {"customer_id": o["customer_id"]})
            for i, o in enumerate(top)
        ]
        return _response(tool_calls=calls)
    orders = json.loads(tool_msgs[0]["content"])
    top = sorted(orders, key=lambda o: o["amount"], reverse=True)[:3]
    customers = [json.loads(m["content"]) for m in tool_msgs[1:4]]
    answer = [
        {"order_id": o["id"], "amount": o["amount"], "customer": c["name"]}
        for o, c in zip(top, customers, strict=True)
    ]
    return _response(content=_final(answer))


class FakeLiteLLM:
    """Replaces ``litellm.acompletion``; records every request."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.requests: list[dict] = []

    async def __call__(self, *, model, messages, tools, **kwargs):
        self.requests.append(
            {"model": model, "messages": [dict(m) for m in messages], "tools": tools, **kwargs}
        )
        result = self.behaviour(messages, tools)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(cost, "_sleep", fake_sleep)
    return slept


def _install(monkeypatch, behaviour) -> FakeLiteLLM:
    fake = FakeLiteLLM(behaviour)
    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


def _by_key(rows: list[Level2Row]) -> dict[tuple, Level2Row]:
    return {(r.model, r.scenario.catalog_size, r.scenario.data, r.strategy): r for r in rows}


def _price(model: str) -> tuple[float, float]:
    p = litellm.model_cost[model]
    return p["input_cost_per_token"], p["output_cost_per_token"]


# ---------------------------------------------------------------- the matcher


async def test_answer_matches_uses_expected_answer():
    s = Scenario(8, "large")
    expected = await expected_answer(s)
    assert answer_matches(_final(expected), expected)
    # Free-form wording and markdown are fine.
    loose = " | ".join(f"#{a['order_id']} — ${a['amount']} — **{a['customer']}**" for a in expected)
    assert answer_matches(loose, expected)
    # Wrong order of the top 3 fails.
    assert not answer_matches(_final(list(reversed(expected))), expected)
    # A missing amount fails.
    assert not answer_matches(_final(expected).replace(str(expected[1]["amount"]), "?"), expected)
    # "Customer 1" is not found inside "Customer 12".
    fake = [{"order_id": 1, "amount": 11, "customer": "Customer 1"}]
    assert not answer_matches("Order 1: amount 11, customer Customer 12.", fake)
    assert not answer_matches("", expected)


# ---------------------------------------------------------------- the loop


async def test_level2_rows_success_and_cost(monkeypatch, no_sleep):
    fake = _install(monkeypatch, scripted_model)
    scenarios = [Scenario(8, "small"), Scenario(64, "large")]
    models = [MODEL, "gpt-4o"]
    rows = await run_level2(scenarios, models, repeats=2)

    # One row per (model x scenario x strategy).
    assert len(rows) == len(models) * len(scenarios) * 2
    by = _by_key(rows)
    assert set(by) == {
        (m, s.catalog_size, s.data, st)
        for m in models
        for s in scenarios
        for st in ("direct", "codemode")
    }
    for (model, *_), row in by.items():
        assert row.success_rate == 1.0, row
        assert row.model_calls == 3.0
        assert row.tokens_in == 3 * USAGE_IN
        assert row.tokens_out == 3 * USAGE_OUT
        p_in, p_out = _price(model)
        assert row.cost_usd == pytest.approx(3 * (USAGE_IN * p_in + USAGE_OUT * p_out))
        assert row.latency_s >= 0
        assert row.errors == {}
        assert row.attempts == 2
    # 8 rows x 2 repeats x 3 calls.
    assert len(fake.requests) == 8 * 2 * 3


async def test_both_strategies_get_the_same_prompt_and_no_hints(monkeypatch, no_sleep):
    fake = _install(monkeypatch, scripted_model)
    s = Scenario(64, "small")
    await run_level2([s], [MODEL], repeats=1)

    firsts = [r for r in fake.requests if len(r["messages"]) == 2]
    assert len(firsts) == 2
    for r in firsts:
        assert r["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_task(s)},
        ]
        assert r["model"] == MODEL
    tool_arrays = {tuple(_tool_names(r["tools"])) for r in fake.requests}
    assert tool_arrays == {
        tuple(t.name for t in tool_set(s)),
        ("codemode_search", "codemode_run"),
    }
    # The prompt never names the result keys the stubs do not show.
    for r in firsts:
        text = json.dumps(r["messages"])
        for key in ('"amount"', "customer_id", '"name"'):
            assert key not in text


async def test_real_tools_run_and_parallel_calls_are_all_fed_back(monkeypatch, no_sleep):
    fake = _install(monkeypatch, scripted_model)
    s = Scenario(8, "large")
    await run_level2([s], [MODEL], repeats=1)
    expected = await expected_answer(s)

    direct_last = next(
        r
        for r in fake.requests
        if "list_orders" in _tool_names(r["tools"]) and (len(r["messages"]) == 2 + 2 + 1 + 3)
    )
    msgs = direct_last["messages"]
    assistant = msgs[4]
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 3
    tool_msgs = msgs[5:]
    assert [m["tool_call_id"] for m in tool_msgs] == [c["id"] for c in assistant["tool_calls"]]
    assert [json.loads(m["content"])["name"] for m in tool_msgs] == [
        e["customer"] for e in expected
    ]
    # The rows came from the real handler: 500 of them.
    assert len(json.loads(msgs[3]["content"])) == 500

    cm_last = next(
        r
        for r in fake.requests
        if "codemode_run" in _tool_names(r["tools"]) and len(r["messages"]) == 6
    )
    search_out, run_out = cm_last["messages"][3]["content"], cm_last["messages"][5]["content"]
    assert "async def list_orders" in search_out  # real CodeMode.search stubs
    assert "Customer 3" in run_out  # real sandbox ran the program


async def test_wrong_answer_is_a_failure(monkeypatch, no_sleep):
    _install(monkeypatch, lambda m, t: _response(content="I could not find the orders."))
    rows = await run_level2([Scenario(8, "small")], [MODEL], repeats=2)
    for r in rows:
        assert r.success_rate == 0.0
        assert r.model_calls == 1.0


async def test_tool_errors_go_back_to_the_model(monkeypatch, no_sleep):
    def bad_then_good(messages, tools):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if "codemode_run" in _tool_names(tools):
            return _response(content="done")
        if not tool_msgs:
            return _response(tool_calls=[_tool_call("x1", "no_such_tool", {})])
        if len(tool_msgs) == 1:
            bad = SimpleNamespace(
                id="x2",
                type="function",
                function=SimpleNamespace(name="list_orders", arguments="{not json"),
            )
            return _response(tool_calls=[bad])
        return _response(content="gave up")

    fake = _install(monkeypatch, bad_then_good)
    await run_level2([Scenario(8, "small")], [MODEL], repeats=1)
    last = [r for r in fake.requests if "list_orders" in _tool_names(r["tools"])][-1]
    tool_msgs = [m for m in last["messages"] if m["role"] == "tool"]
    assert "no_such_tool" in tool_msgs[0]["content"]
    assert "error" in tool_msgs[0]["content"].lower()
    assert "error" in tool_msgs[1]["content"].lower()


class RateLimitError(Exception):
    pass


async def test_model_error_retries_then_succeeds(monkeypatch, no_sleep):
    failures = {"left": 2}

    def flaky(messages, tools):
        if len(messages) == 2 and failures["left"] > 0:
            failures["left"] -= 1
            return RateLimitError("slow down")
        return scripted_model(messages, tools)

    fake = _install(monkeypatch, flaky)
    rows = await run_level2([Scenario(8, "small")], [MODEL], repeats=1)
    direct = _by_key(rows)[(MODEL, 8, "small", "direct")]
    assert direct.success_rate == 1.0
    assert direct.model_calls == 3.0  # failed requests are not model calls with usage
    assert direct.errors == {}
    assert no_sleep == list(cost.RETRY_BACKOFF_S)  # two retries, with backoff
    assert len(fake.requests) == 2 + 3 + 3


async def test_model_error_after_retries_fails_the_attempt_not_the_run(monkeypatch, no_sleep):
    def broken_direct(messages, tools):
        if "list_orders" in _tool_names(tools) and len(messages) > 2:
            return RateLimitError("still slow")
        return scripted_model(messages, tools)

    fake = _install(monkeypatch, broken_direct)
    rows = await run_level2([Scenario(8, "small")], [MODEL], repeats=2)
    by = _by_key(rows)
    direct, cm = by[(MODEL, 8, "small", "direct")], by[(MODEL, 8, "small", "codemode")]
    assert direct.success_rate == 0.0
    assert direct.errors == {"RateLimitError": 2}
    assert direct.model_calls == 1.0  # the first call answered before the provider broke
    assert direct.tokens_in == USAGE_IN
    assert cm.success_rate == 1.0  # the run went on
    # Each failed turn: 1 try + 2 retries.
    direct_requests = [r for r in fake.requests if "list_orders" in _tool_names(r["tools"])]
    assert len(direct_requests) == 2 * (1 + 3)


async def test_cap_of_twelve_model_calls(monkeypatch, no_sleep):
    def loops_forever(messages, tools):
        name = "list_orders" if "list_orders" in _tool_names(tools) else "codemode_search"
        args = {"limit": 5} if name == "list_orders" else {"query": "orders"}
        return _response(tool_calls=[_tool_call(f"c{len(messages)}", name, args)])

    fake = _install(monkeypatch, loops_forever)
    rows = await run_level2([Scenario(8, "small")], [MODEL], repeats=1)
    assert MAX_MODEL_CALLS == 12
    for r in rows:
        assert r.success_rate == 0.0
        assert r.model_calls == 12.0
        assert r.capped == 1.0
        assert r.tokens_in == 12 * USAGE_IN
    assert len(fake.requests) == 2 * 12


async def test_unknown_price_is_refused(monkeypatch, no_sleep):
    _install(monkeypatch, scripted_model)
    with pytest.raises(ValueError, match="price"):
        await run_level2([Scenario(8, "small")], ["no-such-provider/no-such-model"], repeats=1)


# ---------------------------------------------------------------- results and CLI


def test_markdown_and_json(tmp_path):
    s = Scenario(8, "small")
    row = Level2Row(
        model=MODEL,
        scenario=s,
        strategy="direct",
        success_rate=2 / 3,
        model_calls=3.0,
        tokens_in=300.0,
        tokens_out=30.0,
        cost_usd=0.000063,
        latency_s=1.25,
        attempts=3,
        capped=0.0,
        errors={"RateLimitError": 1},
    )
    md = cost.format_markdown_l2([row], "2026-11-25")
    assert "level 2" in md
    assert "| gpt-4o-mini | 8 | small | direct | 67% |" in md
    payload = cost.to_json_l2([row], "2026-11-25", repeats=3)
    assert payload["level"] == 2
    assert payload["rows"][0]["scenario"] == {"catalog_size": 8, "data": "small", "rows": 5}
    assert payload["rows"][0]["errors"] == {"RateLimitError": 1}
    json.dumps(payload)


def test_cli_refuses_without_api_key(monkeypatch, tmp_path, capsys):
    fake = _install(monkeypatch, scripted_model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = cli.main(["cost", "--level", "2", "--models", MODEL, "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert fake.requests == []
    assert not list(tmp_path.iterdir())


def test_cli_requires_models_for_level2(capsys):
    assert cli.main(["cost", "--level", "2"]) == 2
    assert "--models" in capsys.readouterr().err


def test_cli_rejects_bad_scenarios(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    code = cli.main(["cost", "--level", "2", "--models", MODEL, "--scenarios", "64/huge"])
    assert code == 2
    assert "64/huge" in capsys.readouterr().err


def test_cli_level2_writes_results(monkeypatch, tmp_path, capsys, no_sleep):
    fake = _install(monkeypatch, scripted_model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    code = cli.main(
        [
            "cost",
            "--level",
            "2",
            "--models",
            MODEL,
            "--repeats",
            "1",
            "--scenarios",
            "8/small,64/large",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    # The upper bound: 1 model x 2 scenarios x 2 strategies x 1 repeat x 12 calls.
    assert "48" in out
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 2
    assert files[0].endswith("-L2.json")
    assert files[1].endswith("-L2.md")
    payload = json.loads((tmp_path / files[0]).read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 4
    assert all(r["success_rate"] == 1.0 for r in payload["rows"])
    assert len(fake.requests) == 4 * 3
