"""Cost eval, level 1: a scripted model, real tools, tokens counted with ``o200k_base``."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tiktoken")

from astromesh_codemode._synthetic import synthetic_catalog
from astromesh_codemode.eval import __main__ as cli
from astromesh_codemode.eval.cost import Level1Row, format_markdown, run_level1, tool_set
from astromesh_codemode.eval.scenarios import SCENARIOS, Scenario, expected_answer


def _by_key(rows: list[Level1Row]) -> dict[tuple[int, str, str], Level1Row]:
    return {(r.scenario.catalog_size, r.scenario.data, r.strategy): r for r in rows}


def test_scenarios_are_the_product():
    assert {(s.catalog_size, s.data) for s in SCENARIOS} == {
        (n, d) for n in (8, 64, 512) for d in ("small", "large")
    }


def test_both_strategies_get_the_same_read_only_tools():
    for s in SCENARIOS:
        tools = tool_set(s)
        assert tools, s
        assert not any(t.writes for t in tools)
        reads = [t.name for t in synthetic_catalog(s.catalog_size) if not t.writes]
        assert [t.name for t in tools] == reads
    # The write tools exist in the bigger catalogs; they are left out of both strategies.
    assert any(t.writes for t in synthetic_catalog(64))


async def test_level1_deterministic_and_directional():
    first = await run_level1(SCENARIOS)
    second = await run_level1(SCENARIOS)
    assert first == second

    rows = _by_key(first)
    assert len(rows) == 2 * len(SCENARIOS)
    big_direct, big_cm = rows[(512, "large", "direct")], rows[(512, "large", "codemode")]
    assert big_cm.context_tokens < big_direct.context_tokens

    for s in SCENARIOS:
        expected = await expected_answer(s)
        for strategy in ("direct", "codemode"):
            row = rows[(s.catalog_size, s.data, strategy)]
            assert row.answer == expected, (s, strategy)
            assert row.model_calls == 3
            assert row.context_tokens > 0
            assert row.output_tokens > 0


async def test_level1_can_show_codemode_losing():
    # Honesty check: the eval is not built so that Code Mode always wins.
    # With only the 4 core tools and 5 rows, its two long tool descriptions plus the
    # search output outweigh four small schemas: it carries MORE context than direct.
    rows = _by_key(await run_level1([Scenario(4, "small")]))
    assert rows[(4, "small", "codemode")].context_tokens > (
        rows[(4, "small", "direct")].context_tokens
    )
    # And writing the program always costs more output than emitting tool calls.
    for s in SCENARIOS:
        rows = _by_key(await run_level1([s]))
        key = (s.catalog_size, s.data)
        assert rows[(*key, "codemode")].output_tokens > rows[(*key, "direct")].output_tokens


async def test_expected_answer_has_no_ties():
    for data in ("small", "large"):
        answer = await expected_answer(Scenario(8, data))
        assert len(answer) == 3
        amounts = [o["amount"] for o in answer]
        assert amounts == sorted(set(amounts), reverse=True)


def test_cli_writes_results(tmp_path, capsys):
    assert cli.main(["cost", "--level", "1", "--out", str(tmp_path)]) == 0
    (js,) = tmp_path.glob("*-L1.json")
    (md,) = tmp_path.glob("*-L1.md")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["level"] == 1
    assert data["tokenizer"] == "o200k_base"
    assert len(data["rows"]) == 2 * len(SCENARIOS)
    table = md.read_text(encoding="utf-8")
    assert "| 512 | large | codemode |" in table
    assert "codemode/direct" in table
    assert "codemode/direct" in capsys.readouterr().out


async def test_format_markdown_has_ratio_per_scenario():
    rows = await run_level1([Scenario(8, "small")])
    md = format_markdown(rows)
    assert md.count("| 8 | small |") == 2
    assert "ratio" in md.lower()
