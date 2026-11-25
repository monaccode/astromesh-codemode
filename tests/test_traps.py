"""The trap eval: malicious programs against ``CodeMode`` and how many are stopped."""

from __future__ import annotations

import pytest

from astromesh_codemode import CodeMode, Limits, sandbox
from astromesh_codemode.eval import TrapReport, format_report, run_traps, traps
from astromesh_codemode.eval import __main__ as cli

TRAP_NAMES = {
    "os_import",
    "file_read",
    "network",
    "infinite_loop",
    "memory_bomb",
    "deep_recursion",
    "unknown_tool",
    "bad_args",
    "output_flood",
    "call_budget",
    "excluded_write",
}


async def test_all_traps_stopped_by_default():
    report = await run_traps()
    assert set(report.by_trap) == TRAP_NAMES
    assert report.total == 11
    assert report.stopped == 11, format_report(report)
    assert set(report.details) == TRAP_NAMES
    assert all(report.details.values())


async def test_lax_limits_let_some_through():
    report = await run_traps(Limits(max_calls=10_000, max_output_bytes=10**9))
    assert report.by_trap["call_budget"] is False, report.details["call_budget"]
    assert report.stopped < report.total


async def test_a_trap_whose_program_does_not_parse_is_not_stopped(monkeypatch):
    async def broken(limits):
        async with CodeMode([traps.Counter().spec], limits=limits) as cm:
            await traps.attempt(cm, "def (:")
        return traps.Outcome(True, "claims stopped")

    monkeypatch.setattr(traps, "TRAPS", {"broken": broken})
    report = await run_traps()
    assert report.by_trap == {"broken": False}
    assert report.details["broken"].startswith("trap error: BrokenTrapError")


@pytest.mark.parametrize(("stopped", "code"), [(2, 0), (1, 1)])
def test_cli_exit_status(monkeypatch, capsys, stopped, code):
    by_trap = {"os_import": True, "call_budget": stopped == 2}
    report = TrapReport(2, stopped, by_trap, {"os_import": "e1", "call_budget": "e2"})

    async def fake_run_traps():
        return report

    monkeypatch.setattr(cli, "run_traps", fake_run_traps)
    assert cli.main(["traps"]) == code
    out = capsys.readouterr().out
    assert f"{stopped}/2 traps stopped" in out
    assert "call_budget" in out
    assert "e2" in out


async def test_output_flood_exercises_success_path_truncation():
    outcome = await traps.output_flood(Limits())
    assert outcome.stopped, outcome.detail
    assert "max_output_bytes hit: 2" in outcome.detail
    assert "missing notice: 0" in outcome.detail


async def test_output_flood_not_stopped_when_truncation_drops_its_notice(monkeypatch):
    def silent_cut(main, prints, cap):
        full = (f"stdout:\n{prints}\n\n" if prints else "") + main
        return full.encode()[:cap].decode(errors="ignore"), len(full.encode()) > cap

    monkeypatch.setattr(sandbox, "compose", silent_cut)
    outcome = await traps.output_flood(Limits())
    assert not outcome.stopped, outcome.detail


async def test_output_flood_not_stopped_with_lax_output_cap():
    outcome = await traps.output_flood(Limits(max_output_bytes=10**9))
    assert not outcome.stopped, outcome.detail
