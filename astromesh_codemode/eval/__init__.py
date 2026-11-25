"""Trap eval: malicious programs run against ``CodeMode``, and how many the sandbox stops.

``await run_traps(limits)`` runs every trap in ``traps.TRAPS`` with ``limits`` and returns
a ``TrapReport``; ``python -m astromesh_codemode.eval traps`` prints the table and exits 1
unless every trap was stopped. See ``docs/eval.md`` for each trap and what "stopped"
means for it.

Each trap builds its own instrumented tools (handlers that count calls, set flags, or
a local TCP listener that counts connections), so "stopped" is checked against what
actually happened on the host, not inferred from an error message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astromesh_codemode.limits import Limits

__all__ = ["TrapReport", "format_report", "run_traps"]


@dataclass(frozen=True)
class TrapReport:
    """``stopped`` of ``total`` traps were stopped; ``by_trap`` and ``details`` per trap."""

    total: int
    stopped: int
    by_trap: dict[str, bool]
    details: dict[str, str] = field(default_factory=dict)

    @property
    def all_stopped(self) -> bool:
        return self.stopped == self.total


async def run_traps(limits: Limits = Limits()) -> TrapReport:  # noqa: B008 - Limits is frozen
    """Run every trap against a ``CodeMode`` built with ``limits``, one trap at a time.

    An unexpected error inside a trap (its evidence check included) counts as NOT
    stopped, and its detail says why.
    """
    from astromesh_codemode.eval.traps import TRAPS

    by_trap: dict[str, bool] = {}
    details: dict[str, str] = {}
    for name, trap in TRAPS.items():
        try:
            outcome = await trap(limits)
        except Exception as exc:  # noqa: BLE001 - a broken trap is reported, not raised
            by_trap[name], details[name] = False, f"trap error: {type(exc).__name__}: {exc}"
        else:
            by_trap[name], details[name] = outcome.stopped, outcome.detail
    return TrapReport(
        total=len(by_trap), stopped=sum(by_trap.values()), by_trap=by_trap, details=details
    )


def format_report(report: TrapReport) -> str:
    """The report as a plain-text table (trap, stopped, evidence)."""
    width = max([len("trap"), *map(len, report.by_trap)])
    lines = [f"{'trap':<{width}}  stopped  evidence", f"{'-' * width}  -------  --------"]
    for name, stopped in report.by_trap.items():
        mark = "yes" if stopped else "NO"
        lines.append(f"{name:<{width}}  {mark:<7}  {report.details.get(name, '')}")
    lines.append(f"\n{report.stopped}/{report.total} traps stopped")
    return "\n".join(lines)
