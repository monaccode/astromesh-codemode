"""``python -m astromesh_codemode.eval``: the evals from the command line.

- ``traps``: run the traps with the default ``Limits``. Exit status 0 when every trap
  was stopped, 1 when some were not.
- ``cost --level 1``: the cost eval with a scripted model; writes
  ``<out>/<YYYY-MM-DD>-L1.json`` and ``.md`` (``out`` defaults to ``results``).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

from astromesh_codemode.eval import format_report, run_traps


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m astromesh_codemode.eval", description="Evals for Code Mode."
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("traps", help="run the malicious programs with the default limits")
    cost = sub.add_parser("cost", help="context cost: Code Mode vs direct tool calling")
    cost.add_argument("--level", type=int, choices=[1], required=True)
    cost.add_argument("--out", type=Path, default=Path("results"))
    return p


def _cost(level: int, out: Path) -> int:
    from astromesh_codemode.eval.cost import format_markdown, run_level1, to_json
    from astromesh_codemode.eval.scenarios import SCENARIOS

    rows = asyncio.run(run_level1(SCENARIOS))
    date = dt.datetime.now(dt.UTC).date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    md = format_markdown(rows, date)
    payload = json.dumps(to_json(rows, date), indent=2) + "\n"
    # newline="\n": the same bytes on every OS, so a rerun on another machine diffs clean.
    (out / f"{date}-L{level}.json").write_text(payload, encoding="utf-8", newline="\n")
    (out / f"{date}-L{level}.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cost":
        return _cost(args.level, args.out)
    report = asyncio.run(run_traps())
    print(format_report(report))
    return 0 if report.all_stopped else 1


if __name__ == "__main__":
    sys.exit(main())
