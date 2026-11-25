"""``python -m astromesh_codemode.eval``: the evals from the command line.

- ``traps``: run the traps with the default ``Limits``. Exit status 0 when every trap
  was stopped, 1 when some were not.
- ``cost --level 1``: the cost eval with a scripted model; writes
  ``<out>/<YYYY-MM-DD>-L1.json`` and ``.md`` (``out`` defaults to ``results``).
- ``cost --level 2 --models a,b [--repeats 3] [--scenarios 64/large,...]``: the same
  scenarios with real models through litellm (this spends money); writes
  ``<out>/<YYYY-MM-DD>-L2.json`` and ``.md``. Exits 2 without calling any model when a
  requested model's API key env var is not set, or when it has no known price.
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
    cost.add_argument("--level", type=int, choices=[1, 2], required=True)
    cost.add_argument("--out", type=Path, default=Path("results"))
    cost.add_argument("--models", default="", help="level 2: comma-separated litellm model names")
    cost.add_argument("--repeats", type=int, default=3, help="level 2: attempts per row")
    cost.add_argument(
        "--scenarios",
        default="",
        help="level 2: comma-separated <catalog>/<data>, e.g. 64/large (default: all six)",
    )
    return p


def _write(out: Path, name: str, payload: dict, md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on every OS, so a rerun on another machine diffs clean.
    text = json.dumps(payload, indent=2) + "\n"
    (out / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
    (out / f"{name}.md").write_text(md, encoding="utf-8", newline="\n")


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def _cost_level1(out: Path) -> int:
    from astromesh_codemode.eval.cost import format_markdown, run_level1, to_json
    from astromesh_codemode.eval.scenarios import SCENARIOS

    rows = asyncio.run(run_level1(SCENARIOS))
    date = _today()
    md = format_markdown(rows, date)
    _write(out, f"{date}-L1", to_json(rows, date), md)
    print(md)
    return 0


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _parse_scenarios(spec: str) -> list:
    from astromesh_codemode.eval.scenarios import SCENARIOS, Scenario

    if not spec.strip():
        return list(SCENARIOS)
    out = []
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        size, _, data = item.partition("/")
        try:
            out.append(Scenario(int(size), data))  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(
                f"bad scenario {item!r} (want <catalog>/<small|large>): {exc}"
            ) from exc
    return out


def _cost_level2(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        return _fail("--level 2 needs --models, e.g. --models gpt-4o-mini,gpt-4o")
    if args.repeats < 1:
        return _fail(f"--repeats must be >= 1, got {args.repeats}")
    try:
        scenarios = _parse_scenarios(args.scenarios)
    except ValueError as exc:
        return _fail(str(exc))

    from astromesh_codemode.eval.cost import (
        format_markdown_l2,
        max_model_requests,
        missing_api_keys,
        price_per_token,
        run_level2,
        to_json_l2,
    )

    # Refuse before spending anything: every model needs its key and a known price.
    for m in models:
        missing = missing_api_keys(m)
        if missing:
            return _fail(
                f"no API key for {m!r}: set {' or '.join(missing)}. "
                "Level 2 calls real models and spends money; nothing was called."
            )
        if price_per_token(m) is None:
            return _fail(f"no price for {m!r} in litellm.model_cost; nothing was called.")

    calls, requests = max_model_requests(len(models), len(scenarios), args.repeats)
    print(
        f"Level 2: {len(models)} model(s) x {len(scenarios)} scenario(s) x 2 strategies x "
        f"{args.repeats} repeat(s). Upper bound: {calls} model calls "
        f"({requests} requests counting retries)."
    )
    rows = asyncio.run(run_level2(scenarios, models, args.repeats))
    date = _today()
    md = format_markdown_l2(rows, date)
    _write(args.out, f"{date}-L2", to_json_l2(rows, date, args.repeats), md)
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cost":
        return _cost_level1(args.out) if args.level == 1 else _cost_level2(args)
    report = asyncio.run(run_traps())
    print(format_report(report))
    return 0 if report.all_stopped else 1


if __name__ == "__main__":
    sys.exit(main())
