"""``python -m astromesh_codemode.eval traps``: run the traps with the default ``Limits``.

Exit status: 0 when every trap was stopped, 1 when some were not.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from astromesh_codemode.eval import format_report, run_traps


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m astromesh_codemode.eval", description="Security evals for the sandbox."
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("traps", help="run the malicious programs with the default limits")
    return p


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    report = asyncio.run(run_traps())
    print(format_report(report))
    return 0 if report.all_stopped else 1


if __name__ == "__main__":
    sys.exit(main())
