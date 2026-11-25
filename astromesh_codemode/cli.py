"""``astromesh-codemode`` command line."""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="astromesh-codemode", description="Code Mode for AI agents.")
    sub = p.add_subparsers(dest="command", required=True)
    mcp = sub.add_parser("mcp", help="run the MCP proxy (Code Mode over upstream MCP servers)")
    mcp.add_argument("--config", required=True, help="path to the upstream YAML config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import anyio

        from astromesh_codemode.mcp_server import ProxyConfig, serve_stdio
    except ImportError:
        print("error: the MCP proxy needs: pip install 'astromesh-codemode[mcp]'", file=sys.stderr)
        return 2
    try:
        config = ProxyConfig.from_yaml(args.config)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    anyio.run(serve_stdio, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
