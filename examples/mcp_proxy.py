"""Start the Code Mode MCP proxy over examples/upstream.yaml and use it as an MCP client would.

    pip install 'astromesh-codemode[mcp]'     # and Node.js, for the npx upstream
    python examples/mcp_proxy.py

The proxy (``astromesh-codemode mcp --config examples/upstream.yaml``) connects to the
filesystem MCP server, serving the current directory, and exposes only ``codemode_search``
and ``codemode_run``. This script launches it over stdio, lists its tools, searches, and
runs one program that calls an upstream tool inside the sandbox.
"""

import asyncio
import sys
from pathlib import Path

CONFIG = Path(__file__).with_name("upstream.yaml")

PROGRAM = """
listing = await fs_list_directory(path=".")
# the filesystem server sends {"content": "<one line per entry>"}
text = listing["content"] if isinstance(listing, dict) else listing
lines = [line for line in text.splitlines() if line.strip()]
{"entries": len(lines), "dirs": [line[6:] for line in lines if line.startswith("[DIR]")]}
"""


async def main() -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("needs: pip install 'astromesh-codemode[mcp]'")
        return 2

    proxy = StdioServerParameters(
        command=sys.executable,
        args=["-m", "astromesh_codemode.cli", "mcp", "--config", str(CONFIG)],
    )
    async with stdio_client(proxy) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        print("proxy tools:", [t.name for t in tools])  # codemode_search, codemode_run

        found = await session.call_tool(
            "codemode_search", {"query": "list directory files", "limit": 3}
        )
        print(found.content[0].text)

        ran = await session.call_tool("codemode_run", {"code": PROGRAM})
        print(ran.content[0].text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
