# MCP: tools as a source, and the proxy

Install the extra: `pip install 'astromesh-codemode[mcp]'` (`mcp>=1.28,<2`, `pyyaml`).

There are two ways to use MCP servers with Code Mode:

- **`from_mcp(session)`**, in your own code: the tools of a live MCP `ClientSession` become
  `ToolSpec`s you put in a `CodeMode`, next to any others.
- **`astromesh-codemode mcp --config upstream.yaml`**, with no code: a proxy that connects
  to N upstream MCP servers and serves **one** MCP server exposing only `codemode_search`
  and `codemode_run`. Any MCP client (an IDE, a desktop app, an agent) points at the proxy
  instead of at the upstreams.

## The read-only rule

An MCP tool is treated as a **write** (hidden from `search`, absent from the sandbox)
unless:

- the server marks it with the annotation `readOnlyHint: true`, or
- you list it in `read_only` (the server's own tool name, without the prefix).

Many servers do not annotate their tools, so expect to list the ones you want. The proxy
never sets `allow_writes`: in v0.1 it serves reads only.

## `from_mcp`

```
await from_mcp(session, *, prefix="", read_only=frozenset()) -> list[ToolSpec]
```

It pages through `list_tools`, keeps each tool's `inputSchema` and `outputSchema`, and
calls `session.call_tool` when the program calls the tool. A result with `isError` raises
`ToolError` with the server's text (the program can catch it); otherwise the tool returns
`structuredContent` when the server sends it, or the joined text content. (FastMCP servers send a non-object return wrapped as
`{"result": ...}`, so a list comes back as `{"result": [1, 2, 3]}`.)

This example runs an MCP server in memory, so it needs no network:

```python
# needs: mcp
import asyncio

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from astromesh_codemode import CodeMode
from astromesh_codemode.sources import from_mcp

crm = FastMCP("crm")


@crm.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_user(user_id: str) -> dict[str, str]:
    """Fetch a user by id."""
    return {"id": user_id, "name": "Ada"}


@crm.tool()
def list_orders(user_id: str) -> list[int]:
    """List order ids of a user."""
    return [1, 2, 3]


@crm.tool()
def delete_user(user_id: str) -> str:
    """Delete a user."""
    return "deleted"


async def main():
    async with create_connected_server_and_client_session(crm._mcp_server) as session:
        # get_user is annotated read-only; list_orders is listed; delete_user stays hidden.
        specs = await from_mcp(session, prefix="crm_", read_only={"list_orders"})
        async with CodeMode(specs) as cm:
            print(await cm.search(""))
            program = "u = await crm_get_user(user_id='7')\nn = await crm_list_orders(user_id='7')\n[u, n]"
            print(await cm.run(program))
            print(await cm.run("await crm_delete_user(user_id='7')"))  # NameError


asyncio.run(main())
```

With a real server, open the session with the MCP SDK (`stdio_client` or
`streamablehttp_client`, then `ClientSession(...).initialize()`) and pass it the same way.
Keep the session open for as long as the `CodeMode` runs programs.

## The proxy

```bash
astromesh-codemode mcp --config examples/upstream.yaml
```

It serves MCP over stdio. The config ([`examples/upstream.yaml`](../examples/upstream.yaml)):

```yaml
# Proxy config: astromesh-codemode mcp --config examples/upstream.yaml
name: codemode            # server name; tools become codemode_search / codemode_run
limits:                   # optional; defaults shown
  timeout_s: 30
  max_calls: 50
upstreams:
  - name: fs              # tools are exposed as fs_<tool>
    command: npx          # stdio upstream (use `url:` instead for streamable HTTP)
    args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
    # Every upstream tool is treated as a write (hidden) unless it has
    # readOnlyHint: true or is listed here (unprefixed server tool names).
    read_only: [read_file, list_directory, search_files]
```

- **`name`** (default `codemode`): the server name, and the prefix of the two tools.
- **`limits`**: any `Limits` field (`timeout_s`, `max_memory`, `max_calls`,
  `max_output_bytes`). Unknown keys are an error.
- **`upstreams`** (required, non-empty): each one has a unique `name` (its tools are exposed
  as `<name>_<tool>`), exactly one of `command` (+ `args`, a stdio server) or `url` (a
  streamable HTTP server), and an optional `read_only` list.

A bad config exits with status 2 and one `error:` line; so does a missing `mcp` extra.

To use it from an MCP client, register the proxy as a stdio server. For example, in a
client that takes a JSON `mcpServers` map:

```json
{
  "mcpServers": {
    "codemode": {
      "command": "astromesh-codemode",
      "args": ["mcp", "--config", "/path/to/upstream.yaml"]
    }
  }
}
```

[`examples/mcp_proxy.py`](../examples/mcp_proxy.py) starts the proxy with this config as a
subprocess and calls both tools as a client would (it needs Node.js for `npx`).

### From Python

`ProxyConfig.from_yaml(path)` / `ProxyConfig.from_dict(data)` parse and validate the config
(raising `ValueError`). `build_server(config, sessions=None)` connects to the upstreams and
returns a `FastMCP` server; pass already-open `sessions` (by upstream name) to skip
connecting. Release it with `await server.aclose_codemode()`, which closes the sandbox
pool and the upstream connections. `serve_stdio(config)` does all of it and serves stdio.

```python
# needs: mcp, yaml
from astromesh_codemode.mcp_server import ProxyConfig

config = ProxyConfig.from_dict(
    {
        "name": "tools",
        "limits": {"max_calls": 20},
        "upstreams": [{"name": "crm", "command": "crm-mcp"}],
    }
)
print(config.name, config.limits.max_calls, [u.name for u in config.upstreams])

try:
    ProxyConfig.from_dict({"upstreams": [{"name": "crm"}]})
except ValueError as e:
    print(e)  # upstream 'crm': set exactly one of 'command' or 'url'
```

```python
# needs: network
from astromesh_codemode.mcp_server import build_server

config = ProxyConfig.from_yaml("examples/upstream.yaml")


async def serve_in_process():
    server = await build_server(config)  # connects to every upstream
    try:
        await server.run_stdio_async()
    finally:
        await server.aclose_codemode()


asyncio.run(serve_in_process())
```
