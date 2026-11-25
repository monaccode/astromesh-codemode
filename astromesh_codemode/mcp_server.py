"""MCP proxy: expose N upstream MCP servers as one Code Mode server (search + run)."""

from __future__ import annotations

import dataclasses
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import FastMCP

from astromesh_codemode.codemode import CodeMode
from astromesh_codemode.limits import Limits
from astromesh_codemode.sources.mcp import from_mcp

_UPSTREAM_KEYS = {"name", "command", "args", "url", "read_only"}
_LIMIT_KEYS = {f.name for f in dataclasses.fields(Limits)}


@dataclass(frozen=True)
class UpstreamConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    read_only: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProxyConfig:
    upstreams: list[UpstreamConfig]
    name: str = "codemode"
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_dict(cls, data: Any) -> ProxyConfig:
        if not isinstance(data, dict):
            raise ValueError("config must be a mapping")
        unknown = set(data) - {"name", "limits", "upstreams"}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        raw = data.get("upstreams")
        if not isinstance(raw, list) or not raw:
            raise ValueError("'upstreams' must be a non-empty list")
        upstreams: list[UpstreamConfig] = []
        seen: set[str] = set()
        for i, u in enumerate(raw):
            if not isinstance(u, dict) or not isinstance(u.get("name"), str) or not u["name"]:
                raise ValueError(f"upstream #{i + 1}: 'name' is required")
            name = u["name"]
            if extra := set(u) - _UPSTREAM_KEYS:
                raise ValueError(f"upstream '{name}': unknown keys {sorted(extra)}")
            if name in seen:
                raise ValueError(f"duplicate upstream name '{name}'")
            seen.add(name)
            if (u.get("command") is None) == (u.get("url") is None):
                raise ValueError(f"upstream '{name}': set exactly one of 'command' or 'url'")
            args, read_only = u.get("args") or [], u.get("read_only") or []
            for key, val in (("args", args), ("read_only", read_only)):
                if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                    raise ValueError(f"upstream '{name}': '{key}' must be a list of strings")
            upstreams.append(
                UpstreamConfig(name, u.get("command"), list(args), u.get("url"), list(read_only))
            )
        raw_limits = data.get("limits") or {}
        if not isinstance(raw_limits, dict):
            raise ValueError("'limits' must be a mapping")
        if bad := set(raw_limits) - _LIMIT_KEYS:
            raise ValueError(f"unknown limits keys {sorted(bad)}; allowed {sorted(_LIMIT_KEYS)}")
        name = data.get("name", "codemode")
        if not isinstance(name, str) or not name:
            raise ValueError("'name' must be a non-empty string")
        return cls(upstreams, name, Limits(**raw_limits))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProxyConfig:
        import yaml

        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except OSError as e:
            raise ValueError(f"cannot read config {path}: {e.strerror or e}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML in {path}: {e}") from e
        return cls.from_dict(data)


async def _connect(stack: AsyncExitStack, up: UpstreamConfig) -> ClientSession:
    if up.command is not None:
        params = StdioServerParameters(command=up.command, args=up.args)
        read, write = await stack.enter_async_context(stdio_client(params))
    else:
        read, write, _ = await stack.enter_async_context(streamablehttp_client(up.url))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def build_server(
    config: ProxyConfig, *, sessions: dict[str, ClientSession] | None = None
) -> FastMCP:
    """Build the proxy server. Pass open ``sessions`` (by upstream name) to skip connecting.

    Call ``await server.aclose_codemode()`` to release upstream connections and the sandbox.
    """
    stack = AsyncExitStack()
    try:
        specs = []
        for up in config.upstreams:
            session = (sessions or {}).get(up.name) or await _connect(stack, up)
            specs += await from_mcp(session, prefix=f"{up.name}_", read_only=set(up.read_only))
        cm = CodeMode(specs, name=config.name, limits=config.limits)
    except BaseException:
        await stack.aclose()
        raise

    server = FastMCP(config.name)
    search_def, run_def = cm.tool_definitions()

    async def search(query: str, limit: int = 8) -> str:
        return await search_def.handler(query=query, limit=limit)

    async def run(code: str) -> str:
        return await run_def.handler(code=code)

    for fn, tool in ((search, search_def), (run, run_def)):
        server.add_tool(fn, name=tool.name, description=tool.description)
        # Publish the NeutralTool's JSON Schema unchanged.
        server._tool_manager.get_tool(tool.name).parameters = tool.parameters

    async def aclose_codemode() -> None:
        await cm.aclose()
        await stack.aclose()

    server.aclose_codemode = aclose_codemode  # type: ignore[attr-defined]
    return server


async def serve_stdio(config: ProxyConfig) -> None:
    server = await build_server(config)
    try:
        await server.run_stdio_async()
    finally:
        await server.aclose_codemode()  # type: ignore[attr-defined]
