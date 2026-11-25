from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from astromesh_codemode import CodeMode, ToolError
from astromesh_codemode.sources import from_function, from_mcp


def f(a: int, b: str = "x") -> list[str]:
    """Repeat b, a times.

    Second paragraph.
    """
    return [b] * a


async def test_from_function_schema():
    spec = from_function(f)
    assert spec.name == "f"
    assert spec.description == "Repeat b, a times."
    assert spec.input_schema["required"] == ["a"]
    assert spec.input_schema["properties"]["a"] == {"type": "integer"}
    assert spec.input_schema["properties"]["b"]["type"] == "string"
    assert spec.output_schema == {"type": "array", "items": {"type": "string"}}
    assert spec.writes is False
    assert await spec.handler(a=2, b="y") == ["y", "y"]


async def test_from_function_overrides_and_async():
    async def g(x: int | None = None) -> dict[str, int]:
        return {"x": x or 0}

    spec = from_function(g, name="gg", description="d", writes=True)
    assert (spec.name, spec.description, spec.writes) == ("gg", "d", True)
    assert await spec.handler(x=3) == {"x": 3}


def _server() -> FastMCP:
    server = FastMCP("t")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def lookup(key: str) -> str:
        """Look a key up."""
        return f"value:{key}"

    @server.tool()
    def mutate(key: str) -> str:
        """Change something."""
        return key

    @server.tool()
    def listed(key: str) -> dict[str, str]:
        """Listed as read-only by the caller."""
        return {"k": key}

    @server.tool()
    def fail() -> str:
        """Always fails."""
        raise ValueError("nope")

    return server


async def test_from_mcp_readonly_rule():
    async with create_connected_server_and_client_session(_server()._mcp_server) as session:
        specs = await from_mcp(session, prefix="crm_", read_only={"listed"})
    by_name = {s.name: s for s in specs}
    assert by_name["crm_lookup"].writes is False
    assert by_name["crm_mutate"].writes is True
    assert by_name["crm_listed"].writes is False
    assert by_name["crm_lookup"].input_schema["required"] == ["key"]


async def test_from_mcp_call():
    async with create_connected_server_and_client_session(_server()._mcp_server) as session:
        specs = await from_mcp(session, read_only={"listed"})
        cm = CodeMode(specs)
        assert "value:a" in await cm.run("await lookup(key='a')")
        assert "k" in await cm.run("await listed(key='z')")
        # a write tool is not reachable without allow_writes
        assert "NameError" in await cm.run("await mutate(key='a')")


async def test_from_mcp_error_becomes_toolerror():
    async with create_connected_server_and_client_session(_server()._mcp_server) as session:
        specs = await from_mcp(session, read_only={"fail"})
        fail = next(s for s in specs if s.name == "fail")
        with pytest.raises(ToolError, match="nope"):
            await fail.handler()
