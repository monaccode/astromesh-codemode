from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from astromesh_codemode.cli import main
from astromesh_codemode.codemode import CodeMode
from astromesh_codemode.mcp_server import ProxyConfig, build_server


def _upstream() -> FastMCP:
    server = FastMCP("crm")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_user(user_id: str) -> dict[str, str]:
        """Fetch a user by id."""
        return {"id": user_id, "name": "Ada"}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def list_orders(user_id: str) -> list[int]:
        """List order ids of a user."""
        return [1, 2, 3]

    @server.tool()
    def delete_user(user_id: str) -> str:
        """Delete a user."""
        return "deleted"

    return server


def _config(**kw) -> ProxyConfig:
    return ProxyConfig.from_dict({"upstreams": [{"name": "crm", "command": "x"}], **kw})


async def test_proxy_exposes_two_tools_and_runs_reads():
    async with create_connected_server_and_client_session(_upstream()._mcp_server) as up:
        server = await build_server(_config(), sessions={"crm": up})
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            tools = (await client.list_tools()).tools
            assert sorted(t.name for t in tools) == ["codemode_run", "codemode_search"]
            run = next(t for t in tools if t.name == "codemode_run")
            assert run.inputSchema["required"] == ["code"]

            found = await client.call_tool("codemode_search", {"query": ""})
            text = found.content[0].text
            assert "crm_get_user" in text
            assert "crm_list_orders" in text
            assert "crm_delete_user" not in text

            res = await client.call_tool(
                "codemode_run", {"code": "u = await crm_get_user(user_id='7')\nu['name']"}
            )
            assert "Ada" in res.content[0].text

            res = await client.call_tool(
                "codemode_run", {"code": "await crm_delete_user(user_id='1')"}
            )
            assert "NameError" in res.content[0].text
        await server.aclose_codemode()


async def test_read_only_list_and_custom_name():
    async with create_connected_server_and_client_session(_upstream()._mcp_server) as up:
        cfg = ProxyConfig.from_dict(
            {
                "name": "gw",
                "upstreams": [{"name": "crm", "command": "x", "read_only": ["delete_user"]}],
            }
        )
        server = await build_server(cfg, sessions={"crm": up})
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            assert names == ["gw_run", "gw_search"]
            text = (await client.call_tool("gw_search", {"query": "delete"})).content[0].text
            assert "crm_delete_user" in text
        await server.aclose_codemode()


def _write(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_from_yaml_ok(tmp_path):
    p = _write(
        tmp_path,
        "name: gw\nlimits: {max_calls: 5, timeout_s: 3}\nupstreams:\n"
        "  - name: a\n    command: npx\n    args: [-y, srv]\n    read_only: [x]\n"
        "  - name: b\n    url: http://localhost:9/mcp\n",
    )
    cfg = ProxyConfig.from_yaml(p)
    assert cfg.name == "gw"
    assert cfg.limits.max_calls == 5
    assert cfg.upstreams[0].args == ["-y", "srv"]
    assert cfg.upstreams[0].read_only == ["x"]
    assert cfg.upstreams[1].url == "http://localhost:9/mcp"


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("upstreams: [{name: a}]", "a"),
        ("upstreams: [{name: a, command: x, url: http://h}]", "a"),
        ("upstreams: []", "upstreams"),
        ("upstreams: [{command: x}]", "name"),
        ("upstreams: [{name: a, command: x}, {name: a, url: http://h}]", "duplicate"),
        ("upstreams: [{name: a, command: x}]\nlimits: {bogus: 1}", "bogus"),
        ("upstreams: [{name: a, command: x, wat: 1}]", "wat"),
    ],
)
def test_from_yaml_invalid(tmp_path, body, needle):
    with pytest.raises(ValueError, match=needle):
        ProxyConfig.from_yaml(_write(tmp_path, body))


def test_cli_bad_config_exits_2(tmp_path, capsys):
    assert main(["mcp", "--config", str(tmp_path / "missing.yaml")]) == 2
    err = capsys.readouterr().err
    assert err.strip().count("\n") == 0
    assert "missing.yaml" in err
    assert main(["mcp", "--config", str(_write(tmp_path, "upstreams: []"))]) == 2


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "mcp" in capsys.readouterr().out


async def test_published_schema_is_the_neutral_schema():
    neutral = {t.name: t for t in CodeMode([], name="codemode").tool_definitions()}
    async with create_connected_server_and_client_session(_upstream()._mcp_server) as up:
        server = await build_server(_config(), sessions={"crm": up})
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 2
            for t in tools:
                assert t.inputSchema == neutral[t.name].parameters
                assert t.description == neutral[t.name].description
            only_query = await client.call_tool("codemode_search", {"query": "user"})
            assert not only_query.isError
            assert "crm_get_user" in only_query.content[0].text
            with_limit = await client.call_tool("codemode_search", {"query": "", "limit": 1})
            assert not with_limit.isError
            assert with_limit.content[0].text.count("async def") == 1
        await server.aclose_codemode()
