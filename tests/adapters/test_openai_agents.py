import json

import pytest

pytest.importorskip("agents")

from astromesh_codemode.adapters.openai_agents import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


async def _call(tool, **kw):
    from agents.tool_context import ToolContext

    raw = json.dumps(kw)
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="c1", tool_arguments=raw)
    return await tool.on_invoke_tool(ctx, raw)


async def test_openai_agents():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert set(tools[1].params_json_schema["properties"]) == {"code"}
    assert "3" in await _call(tools[1], code=CODE)
    assert "get_rows" in await _call(tools[0], query="rows")
    await cm.aclose()
