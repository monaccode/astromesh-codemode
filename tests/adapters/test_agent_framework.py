import pytest

pytest.importorskip("agent_framework")

from astromesh_codemode.adapters.agent_framework import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


async def _call(tool, **kw):
    return (await tool.invoke(arguments=None, **kw))[0].text


async def test_agent_framework():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert set(tools[1].parameters()["properties"]) == {"code"}
    assert "3" in await _call(tools[1], code=CODE)
    assert "get_rows" in await _call(tools[0], query="rows")
    await cm.aclose()
