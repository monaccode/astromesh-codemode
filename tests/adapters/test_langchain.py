import pytest

pytest.importorskip("langchain_core")

from astromesh_codemode.adapters.langchain import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


async def test_langchain_async():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert tools[1].description == cm.tool_definitions()[1].description
    assert set(tools[1].args) == {"code"}
    assert "3" in await tools[1].ainvoke({"code": CODE})
    assert "get_rows" in await tools[0].ainvoke({"query": "rows"})
    await cm.aclose()


def test_langchain_sync_reuses_pool_across_calls():
    cm = make_cm()
    run = as_tools(cm)[1]
    assert "3" in run.invoke({"code": CODE})
    assert "3" in run.invoke({"code": CODE})
