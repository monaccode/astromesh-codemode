import pytest

pytest.importorskip("llama_index.core")

from astromesh_codemode.adapters.llamaindex import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


def test_llamaindex_sync_reuses_pool():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.metadata.name for t in tools] == NAMES
    assert set(tools[1].metadata.fn_schema.model_json_schema()["properties"]) == {"code"}
    assert "3" in tools[1].call(code=CODE).content
    assert "3" in tools[1].call(code=CODE).content


async def test_llamaindex_async():
    cm = make_cm()
    tools = as_tools(cm)
    assert "3" in (await tools[1].acall(code=CODE)).content
    assert "get_rows" in (await tools[0].acall(query="rows")).content
    await cm.aclose()
