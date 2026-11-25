import pytest

pytest.importorskip("crewai")

from astromesh_codemode.adapters.crewai import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


def test_crewai_sync_reuses_pool():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert set(tools[1].args_schema.model_json_schema()["properties"]) == {"code"}
    assert "3" in tools[1].run(code=CODE)
    assert "3" in tools[1].run(code=CODE)
    assert "get_rows" in tools[0].run(query="rows")


async def test_crewai_async():
    cm = make_cm()
    tools = as_tools(cm)
    assert "3" in await tools[1]._arun(code=CODE)
    await cm.aclose()
