import pytest

pytest.importorskip("google.adk")

from astromesh_codemode.adapters.adk import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


async def test_adk():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    decl = tools[1]._get_declaration()
    assert "code" in str(decl.model_dump(exclude_none=True)["parameters_json_schema"])
    assert "3" in await tools[1].run_async(args={"code": CODE}, tool_context=None)
    assert "get_rows" in await tools[0].run_async(args={"query": "rows"}, tool_context=None)
    await cm.aclose()
