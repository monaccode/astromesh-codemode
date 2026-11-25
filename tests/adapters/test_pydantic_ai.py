import pytest

pytest.importorskip("pydantic_ai")

from astromesh_codemode.adapters.pydantic_ai import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


async def test_pydantic_ai():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert set(tools[1].function_schema.json_schema["properties"]) == {"code"}
    assert "3" in await tools[1].function(code=CODE)
    assert "get_rows" in await tools[0].function(query="rows")
    await cm.aclose()
