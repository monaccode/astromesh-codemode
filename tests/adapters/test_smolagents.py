import pytest

pytest.importorskip("smolagents")

from astromesh_codemode.adapters.smolagents import as_tools
from tests.adapters._helpers import CODE, NAMES, make_cm


def test_smolagents_sync_reuses_pool():
    cm = make_cm()
    tools = as_tools(cm)
    assert [t.name for t in tools] == NAMES
    assert set(tools[1].inputs) == {"code"}
    assert "limit" in tools[0].inputs
    assert "3" in tools[1](code=CODE)
    assert "3" in tools[1](code=CODE)
    assert "get_rows" in tools[0](query="rows")
