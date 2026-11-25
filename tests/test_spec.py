import astromesh_codemode
from astromesh_codemode import ToolSpec


async def handler(**kwargs):
    return kwargs


def test_toolspec_defaults():
    spec = ToolSpec("a", "d", {"type": "object"}, handler)
    assert spec.writes is False
    assert spec.output_schema is None


def test_version():
    assert astromesh_codemode.__version__ == "0.1.0"
