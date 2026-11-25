import ast

import pytest

from astromesh_codemode.spec import ToolSpec
from astromesh_codemode.stubs import annotation, index_by_py_name, py_name, render_stub


async def _h(**kwargs):
    return None


def _spec(name, **kw):
    return ToolSpec(
        name=name,
        description=kw.pop("description", "d"),
        input_schema=kw.pop("input_schema", {"type": "object", "properties": {}}),
        handler=_h,
        **kw,
    )


def test_py_name():
    assert py_name("crm-get contact") == "crm_get_contact"
    assert py_name("1abc") == "_1abc"


def test_annotation_basic():
    assert annotation({"type": "string"}) == "str"
    assert annotation({"type": "integer"}) == "int"
    assert annotation({"type": "number"}) == "float"
    assert annotation({"type": "boolean"}) == "bool"
    assert annotation({"type": "array", "items": {"type": "string"}}) == "list[str]"
    assert annotation({"type": "object"}) == "dict"
    assert annotation({"type": ["string", "null"]}) == "str | None"
    assert annotation({"enum": ["a", "b"]}) == 'Literal["a", "b"]'
    assert annotation({}) == "Any"


def test_render_stub():
    spec = _spec(
        "crm-get contact",
        description="Fetch a contact.",
        input_schema={
            "type": "object",
            "properties": {
                "fields": {"type": "array", "items": {"type": "string"}},
                "id": {"type": "integer"},
            },
            "required": ["id"],
        },
    )
    out = render_stub(spec)
    assert "async def crm_get_contact(id: int, fields: list[str] | None = None) -> Any:" in out
    assert '    """Fetch a contact."""' in out
    compile(out, "<stub>", "exec")
    ast.parse(out)


def test_render_stub_output_schema_and_multiline_description():
    spec = _spec(
        "t",
        description='Line one\nline "two" \\ """x"""',
        output_schema={"type": "integer"},
    )
    out = render_stub(spec)
    assert "-> int:" in out
    assert len(out.splitlines()) == 3
    compile(out, "<stub>", "exec")


def test_collision():
    with pytest.raises(ValueError, match=r"get-user.*get_user") as exc:
        index_by_py_name([_spec("get-user"), _spec("get_user")])
    assert "get_user" in str(exc.value)


def test_index_ok():
    a, b = _spec("a-b"), _spec("c")
    assert index_by_py_name([a, b]) == {"a_b": a, "c": b}


def test_render_stub_invalid_param_names():
    spec = _spec(
        "t",
        description="Does things.",
        input_schema={
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "a-b": {"type": "integer"},
                "ok": {"type": "boolean"},
            },
            "required": ["from"],
        },
    )
    out = render_stub(spec)
    assert "ok: bool | None = None" in out
    assert "**kwargs: Any" in out
    assert '- "from": str, required' in out
    assert '- "a-b": int' in out
    ast.parse(out)
