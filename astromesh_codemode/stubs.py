"""Python stubs rendered from JSON Schema; this is what the model reads."""

from __future__ import annotations

import json
import keyword
import re

from astromesh_codemode.spec import ToolSpec

_SIMPLE = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "object": "dict",
    "null": "None",
}


def py_name(tool_name: str) -> str:
    name = re.sub(r"\W", "_", tool_name)
    return f"_{name}" if name[:1].isdigit() else name


def annotation(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "Any"
    if schema.get("enum"):
        return "Literal[" + ", ".join(json.dumps(v) for v in schema["enum"]) + "]"
    t = schema.get("type")
    if isinstance(t, list):
        parts = [annotation({**schema, "type": x}) for x in t]
        return " | ".join(dict.fromkeys(parts)) or "Any"
    if t == "array":
        items = schema.get("items")
        return f"list[{annotation(items)}]" if isinstance(items, dict) and items else "list"
    return _SIMPLE.get(t, "Any")


def _escape(text: str) -> str:
    return " ".join(text.split()).replace("\\", "\\\\").replace('"', "'")


def _is_valid_param(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def render_stub(spec: ToolSpec) -> str:
    props = spec.input_schema.get("properties", {}) or {}
    required = set(spec.input_schema.get("required", []) or [])
    valid = {n: s for n, s in props.items() if _is_valid_param(n)}
    invalid = {n: s for n, s in props.items() if not _is_valid_param(n)}
    params = [f"{n}: {annotation(s)}" for n, s in valid.items() if n in required]
    params += [f"{n}: {annotation(s)} | None = None" for n, s in valid.items() if n not in required]
    if invalid:
        params.append("**kwargs: Any")
    ret = annotation(spec.output_schema) if spec.output_schema else "Any"
    sig = f"async def {py_name(spec.name)}({', '.join(params)}) -> {ret}:"
    desc = _escape(spec.description)
    if not invalid:
        return f'{sig}\n    """{desc}"""\n    ...'
    lines = [f'    """{desc}', ""]
    for n, s in invalid.items():
        req = ", required" if n in required else ""
        lines.append(f'    - "{_escape(n)}": {annotation(s)}{req}')
    lines.append('    """')
    return sig + "\n" + "\n".join(lines) + "\n    ..."


def index_by_py_name(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    index: dict[str, ToolSpec] = {}
    for spec in specs:
        key = py_name(spec.name)
        if key in index:
            raise ValueError(
                f"tool names {index[key].name!r} and {spec.name!r} both map to "
                f"the Python name {key!r}"
            )
        index[key] = spec
    return index
