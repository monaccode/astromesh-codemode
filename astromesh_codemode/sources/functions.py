"""Turn a plain Python function into a ToolSpec."""

from __future__ import annotations

import asyncio
import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from astromesh_codemode.spec import ToolSpec

_SCALARS: dict[Any, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    type(None): "null",
}


def _schema_for(tp: Any) -> dict:
    if tp is Any or tp is inspect.Parameter.empty:
        return {}
    if tp in _SCALARS:
        return {"type": _SCALARS[tp]}
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_schema_for(a) for a in args]}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if tp in (list, tuple, set, frozenset) or origin in (list, set, frozenset):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _schema_for(args[0])
        return schema
    if origin is tuple:
        return {"type": "array"}
    if tp is dict or origin is dict:
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _schema_for(args[1])
        return schema
    return {}


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())


def from_function(
    fn: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    writes: bool = False,
) -> ToolSpec:
    """Build a ToolSpec from type hints and the docstring; sync functions run in a thread."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable hints degrade to untyped
        hints = {}
    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[pname] = _schema_for(hints.get(pname, Any))
        if param.default is param.empty:
            required.append(pname)
    input_schema: dict = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required

    output_schema = None
    if "return" in hints and hints["return"] is not type(None):
        output_schema = _schema_for(hints["return"]) or None

    if inspect.iscoroutinefunction(fn):
        handler = fn
    else:

        async def handler(**kwargs: Any) -> Any:
            return await asyncio.to_thread(fn, **kwargs)

    return ToolSpec(
        name=name or fn.__name__,
        description=description if description is not None else _first_paragraph(fn.__doc__),
        input_schema=input_schema,
        handler=handler,
        output_schema=output_schema,
        writes=writes,
    )
