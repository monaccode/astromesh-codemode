"""The two Code Mode tools as smolagents ``Tool``s (sync, through a background loop)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import require, sync_handler
from astromesh_codemode.codemode import CodeMode, NeutralTool

_TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def _inputs(t: NeutralTool) -> dict[str, dict[str, Any]]:
    required = set(t.parameters.get("required", ()))
    inputs: dict[str, dict[str, Any]] = {}
    for key, prop in t.parameters.get("properties", {}).items():
        spec: dict[str, Any] = {
            "type": _TYPES.get(prop.get("type"), "any"),
            "description": prop.get("description", key),
        }
        if key not in required:
            spec["nullable"] = True
        inputs[key] = spec
    return inputs


def _tool(cm: CodeMode, smol: Any, t: NeutralTool) -> Any:
    call = sync_handler(cm, t)

    class CodeModeTool(smol.Tool):
        name = t.name
        description = t.description
        inputs = _inputs(t)
        output_type = "string"
        skip_forward_signature_validation = True  # forward takes the schema's keys generically

        def forward(self, **kwargs: Any) -> str:
            return call(**kwargs)

    return CodeModeTool()


def as_tools(cm: CodeMode) -> list[Any]:
    smol = require("smolagents", "smolagents")
    return [_tool(cm, smol, t) for t in cm.tool_definitions()]
