"""The two Code Mode tools as PydanticAI ``Tool``s (async)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import async_handler, require
from astromesh_codemode.codemode import CodeMode


def as_tools(cm: CodeMode) -> list[Any]:
    pai = require("pydantic_ai", "pydantic-ai")
    return [
        pai.Tool.from_schema(
            async_handler(t), name=t.name, description=t.description, json_schema=t.parameters
        )
        for t in cm.tool_definitions()
    ]
