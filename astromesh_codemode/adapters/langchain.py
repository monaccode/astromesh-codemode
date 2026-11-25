"""The two Code Mode tools as LangChain ``StructuredTool``s (sync and async both work)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import async_handler, require, sync_handler
from astromesh_codemode.codemode import CodeMode


def as_tools(cm: CodeMode) -> list[Any]:
    lc = require("langchain_core.tools", "langchain")
    return [
        lc.StructuredTool(
            name=t.name,
            description=t.description,
            args_schema=t.parameters,
            func=sync_handler(cm, t),
            coroutine=async_handler(t),
        )
        for t in cm.tool_definitions()
    ]
