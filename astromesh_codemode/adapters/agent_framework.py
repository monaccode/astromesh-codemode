"""The two Code Mode tools as Microsoft Agent Framework ``FunctionTool``s (async)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import async_handler, require
from astromesh_codemode.codemode import CodeMode


def as_tools(cm: CodeMode) -> list[Any]:
    af = require("agent_framework", "agent-framework")
    return [
        af.tool(async_handler(t), name=t.name, description=t.description, schema=t.parameters)
        for t in cm.tool_definitions()
    ]
