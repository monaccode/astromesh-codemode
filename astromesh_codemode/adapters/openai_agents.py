"""The two Code Mode tools as OpenAI Agents SDK ``FunctionTool``s (async)."""

from __future__ import annotations

import json
from typing import Any

from astromesh_codemode.adapters._common import require
from astromesh_codemode.codemode import CodeMode, NeutralTool


def _tool(agents: Any, t: NeutralTool) -> Any:
    async def invoke(_ctx: Any, raw: str) -> str:
        args = json.loads(raw or "{}")
        return await t.handler(**{k: v for k, v in args.items() if v is not None})

    return agents.FunctionTool(
        name=t.name,
        description=t.description,
        params_json_schema=t.parameters,
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )


def as_tools(cm: CodeMode) -> list[Any]:
    agents = require("agents", "openai-agents")
    return [_tool(agents, t) for t in cm.tool_definitions()]
