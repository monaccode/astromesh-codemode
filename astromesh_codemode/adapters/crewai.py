"""The two Code Mode tools as CrewAI ``BaseTool``s.

``run()`` (what crews call) goes through a background loop per ``CodeMode`` (see
``_common``); ``arun()`` awaits directly.
"""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import (
    async_handler,
    model_from_schema,
    require,
    sync_handler,
)
from astromesh_codemode.codemode import CodeMode, NeutralTool


def _tool(cm: CodeMode, tools: Any, pydantic: Any, t: NeutralTool) -> Any:
    sync_call, async_call = sync_handler(cm, t), async_handler(t)
    args_model = model_from_schema(f"{t.name}_args", t.parameters, pydantic)

    class CodeModeTool(tools.BaseTool):
        name: str = t.name
        description: str = t.description
        args_schema: type[pydantic.BaseModel] = args_model

        def _run(self, **kwargs: Any) -> str:
            return sync_call(**kwargs)

        async def _arun(self, **kwargs: Any) -> str:
            return await async_call(**kwargs)

    return CodeModeTool()


def as_tools(cm: CodeMode) -> list[Any]:
    tools = require("crewai.tools", "crewai")
    pydantic = require("pydantic", "crewai")
    return [_tool(cm, tools, pydantic, t) for t in cm.tool_definitions()]
