"""The two Code Mode tools as Google ADK ``BaseTool``s (async)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import require
from astromesh_codemode.codemode import CodeMode, NeutralTool


def _tool_class(tools: Any, genai_types: Any) -> Any:
    class CodeModeTool(tools.BaseTool):
        def __init__(self, neutral: NeutralTool):
            super().__init__(name=neutral.name, description=neutral.description)
            self._neutral = neutral

        def _get_declaration(self) -> Any:
            return genai_types.FunctionDeclaration(
                name=self.name,
                description=self.description,
                parameters_json_schema=self._neutral.parameters,
            )

        async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> str:
            return await self._neutral.handler(**{k: v for k, v in args.items() if v is not None})

    return CodeModeTool


def as_tools(cm: CodeMode) -> list[Any]:
    tools = require("google.adk.tools", "adk")
    types = require("google.genai.types", "adk")
    cls = _tool_class(tools, types)
    return [cls(t) for t in cm.tool_definitions()]
