"""The two Code Mode tools as LlamaIndex ``FunctionTool``s (``call`` and ``acall``)."""

from __future__ import annotations

from typing import Any

from astromesh_codemode.adapters._common import (
    async_handler,
    model_from_schema,
    require,
    sync_handler,
)
from astromesh_codemode.codemode import CodeMode


def as_tools(cm: CodeMode) -> list[Any]:
    li = require("llama_index.core.tools", "llamaindex")
    pydantic = require("pydantic", "llamaindex")
    return [
        li.FunctionTool.from_defaults(
            fn=sync_handler(cm, t),
            async_fn=async_handler(t),
            name=t.name,
            description=t.description,
            fn_schema=model_from_schema(f"{t.name}_args", t.parameters, pydantic),
        )
        for t in cm.tool_definitions()
    ]
