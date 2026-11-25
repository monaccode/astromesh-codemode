"""Turn the tools of a live MCP server into ToolSpecs.

Read-only rule: an MCP tool is a write unless the server marks it ``readOnlyHint: true``
or the caller lists it in ``read_only``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astromesh_codemode.spec import ToolError, ToolSpec

if TYPE_CHECKING:
    from mcp import ClientSession


def _text(result: Any) -> str:
    return "\n".join(p.text for p in result.content if getattr(p, "type", None) == "text")


def _make_handler(session: ClientSession, tool_name: str):
    async def handler(**kwargs: Any) -> Any:
        result = await session.call_tool(tool_name, kwargs)
        if result.isError:
            raise ToolError(_text(result) or f"{tool_name} failed")
        if result.structuredContent is not None:
            return result.structuredContent
        return _text(result)

    return handler


async def from_mcp(
    session: ClientSession,
    *,
    prefix: str = "",
    read_only: set[str] | frozenset[str] = frozenset(),
) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    cursor: str | None = None
    while True:
        page = await session.list_tools(cursor=cursor)
        for tool in page.tools:
            hinted = bool(tool.annotations and tool.annotations.readOnlyHint)
            specs.append(
                ToolSpec(
                    name=f"{prefix}{tool.name}",
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    handler=_make_handler(session, tool.name),
                    output_schema=tool.outputSchema,
                    writes=not (hinted or tool.name in read_only),
                )
            )
        cursor = page.nextCursor
        if not cursor:
            return specs
