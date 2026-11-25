"""``CodeMode``: a group of tools exposed to the model as two tools, ``search`` and ``run``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Self

import pydantic_monty as monty

from astromesh_codemode.limits import Limits
from astromesh_codemode.sandbox import execute, make_validators
from astromesh_codemode.search import MAX_LIMIT, Index
from astromesh_codemode.spec import ToolSpec
from astromesh_codemode.stubs import index_by_py_name, render_stub
from astromesh_codemode.trace import Trace


@dataclass(frozen=True)
class NeutralTool:
    """A framework-neutral tool definition; adapters turn it into their own type."""

    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[str]]


class CodeMode:
    """Wraps tools behind ``{name}_search`` and ``{name}_run``.

    v0.1 is read-only: tools with ``writes=True`` are invisible to ``search`` and
    absent from the sandbox unless ``allow_writes=True``.

    One Monty worker pool is started lazily on the first ``run`` and closed by
    ``aclose()`` (or ``async with``). Every ``run`` checks out a fresh session, so
    no state survives between runs, and concurrent runs never share counters.
    """

    def __init__(
        self,
        tools: list[ToolSpec],
        *,
        name: str = "codemode",
        limits: Limits = Limits(),  # noqa: B008 - Limits is frozen
        allow_writes: bool = False,
    ):
        index_by_py_name(tools)  # name collisions are an error even among hidden tools
        exposed = [t for t in tools if allow_writes or not t.writes]
        self.name = name
        self.limits = limits
        self.allow_writes = allow_writes
        self._tools = index_by_py_name(exposed)
        self._validators = make_validators(self._tools)
        self._index = Index(exposed)
        self._pool: monty.AsyncMonty | None = None
        self._pool_lock = asyncio.Lock()
        self.last_trace: Trace | None = None
        """Trace of the most recently *finished* ``run``. With concurrent runs the last
        one to finish wins; use ``run_traced`` to get each run's own trace."""

    async def search(self, query: str, limit: int = 8) -> str:
        """Typed Python stubs of the tools matching ``query`` (at most 25)."""
        specs = self._index.search(query, limit)
        if not specs:
            return f"No tools match {query!r}. Try other words, or an empty query to list tools."
        return "\n\n".join(render_stub(s) for s in specs)

    async def run(self, code: str) -> str:
        """Run ``code`` in the sandbox and return its output as text."""
        text, trace = await self.run_traced(code)
        self.last_trace = trace
        return text

    async def run_traced(self, code: str) -> tuple[str, Trace]:
        """Like ``run`` but returns this run's own trace instead of setting ``last_trace``."""
        pool = await self._get_pool()
        return await execute(pool, code, self._tools, self._validators, self.limits)

    def tool_definitions(self) -> list[NeutralTool]:
        search_name, run_name = f"{self.name}_search", f"{self.name}_run"
        search = NeutralTool(
            name=search_name,
            description=(
                "Find tools by keyword. Returns their Python signatures (async def stubs) "
                f"with descriptions. Always use this before {run_name}: only the functions "
                "it shows exist, and their parameters are exactly as written. An empty "
                "query lists tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words describing the task."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_LIMIT,
                        "default": 8,
                        "description": "Maximum number of tools to return.",
                    },
                },
                "required": ["query"],
            },
            handler=self.search,
        )
        lim = self.limits
        run = NeutralTool(
            name=run_name,
            description=(
                f"Run a Python program that calls the tools found with {search_name}. "
                f"Search first; do not guess tool names. Call every tool with await and "
                "keyword arguments, e.g. rows = await get_rows(n=10); asyncio.gather works "
                "for parallel calls. The value of the program is its last expression (a "
                "top-level return also works). print() output is captured and returned too. "
                f"Output is capped at {lim.max_output_bytes} bytes, so filter and aggregate "
                "in code and return only what is needed. Tool failures raise exceptions you "
                f"can catch with try/except. Limits per run: {lim.max_calls} tool calls, "
                f"{lim.timeout_s:g}s. Each run starts empty: no variables carry over, and "
                "there is no filesystem or network access."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source."}},
                "required": ["code"],
            },
            handler=self.run,
        )
        return [search, run]

    async def _get_pool(self) -> monty.AsyncMonty:
        async with self._pool_lock:
            if self._pool is None:
                pool = monty.AsyncMonty()
                await pool.__aenter__()
                self._pool = pool
            return self._pool

    async def aclose(self) -> None:
        """Stop the worker pool. Safe to call twice; a later ``run`` starts a new pool."""
        async with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            await pool.__aexit__(None, None, None)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"CodeMode(name={self.name!r}, tools={len(self._tools)})"
