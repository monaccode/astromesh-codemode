"""What every framework adapter shares: the import guard, the sync bridge and schema helpers.

Loop affinity. ``CodeMode`` starts its Monty worker pool on the event loop of the first
``run`` and the pool belongs to that loop. Async framework entry points await the handler
directly, so they use the caller's loop. Sync entry points (smolagents, CrewAI ``run``,
LangChain ``invoke``, LlamaIndex ``call``) go through ``run_sync``: the call is dispatched
to ONE background loop thread per ``CodeMode``, so the pool is created once and reused by
every sync call. Pick one style per ``CodeMode``: a pool started by sync calls is bound to
the background loop, so awaiting ``run`` from your own loop afterwards fails; build one
``CodeMode`` per style.

A sync call made from a thread whose own event loop is running is refused (it would block
that loop): use the framework's async API there, or ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import weakref
from types import ModuleType
from typing import Any

from astromesh_codemode.codemode import CodeMode, NeutralTool

__all__ = [
    "MissingExtraError",
    "async_handler",
    "model_from_schema",
    "require",
    "run_sync",
    "sync_handler",
]


class MissingExtraError(ImportError):
    """The framework an adapter needs is not installed; the message names the extra."""


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise ``MissingExtraError`` naming ``astromesh-codemode[extra]``."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"{module} is not installed. Install it with: pip install 'astromesh-codemode[{extra}]'"
        ) from exc


_bridges: weakref.WeakKeyDictionary[CodeMode, asyncio.AbstractEventLoop] = (
    weakref.WeakKeyDictionary()
)
_bridges_lock = threading.Lock()


def _stop(loop: asyncio.AbstractEventLoop) -> None:
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)


def _bridge_loop(cm: CodeMode) -> asyncio.AbstractEventLoop:
    with _bridges_lock:
        loop = _bridges.get(cm)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name=f"astromesh-codemode-{cm.name}", daemon=True
            ).start()
            _bridges[cm] = loop
            weakref.finalize(cm, _stop, loop)
        return loop


def run_sync(cm: CodeMode, coro: Any) -> Any:
    """Run ``coro`` to completion from sync code on ``cm``'s background loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        raise RuntimeError(
            "astromesh-codemode: a sync tool call was made inside a running event loop; "
            "use the framework's async tool API, or call it from a worker thread"
        )
    return asyncio.run_coroutine_threadsafe(coro, _bridge_loop(cm)).result()


def _drop_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Frameworks fill optional parameters with None; the handlers want them absent."""
    return {k: v for k, v in kwargs.items() if v is not None}


def sync_handler(cm: CodeMode, tool: NeutralTool) -> Any:
    """A sync function that calls ``tool.handler`` through ``run_sync``."""

    def call(**kwargs: Any) -> str:
        return run_sync(cm, tool.handler(**_drop_none(kwargs)))

    return call


def async_handler(tool: NeutralTool) -> Any:
    """An async function that awaits ``tool.handler`` on the caller's loop."""

    async def call(**kwargs: Any) -> str:
        return await tool.handler(**_drop_none(kwargs))

    return call


_JSON_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


def model_from_schema(name: str, schema: dict[str, Any], pydantic: ModuleType) -> Any:
    """A pydantic model for a flat JSON Schema object (what ``NeutralTool`` parameters are)."""
    required = set(schema.get("required", ()))
    fields: dict[str, Any] = {}
    for key, prop in schema.get("properties", {}).items():
        typ = _JSON_TYPES.get(prop.get("type"), Any)
        constraints = {
            attr: prop[js] for attr, js in (("ge", "minimum"), ("le", "maximum")) if js in prop
        }
        info = pydantic.Field(
            ... if key in required else prop.get("default"),
            description=prop.get("description"),
            **constraints,
        )
        fields[key] = (typ if key in required else typ | None, info)
    return pydantic.create_model(name, **fields)
