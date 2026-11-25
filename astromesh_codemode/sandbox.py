"""Runs one model-written program in a fresh Monty session, with tools as async host functions.

This module is the trust boundary. The program only sees what ``external_lookup``
gives it (one closure per exposed tool); Monty denies filesystem, network and
process access by default because no ``os`` handler or mount is passed. Every
failure the model can cause comes back as short text, never as a host traceback,
and no tool handler outlives the run that started it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from datetime import time as dtime
from typing import Any

import jsonschema
import pydantic_monty as monty

from astromesh_codemode.limits import Limits
from astromesh_codemode.spec import ToolError, ToolSpec
from astromesh_codemode.trace import CallRecord, Trace

ARGS_PREVIEW_CHARS = 200
RESULT_PREVIEW_CHARS = 500
PRINT_CAP_FLOOR = 1024 * 1024
"""Printed output is collected up to max(this, 4 * max_output_bytes); past that the run fails."""
CANCEL_GRACE_S = 1.0
"""How long a run waits for cancelled handlers to unwind before returning anyway."""

_SCALARS = (str, int, float, bool, bytes, type(None), datetime, date, dtime, timedelta)


def safe_repr(value: Any) -> str:
    """``repr`` that never raises (huge ints, broken ``__repr__``)."""
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - model-controlled values must not crash the host
        return f"<unrenderable {type(value).__name__}>"


def safe_str(value: Any) -> str:
    """``str`` that never raises."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - same reason as safe_repr
        return f"<unrenderable {type(value).__name__}>"


def _fallback(value: Any) -> Any:
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=safe_repr)
    return safe_repr(value)


def to_text(value: Any) -> str:
    """Render a value for the model: JSON when possible, Python-ish otherwise. Never raises."""
    if value is None:
        return "None"
    try:
        return json.dumps(value, ensure_ascii=False, default=_fallback)
    except Exception:  # noqa: BLE001 - e.g. ints past the digit limit, cycles, odd keys
        return safe_repr(value)


def _key(k: Any) -> Any:
    return k if isinstance(k, (str, int, float, bool)) else safe_repr(k)


def to_sandbox(value: Any, depth: int = 0) -> Any:
    """Make a handler's return value convertible into the sandbox.

    Containers are copied, dataclasses and pydantic models become dicts, and any
    other object becomes its ``repr`` so the program still gets something readable.
    """
    if isinstance(value, _SCALARS) or depth > 64:
        return value if isinstance(value, _SCALARS) else safe_repr(value)
    if isinstance(value, dict):
        return {_key(k): to_sandbox(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [to_sandbox(v, depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_sandbox(dataclasses.asdict(value), depth + 1)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return to_sandbox(dump(mode="json"), depth + 1)
        except Exception:  # noqa: BLE001 - a broken model_dump still has a repr
            pass
    return safe_repr(value)


def _cut(text: str, max_bytes: int) -> str:
    return text.encode()[: max(0, max_bytes)].decode(errors="ignore")


def compose(main: str, prints: str, cap: int) -> tuple[str, bool]:
    """Join captured stdout and the main line under ``cap`` UTF-8 bytes.

    The main line (result or error) wins: stdout is cut first.
    """
    head = f"stdout:\n{prints.rstrip()}\n\n" if prints else ""
    full = head + main
    if len(full.encode()) <= cap:
        return full, False
    notice = f"\n[truncated: output is capped at {cap} bytes]\n"
    budget = cap - len(notice.encode())
    main_bytes = len(main.encode())
    if head and main_bytes <= budget:
        return _cut(head, budget - main_bytes) + notice + main, True
    return _cut(main, budget) + notice.rstrip("\n"), True


def error_text(exc: monty.MontyError) -> tuple[str, str]:
    """``(type, "<Type> (line N): <message>")``, without the line when Monty gives none."""
    if isinstance(exc, monty.MontyTypingError):
        return "TypeError", f"TypeError: {exc.display()}"
    inner = exc.exception()
    kind = type(inner).__name__ if isinstance(inner, BaseException) else type(exc).__name__
    if isinstance(exc, (monty.MontySyntaxError, monty.MontyRuntimeError)):
        message = exc.display("msg")
        frames = exc.traceback()
    else:
        message, frames = safe_str(exc), []
    line = f" (line {frames[-1].line})" if frames else ""
    return kind, f"{kind}{line}: {message}"


class _Run:
    """State of one run. Never shared between runs, so concurrent runs stay isolated."""

    def __init__(self, tools: dict[str, ToolSpec], validators: dict[str, Any], limits: Limits):
        self.tools = tools
        self.validators = validators
        self.limits = limits
        self.calls: list[CallRecord] = []
        self.count = 0
        self.limit_hit: str | None = None
        self.closed = False
        # handler task -> (tool, args preview, start ms) until its outcome is recorded
        self.pending: dict[asyncio.Future[Any], tuple[str, str, float]] = {}
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def record(
        self,
        tool: str,
        preview: str,
        start: float,
        nbytes: int = 0,
        error: str | None = None,
        error_type: str | None = None,
    ) -> None:
        elapsed = self.ms() - start
        self.calls.append(CallRecord(tool, preview, elapsed, nbytes, error, start, error_type))

    def external_lookup(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        return {py: self._bind(py, spec) for py, spec in self.tools.items()}

    def _bind(self, py: str, spec: ToolSpec) -> Callable[..., Awaitable[Any]]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            if self.closed:
                raise RuntimeError("the run is over")
            start = self.ms()
            preview = _cut(to_text(kwargs), ARGS_PREVIEW_CHARS)

            def reject(exc: Exception) -> Exception:
                kind = type(exc).__name__
                self.record(py, preview, start, 0, f"{kind}: {exc}", kind)
                return exc

            self.count += 1
            if self.count > self.limits.max_calls:
                self.limit_hit = "max_calls"
                raise reject(RuntimeError(max_calls_message(self.limits)))
            if args:
                msg = f"{py}() takes keyword arguments only, e.g. {py}(name=value)"
                raise reject(TypeError(msg))
            problem = _validate(self.validators[py], kwargs)
            if problem:
                raise reject(TypeError(f"{py}(): invalid arguments: {problem}"))

            task: asyncio.Future[Any] | None = None
            try:
                task = asyncio.ensure_future(spec.handler(**kwargs))
                self.pending[task] = (py, preview, start)
                value = await task
                if self.pending.pop(task, None) is None:
                    raise asyncio.CancelledError  # the run ended; already recorded
                value = to_sandbox(value)
                nbytes = len(to_text(value).encode())
            except asyncio.CancelledError:
                if task is not None and self.pending.pop(task, None) is not None:
                    self.record(py, preview, start, 0, "cancelled", "CancelledError")
                raise
            except ToolError as e:
                self.pending.pop(task, None)
                msg = safe_str(e)
                self.record(py, preview, start, 0, f"ToolError: {msg}", "ToolError")
                raise ToolError(msg) from None
            except Exception as e:  # noqa: BLE001 - host detail stays in the trace
                self.pending.pop(task, None)
                kind = type(e).__name__
                self.record(py, preview, start, 0, f"{kind}: {safe_repr(e)}", kind)
                raise RuntimeError(f"{py}() failed with {kind}") from None
            self.record(py, preview, start, nbytes)
            return value

        call.__name__ = py
        return call

    async def cancel_pending(self) -> None:
        """Cancel handlers still running when the run ends and record them as cancelled."""
        self.closed = True
        tasks = list(self.pending)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=CANCEL_GRACE_S)
        for task in tasks:
            info = self.pending.pop(task, None)
            if info is not None:
                self.record(*info, 0, "cancelled", "CancelledError")


def max_calls_message(limits: Limits) -> str:
    return f"max_calls limit reached: at most {limits.max_calls} tool calls per run"


def _validate(validator: Any, kwargs: dict[str, Any]) -> str | None:
    try:
        error = jsonschema.exceptions.best_match(validator.iter_errors(kwargs))
        if error is None:
            return None
        where = "/".join(str(p) for p in error.absolute_path)
        return f"{error.message} (at {where})" if where else error.message
    except Exception:  # noqa: BLE001 - e.g. values too large to describe
        return "arguments could not be validated against the tool schema"


def make_validators(tools: dict[str, ToolSpec]) -> dict[str, Any]:
    out = {}
    for py, spec in tools.items():
        schema = spec.input_schema or {"type": "object"}
        cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
        out[py] = cls(schema)
    return out


async def execute(
    pool: monty.AsyncMonty,
    code: str,
    tools: dict[str, ToolSpec],
    validators: dict[str, Any],
    limits: Limits,
) -> tuple[str, Trace]:
    """Run ``code`` in a fresh session and return the model-facing text plus its trace.

    Always returns text: model-caused failures, limits and unrenderable values all
    become an error line. Handlers still running when the run ends are cancelled.
    """
    started_ns = time.time_ns()
    run = _Run(tools, validators, limits)
    prints = monty.CollectString(max_bytes=max(PRINT_CAP_FLOOR, 4 * limits.max_output_bytes))
    monty_limits: monty.ResourceLimits = {
        "max_feed_duration_secs": limits.timeout_s,
        "max_memory": limits.max_memory,
        # every tool call is a suspension; keep Monty's own cap above ours
        "max_suspensions": max(1000, 4 * limits.max_calls + 100),
    }

    async def feed() -> Any:
        async with pool.checkout(limits=monty_limits) as session:
            return await session.feed_run(
                code, external_lookup=run.external_lookup(), print_callback=prints
            )

    value: Any = None
    error: str | None = None
    error_type: str | None = None
    try:
        value = await asyncio.wait_for(feed(), timeout=limits.timeout_s)
    except TimeoutError:
        # Host-side bound: Monty's clock stops while waiting on tool handlers.
        # Cancelling the checkout discards the worker.
        run.limit_hit = "timeout"
        error_type = "TimeoutError"
        error = f"TimeoutError: run exceeded the time limit of {limits.timeout_s:g}s"
    except monty.MontyError as e:
        inner: BaseException | None = None
        try:
            error_type, error = error_text(e)
            if not isinstance(e, monty.MontyTypingError):
                inner = e.exception()
        except Exception:  # noqa: BLE001 - backstop: run always returns text
            error_type, error = "RuntimeError", "RuntimeError: the run failed"
        if run.limit_hit is None:
            if isinstance(inner, TimeoutError):
                run.limit_hit = "timeout"
            elif isinstance(inner, MemoryError):
                run.limit_hit = "max_memory"
    finally:
        await run.cancel_pending()
    if error is None and run.limit_hit == "max_calls":
        # the program caught the limit error and carried on; the limit still wins
        error_type = "RuntimeError"
        error = f"RuntimeError: {max_calls_message(limits)}"
    try:
        main = error if error is not None else f"result: {to_text(value)}"
        text, truncated = compose(main, _safe_output(prints), limits.max_output_bytes)
    except Exception:  # noqa: BLE001 - backstop: run always returns text
        error_type = "RuntimeError"
        error = text = "RuntimeError: the result could not be rendered"
        truncated = False
    if truncated and run.limit_hit is None:
        run.limit_hit = "max_output_bytes"
    trace = Trace(
        code=code,
        calls=list(run.calls),
        result_preview=text[:RESULT_PREVIEW_CHARS],
        error=error,
        limit_hit=run.limit_hit,
        ms=run.ms(),
        started_ns=started_ns,
        error_type=error_type,
    )
    return text, trace


def _safe_output(prints: monty.CollectString) -> str:
    try:
        return prints.output
    except Exception:  # noqa: BLE001 - lost prints must not hide the result
        return ""
