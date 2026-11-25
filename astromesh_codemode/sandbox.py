"""Runs one model-written program in a fresh Monty session, with tools as async host functions.

This module is the trust boundary. The program only sees what ``external_lookup``
gives it (one closure per exposed tool); Monty denies filesystem, network and
process access by default because no ``os`` handler or mount is passed. Every
failure the model can cause comes back as short text, never as a host traceback.
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

_SCALARS = (str, int, float, bool, bytes, type(None), datetime, date, dtime, timedelta)


def _fallback(value: Any) -> Any:
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


def to_text(value: Any) -> str:
    """Render a value for the model: JSON when possible, Python-ish otherwise."""
    if value is None:
        return "None"
    try:
        return json.dumps(value, ensure_ascii=False, default=_fallback)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def to_sandbox(value: Any, depth: int = 0) -> Any:
    """Make a handler's return value convertible into the sandbox.

    Containers are copied, dataclasses and pydantic models become dicts, and any
    other object becomes its ``repr`` so the program still gets something readable.
    """
    if isinstance(value, _SCALARS) or depth > 64:
        return value if isinstance(value, _SCALARS) else repr(value)
    if isinstance(value, dict):
        return {
            (k if isinstance(k, (str, int, float, bool)) else repr(k)): to_sandbox(v, depth + 1)
            for k, v in value.items()
        }
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
    return repr(value)


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


def error_text(exc: monty.MontyError) -> str:
    """``"<Type> (line N): <message>"``, without the line when Monty gives none."""
    if isinstance(exc, monty.MontyTypingError):
        return f"TypeError: {exc.display()}"
    inner = exc.exception()
    kind = type(inner).__name__ if isinstance(inner, BaseException) else type(exc).__name__
    if isinstance(exc, (monty.MontySyntaxError, monty.MontyRuntimeError)):
        message = exc.display("msg")
        frames = exc.traceback()
    else:
        message, frames = str(exc), []
    line = f" (line {frames[-1].line})" if frames else ""
    return f"{kind}{line}: {message}"


class _Run:
    """State of one run. Never shared between runs, so concurrent runs stay isolated."""

    def __init__(self, tools: dict[str, ToolSpec], validators: dict[str, Any], limits: Limits):
        self.tools = tools
        self.validators = validators
        self.limits = limits
        self.calls: list[CallRecord] = []
        self.count = 0
        self.limit_hit: str | None = None
        self.t0 = time.perf_counter()

    def _ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def external_lookup(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        return {py: self._bind(py, spec) for py, spec in self.tools.items()}

    def _bind(self, py: str, spec: ToolSpec) -> Callable[..., Awaitable[Any]]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            start = self._ms()
            preview = _cut(to_text(kwargs), ARGS_PREVIEW_CHARS)

            def record(nbytes: int, error: str | None) -> None:
                ms = self._ms() - start
                self.calls.append(CallRecord(py, preview, ms, nbytes, error, start))

            self.count += 1
            if self.count > self.limits.max_calls:
                self.limit_hit = "max_calls"
                msg = max_calls_message(self.limits)
                record(0, msg)
                raise RuntimeError(msg)
            if args:
                msg = f"{py}() takes keyword arguments only, e.g. {py}(name=value)"
                record(0, msg)
                raise TypeError(msg)
            problem = _validate(self.validators[py], kwargs)
            if problem:
                msg = f"{py}(): invalid arguments: {problem}"
                record(0, msg)
                raise TypeError(msg)
            try:
                value = await spec.handler(**kwargs)
            except ToolError as e:
                record(0, f"ToolError: {e}")
                raise ToolError(str(e)) from None
            except Exception as e:  # noqa: BLE001 - host detail stays in the trace
                record(0, f"{type(e).__name__}: {e!r}")
                raise RuntimeError(f"{py}() failed with {type(e).__name__}") from None
            value = to_sandbox(value)
            record(len(to_text(value).encode()), None)
            return value

        call.__name__ = py
        return call


def max_calls_message(limits: Limits) -> str:
    return f"max_calls limit reached: at most {limits.max_calls} tool calls per run"


def _validate(validator: Any, kwargs: dict[str, Any]) -> str | None:
    error = jsonschema.exceptions.best_match(validator.iter_errors(kwargs))
    if error is None:
        return None
    where = "/".join(str(p) for p in error.absolute_path)
    return f"{error.message} (at {where})" if where else error.message


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
    """Run ``code`` in a fresh session and return the model-facing text plus its trace."""
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

    error: str | None = None
    try:
        value = await asyncio.wait_for(feed(), timeout=limits.timeout_s)
        main = f"result: {to_text(value)}"
    except TimeoutError:
        # Host-side bound: Monty's clock stops while waiting on tool handlers.
        # Cancelling the checkout discards the worker, so nothing keeps running.
        run.limit_hit = "timeout"
        error = f"TimeoutError: run exceeded the time limit of {limits.timeout_s:g}s"
    except monty.MontyError as e:
        error = error_text(e)
        inner = e.exception() if not isinstance(e, monty.MontyTypingError) else None
        if run.limit_hit is None:
            if isinstance(inner, TimeoutError):
                run.limit_hit = "timeout"
            elif isinstance(inner, MemoryError):
                run.limit_hit = "max_memory"
    if error is None and run.limit_hit == "max_calls":
        # the program caught the limit error and carried on; the limit still wins
        error = f"RuntimeError: {max_calls_message(limits)}"
    if error is not None:
        main = error
    text, truncated = compose(main, _safe_output(prints), limits.max_output_bytes)
    if truncated and run.limit_hit is None:
        run.limit_hit = "max_output_bytes"
    trace = Trace(
        code=code,
        calls=list(run.calls),
        result_preview=text[:RESULT_PREVIEW_CHARS],
        error=error,
        limit_hit=run.limit_hit,
        ms=run._ms(),
        started_ns=started_ns,
    )
    return text, trace


def _safe_output(prints: monty.CollectString) -> str:
    try:
        return prints.output
    except Exception:  # noqa: BLE001 - lost prints must not hide the result
        return ""
