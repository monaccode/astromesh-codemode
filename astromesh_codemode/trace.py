"""What happened inside one ``run``: tool calls, result, limits; optional OpenTelemetry export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallRecord:
    """One tool call made by the sandboxed program."""

    tool: str
    args_preview: str
    ms: float
    bytes: int
    error: str | None
    start_ms: float = 0.0
    """Offset of the call from the start of the run, in milliseconds."""


@dataclass(frozen=True)
class Trace:
    """A finished run. ``error`` is the text shown to the model when the run failed."""

    code: str
    calls: list[CallRecord]
    result_preview: str
    error: str | None
    limit_hit: str | None
    ms: float
    started_ns: int = field(default=0, compare=False)
    """Wall-clock start of the run (``time.time_ns()``), used to place OpenTelemetry spans."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def emit_otel(trace: Trace, *, tracer_provider: Any | None = None) -> None:
    """Emit a ``codemode.run`` span with one child span per tool call.

    Needs the ``otel`` extra (``opentelemetry-api``); without it this is a no-op.
    ``tracer_provider`` defaults to the globally configured one.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return
    tracer = otel_trace.get_tracer("astromesh_codemode", tracer_provider=tracer_provider)
    start = trace.started_ns or None
    end = start + int(trace.ms * 1e6) if start else None
    root = tracer.start_span("codemode.run", start_time=start)
    root.set_attribute("codemode.calls", len(trace.calls))
    root.set_attribute("codemode.ms", trace.ms)
    if trace.limit_hit:
        root.set_attribute("codemode.limit_hit", trace.limit_hit)
    if trace.error:
        root.set_attribute("codemode.error", trace.error)
        root.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, trace.error))
    ctx = otel_trace.set_span_in_context(root)
    for call in trace.calls:
        c_start = start + int(call.start_ms * 1e6) if start else None
        span = tracer.start_span(call.tool, context=ctx, start_time=c_start)
        span.set_attribute("codemode.tool", call.tool)
        span.set_attribute("codemode.args", call.args_preview)
        span.set_attribute("codemode.bytes", call.bytes)
        span.set_attribute("codemode.ms", call.ms)
        if call.error:
            span.set_attribute("codemode.error", call.error)
            span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, call.error))
        span.end(end_time=c_start + int(call.ms * 1e6) if c_start else None)
    root.end(end_time=end)
