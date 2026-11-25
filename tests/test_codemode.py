import pytest

from astromesh_codemode import CodeMode, Limits, ToolSpec, Trace
from astromesh_codemode.trace import CallRecord, emit_otel
from tests.fixtures import make_tools


async def test_search_returns_stubs():
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        out = await cm.search("rows")
        assert "async def get_rows(" in out
        assert "Return n rows, each with an id and a value." in out


async def test_search_without_matches():
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        out = await cm.search("zzzqqq")
        assert "No tools" in out


async def test_tool_definitions():
    specs, _ = make_tools()
    cm = CodeMode(specs)
    search, run = cm.tool_definitions()
    assert search.name == "codemode_search"
    assert run.name == "codemode_run"
    assert search.parameters["required"] == ["query"]
    assert run.parameters["required"] == ["code"]
    assert "search" in run.description.lower()
    assert "codemode_search" in run.description
    assert "await" in run.description
    assert "keyword" in run.description
    assert "last expression" in run.description
    assert "before" in search.description.lower()
    assert search.handler == cm.search
    assert run.handler == cm.run


async def test_custom_name():
    specs, _ = make_tools()
    search, run = CodeMode(specs, name="crm").tool_definitions()
    assert (search.name, run.name) == ("crm_search", "crm_run")
    assert "crm_search" in run.description


async def test_handlers_work_through_definitions():
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        search, run = cm.tool_definitions()
        assert "get_rows" in await search.handler(query="rows")
        assert await run.handler(code="1 + 2") == "result: 3"


async def test_writes_excluded():
    specs, calls = make_tools()
    async with CodeMode(specs) as cm:
        assert "save" not in await cm.search("persist save value")
        assert "save" not in await cm.search("")
        out = await cm.run("await save(x=1)")
        assert out.startswith("NameError")
        assert calls["save"] == 0
    async with CodeMode(specs, allow_writes=True) as cm:
        assert "async def save(" in await cm.search("persist save value")
        out = await cm.run("await save(x=1)")
        assert '"saved": 1' in out
        assert calls["save"] == 1


def test_name_collision_raises():
    specs, _ = make_tools()
    clash = ToolSpec("get-rows", "dup", {"type": "object"}, specs[0].handler)
    with pytest.raises(ValueError, match="get-rows"):
        CodeMode([*specs, clash])


def test_limits_defaults_and_validation():
    lim = Limits()
    assert (lim.timeout_s, lim.max_memory, lim.max_calls, lim.max_output_bytes) == (
        30,
        64 * 1024 * 1024,
        50,
        16384,
    )
    with pytest.raises(ValueError, match="max_calls"):
        Limits(max_calls=0)
    with pytest.raises(ValueError, match="timeout_s"):
        Limits(timeout_s=-1)


def test_trace_to_dict():
    t = Trace(
        code="1",
        calls=[CallRecord("get_rows", "{}", 1.5, 10, None)],
        result_preview="result: 1",
        error=None,
        limit_hit=None,
        ms=3.0,
    )
    d = t.to_dict()
    assert d["calls"][0]["tool"] == "get_rows"
    assert d["ms"] == 3.0
    assert d["limit_hit"] is None


async def test_last_trace_set_by_run():
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        assert cm.last_trace is None
        await cm.run("1")
        assert cm.last_trace.code == "1"
        assert cm.last_trace.result_preview == "result: 1"


async def test_emit_otel():
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        await cm.run("a = await get_rows(n=2)\nb = await stamp()\nlen(a)")
        emit_otel(cm.last_trace, tracer_provider=provider)

    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "codemode.run")
    children = [s for s in spans if s.parent is not None]
    assert sorted(s.name for s in children) == ["get_rows", "stamp"]
    assert all(s.parent.span_id == root.context.span_id for s in children)
    assert root.attributes["codemode.calls"] == 2


def test_emit_otel_without_sdk_is_noop():
    t = Trace("1", [], "result: 1", None, None, 1.0)
    emit_otel(t)  # global no-op provider: must not raise


def _otel_provider():
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


SENSITIVE_CODE = (
    "try:\n    await boom()\nexcept Exception:\n    pass\nawait get_rows(n=31337)\nawait crash()"
)


def _span_text(span) -> str:
    parts = [str(v) for v in span.attributes.values()]
    parts.append(str(span.status.description))
    return " ".join(parts)


async def test_emit_otel_hides_sensitive_data_by_default():
    provider, exporter = _otel_provider()
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        await cm.run(SENSITIVE_CODE)
        emit_otel(cm.last_trace, tracer_provider=provider)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert "codemode.args" not in span.attributes
        text = _span_text(span)
        for secret in ("31337", "crm caído", "secret", "postgres", "failed with"):
            assert secret not in text, (span.name, secret)
    by_name = {s.name: s for s in spans}
    assert by_name["boom"].attributes["codemode.error_type"] == "ToolError"
    assert by_name["boom"].status.description == "ToolError"
    assert by_name["crash"].attributes["codemode.error_type"] == "KeyError"
    assert by_name["codemode.run"].attributes["codemode.error_type"] == "RuntimeError"
    assert by_name["codemode.run"].status.description == "RuntimeError"
    assert by_name["get_rows"].attributes["codemode.bytes"] > 0


async def test_emit_otel_include_args():
    provider, exporter = _otel_provider()
    specs, _ = make_tools()
    async with CodeMode(specs) as cm:
        await cm.run(SENSITIVE_CODE)
        emit_otel(cm.last_trace, tracer_provider=provider, include_args=True)
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert "31337" in by_name["get_rows"].attributes["codemode.args"]
    assert "crm caído" in by_name["boom"].attributes["codemode.error"]
    assert "crash() failed" in by_name["codemode.run"].attributes["codemode.error"]
