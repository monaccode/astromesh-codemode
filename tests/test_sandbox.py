import asyncio

import pytest

from astromesh_codemode import CodeMode, Limits
from tests.fixtures import STAMP, make_tools


@pytest.fixture
async def setup():
    specs, calls = make_tools()
    async with CodeMode(specs) as cm:
        yield cm, calls


async def test_run_filters_large_data(setup):
    cm, _ = setup
    out = await cm.run("rows = await get_rows(n=500)\n[r for r in rows if r['id'] in (7, 42)]")
    assert '"id": 7' in out
    assert '"id": 42' in out
    assert '"id": 3' not in out
    assert len(cm.last_trace.calls) == 1
    assert cm.last_trace.calls[0].tool == "get_rows"
    assert cm.last_trace.calls[0].error is None
    assert cm.last_trace.calls[0].bytes > 0


async def test_tool_error_is_catchable(setup):
    cm, _ = setup
    out = await cm.run("try:\n    await boom()\nexcept Exception as e:\n    msg = str(e)\nmsg")
    assert "crm caído" in out
    assert cm.last_trace.error is None
    assert cm.last_trace.calls[0].error is not None


async def test_uncaught_tool_error_has_type_and_line(setup):
    cm, _ = setup
    out = await cm.run("x = 1\nawait boom()")
    assert out.startswith("Exception (line 2): crm caído")
    assert "Traceback" not in out
    assert cm.last_trace.error == out


async def test_unexpected_host_error_is_not_leaked(setup):
    cm, calls = setup
    out = await cm.run("await crash()")
    assert calls["crash"] == 1
    assert "secret" not in out
    assert "crash" in out
    assert "KeyError" in out
    assert "secret" in cm.last_trace.calls[0].error  # the host keeps the detail


async def test_args_validated_against_schema(setup):
    cm, calls = setup
    out = await cm.run('await get_rows(n="x")')
    assert out.startswith("TypeError")
    assert "get_rows" in out
    assert calls["get_rows"] == 0


async def test_positional_args_rejected(setup):
    cm, calls = setup
    out = await cm.run("await get_rows(5)")
    assert out.startswith("TypeError")
    assert "keyword" in out
    assert calls["get_rows"] == 0


async def test_max_calls():
    specs, calls = make_tools()
    async with CodeMode(specs, limits=Limits(max_calls=3)) as cm:
        out = await cm.run("for i in range(5):\n    await get_rows(n=1)")
        assert "max_calls" in out
        assert cm.last_trace.limit_hit == "max_calls"
        assert calls["get_rows"] == 3


async def test_max_calls_cannot_be_swallowed():
    specs, calls = make_tools()
    async with CodeMode(specs, limits=Limits(max_calls=3)) as cm:
        code = (
            "n = 0\nfor i in range(5):\n    try:\n        await get_rows(n=1)\n"
            "        n += 1\n    except Exception:\n        pass\nn"
        )
        out = await cm.run(code)
        assert "max_calls" in out
        assert cm.last_trace.limit_hit == "max_calls"
        assert calls["get_rows"] == 3


async def test_output_truncated(setup):
    cm, _ = setup
    out = await cm.run('"x" * 50_000')
    assert len(out.encode()) <= Limits().max_output_bytes
    assert "[truncated" in out
    assert cm.last_trace.limit_hit == "max_output_bytes"


async def test_huge_prints_keep_the_result(setup):
    cm, _ = setup
    out = await cm.run('for i in range(5000):\n    print("line", i)\n"done"')
    assert len(out.encode()) <= Limits().max_output_bytes
    assert "[truncated" in out
    assert out.endswith('result: "done"')


async def test_timeout():
    specs, _ = make_tools()
    async with CodeMode(specs, limits=Limits(timeout_s=1)) as cm:
        out = await cm.run("while True:\n    pass")
        assert "TimeoutError" in out
        assert cm.last_trace.limit_hit == "timeout"


async def test_timeout_while_waiting_on_host():
    async def slow() -> int:
        await asyncio.sleep(30)
        return 1

    from astromesh_codemode import ToolSpec

    spec = ToolSpec("slow", "Sleeps.", {"type": "object", "properties": {}}, slow)
    async with CodeMode([spec], limits=Limits(timeout_s=1)) as cm:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await cm.run("await slow()")
        assert loop.time() - t0 < 5
        assert "TimeoutError" in out
        assert cm.last_trace.limit_hit == "timeout"
        # the pool still serves runs after a cancelled session
        assert await cm.run("1 + 1") == "result: 2"


async def test_memory_limit():
    specs, _ = make_tools()
    async with CodeMode(specs, limits=Limits(max_memory=1024 * 1024)) as cm:
        out = await cm.run("x = 'a' * (10 ** 8)\nlen(x)")
        assert out.startswith("MemoryError")
        assert cm.last_trace.limit_hit == "max_memory"


async def test_non_json_values(setup):
    cm, _ = setup
    out = await cm.run("return await stamp()")
    assert STAMP.isoformat() in out


async def test_unconvertible_tool_value_is_readable(setup):
    cm, _ = setup
    out = await cm.run("await opaque()")
    assert "Opaque(42)" in out
    assert cm.last_trace.error is None


async def test_print_only(setup):
    cm, _ = setup
    out = await cm.run('print("hola")')
    assert "hola" in out
    assert "result: None" in out


async def test_syntax_error_has_line(setup):
    cm, _ = setup
    out = await cm.run("x = 1\ndef f(:\n    pass")
    assert out.startswith("SyntaxError (line 2):")


async def test_sandbox_has_no_host_access(setup):
    cm, _ = setup
    out = await cm.run("import os\nos.listdir('/')")
    assert out.startswith("PermissionError")
    out = await cm.run("open('/etc/passwd').read()")
    assert out.startswith("PermissionError")


async def test_concurrent_runs_isolated(setup):
    cm, _ = setup
    two = "a = await get_rows(n=1)\nb = await get_rows(n=1)\n'two'"
    three = "for i in range(3):\n    await get_rows(n=1)\n'three'"
    (out2, t2), (out3, t3) = await asyncio.gather(cm.run_traced(two), cm.run_traced(three))
    assert out2 == 'result: "two"'
    assert out3 == 'result: "three"'
    assert len(t2.calls) == 2
    assert len(t3.calls) == 3


async def test_concurrent_max_calls_are_per_run():
    specs, _ = make_tools()
    async with CodeMode(specs, limits=Limits(max_calls=3)) as cm:
        code = "for i in range(3):\n    await get_rows(n=1)\n'ok'"
        results = await asyncio.gather(*(cm.run_traced(code) for _ in range(3)))
        assert all(out == 'result: "ok"' for out, _ in results)
        assert all(t.limit_hit is None for _, t in results)


async def test_no_state_between_runs(setup):
    cm, _ = setup
    assert await cm.run("x = 1\nx") == "result: 1"
    out = await cm.run("x")
    assert out.startswith("NameError")
