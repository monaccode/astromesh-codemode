# The sandbox

`codemode_run` executes a Python program the model wrote. It runs in
[Monty](https://github.com/pydantic/monty) (`pydantic-monty`, pinned to `>=1.7,<1.8`), a
Python interpreter written in Rust that runs in separate worker processes. This page says
what a program can do, what it gets back, and where the limits are. The security tests
against it are in [`eval.md`](eval.md).

## What a program sees

- **The exposed tools, as `async` functions, and nothing else from the host.** Each tool is
  bound under its Python name (non-identifier characters become `_`, so an MCP tool
  `fs_read-file` is `fs_read_file`). Hidden write tools are not defined at all: calling one
  is a `NameError`.
- **Keyword arguments only.** `await get_customer(customer_id=3)`. A positional call is a
  `TypeError`. Arguments are validated against the tool's JSON Schema **before** the handler
  runs; a call that does not match is a `TypeError` naming the problem, and the handler
  never runs.
- **No filesystem, no network, no processes, no environment variables.** Monty gets no
  `os` handler and no mounts, so `open()` fails with `PermissionError`, `socket` and
  `subprocess` do not exist, and `os.environ` / `os.getenv` give nothing from the host.
- **A subset of Python.** Monty implements much of the language (functions, comprehensions,
  `async`/`await`, `try`/`except`, classes) and a few standard modules such as `asyncio`,
  `json`, `re`, `math` and `datetime`. Nothing is pre-imported: a program that uses
  `asyncio.gather` must `import asyncio` first. Third-party packages cannot be imported.
- **A fresh session per run.** No variable survives from one run to the next, and
  concurrent runs never share state, counters or limits.

## What comes back to the model

Always text, never a host traceback:

- The value of the program is its **last expression** (a top-level `return` also works),
  rendered as JSON when possible: `result: {"total": 3}`.
- Anything `print()`ed comes first, under `stdout:`.
- A failure is one line, `<ErrorType> (line N): <message>`, for example
  `NameError (line 1): name 'delete_order' is not defined`.
- A tool that raises `ToolError("...")` surfaces as that `ToolError` with your message, and
  the program can catch it. Any other exception in a handler surfaces as
  `RuntimeError: <tool>() failed with <ExceptionType>`: the message, which may carry host
  detail, stays in the trace.

```python
import asyncio

from astromesh_codemode import CodeMode, Limits, ToolError
from astromesh_codemode.sources import from_function


async def get_rows(n: int) -> list[dict]:
    """Return n rows, each with an id and a value."""
    return [{"id": i, "value": i * 10} for i in range(1, n + 1)]


async def flaky() -> str:
    """A tool that fails with a message the model may see."""
    raise ToolError("the CRM is down, try later")


PROGRAM = """
import asyncio
a, b = await asyncio.gather(get_rows(n=3), get_rows(n=2))
print("fetched", len(a) + len(b))
try:
    await flaky()
except Exception as e:
    print("flaky:", e)
{"total": sum(r["value"] for r in a + b)}
"""


async def main():
    async with CodeMode([from_function(get_rows), from_function(flaky)]) as cm:
        print(await cm.run(PROGRAM))
        print(await cm.run("await get_rows(3)"))  # positional: TypeError
        print(await cm.run("await get_rows(n='three')"))  # schema: TypeError, handler not run


asyncio.run(main())
```

## Limits

`Limits` is frozen and applies to every run of a `CodeMode`:

| Field | Default | Bounds |
|---|---|---|
| `timeout_s` | `30` | Wall clock for the whole run, **including time spent waiting on tool handlers** (Monty's own clock stops while it waits on the host, so the host enforces this too). A handler still running at the deadline is cancelled. |
| `max_memory` | `64 * 1024 * 1024` | The sandbox heap, in bytes. Past it the run fails with `MemoryError`. |
| `max_calls` | `50` | Tool calls per run. The call past the budget raises; if the program catches that and carries on, the run still ends as an error. |
| `max_output_bytes` | `16384` | UTF-8 size of the text returned to the model (minimum 256). Longer output is cut, stdout first, and carries `[truncated: output is capped at <cap> bytes]`. Printed output is collected up to max(1 MiB, 4 × this); past that the run fails with `MemoryError`. |

The `run` tool's description tells the model its call budget, timeout and output cap, and
to filter and aggregate in code.

```python
tight = Limits(timeout_s=5, max_calls=3, max_output_bytes=512)


async def limits_demo():
    async with CodeMode([from_function(get_rows)], limits=tight) as cm:
        text, trace = await cm.run_traced("for i in range(10):\n    await get_rows(n=1)")
        print(text, trace.limit_hit)  # max_calls
        text, trace = await cm.run_traced("'x' * 10000")
        print(len(text.encode()), trace.limit_hit)  # <= 512, max_output_bytes


asyncio.run(limits_demo())
```

## Writes

`CodeMode` v0.1 is read-only by default. A `ToolSpec` with `writes=True`
(`from_function(fn, writes=True)`) is not shown by `search` and not defined in the sandbox
unless the `CodeMode` is built with `allow_writes=True`. Name collisions are checked across
all tools, hidden ones included. There is no per-call approval step yet: with
`allow_writes=True`, a program can call a write tool as many times as `max_calls` allows.

## Traces

Every run produces a `Trace`: `code`, `calls` (one `CallRecord` per tool call: `tool`,
`args_preview`, `ms`, `bytes`, `error`, `error_type`, `start_ms`), `result_preview`,
`error`, `error_type`, `limit_hit` (`timeout`, `max_memory`, `max_calls`,
`max_output_bytes` or `None`) and `ms`. `Trace.to_dict()` gives it as a dict.

- `await cm.run_traced(code)` returns `(text, trace)` for that run.
- `await cm.run(code)` returns the text and sets `cm.last_trace`. With concurrent runs the
  **last one to finish wins**, so under concurrency use `run_traced`.
- `CallRecord.error` is host-side detail and may include a raw exception `repr`;
  `error_type` is only the type name.

`emit_otel(trace, tracer_provider=None, include_args=False)` (from
`astromesh_codemode.trace`, extra `otel`) emits a `codemode.run` span with one child span
per tool call. By default it exports tool names, durations, byte counts, the limit hit and
error *types*; argument values and error messages only with `include_args=True`. Without
`opentelemetry-api` installed it does nothing.

```python
from astromesh_codemode.trace import emit_otel


async def trace_demo():
    async with CodeMode([from_function(get_rows)]) as cm:
        text, trace = await cm.run_traced("len(await get_rows(n=5))")
        print(text, trace.to_dict()["calls"][0]["tool"])
        emit_otel(trace)  # no-op without opentelemetry-api


asyncio.run(trace_demo())
```

## Lifecycle

- The Monty worker pool starts lazily on the first `run` and is stopped by
  `await cm.aclose()` (or leaving `async with`). From sync framework code, use
  `astromesh_codemode.adapters.close_sync(cm)`. See [`frameworks.md`](frameworks.md).
- The pool is bound to the event loop it was created on. Do not use one `CodeMode` from
  several event loops.
- Do not close while runs are in flight. Closing twice is fine, and a later `run` starts a
  new pool.
