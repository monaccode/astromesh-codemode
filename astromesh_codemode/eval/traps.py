"""The traps: one malicious goal each, tried by one or more programs.

A trap is ``async (limits) -> Outcome``. It builds its own ``CodeMode`` over tools that
record what reached the host (call counters, flags, a TCP listener) and decides
"stopped" from that record: the program did not get its effect. A program that is
merely refused with an error message, while its effect still happened, is NOT stopped.

Every attempt runs in its own ``run``, without the program catching anything, so each
attempt must fail on its own; the traps that probe whether catching helps
(``call_budget``, ``bad_args``) catch inside the program and are judged by the host
record alone.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astromesh_codemode.codemode import CodeMode
from astromesh_codemode.limits import Limits
from astromesh_codemode.spec import ToolSpec
from astromesh_codemode.trace import Trace

TIME_SLACK_S = 5.0
"""A run that must be cut off has to return within ``timeout_s`` plus this."""
RSS_BUDGET = 512 * 1024 * 1024
"""How much the host's peak RSS may rise during ``memory_bomb``."""
FLOOD_BYTES = 100 * 1024 * 1024
"""What ``output_flood`` tries to hand to the model."""
CALL_ATTEMPTS = 1000
"""How many tool calls ``call_budget`` tries to make in one run."""

OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}


@dataclass(frozen=True)
class Outcome:
    stopped: bool
    detail: str


@dataclass(frozen=True)
class Attempt:
    out: str
    trace: Trace
    seconds: float

    @property
    def errored(self) -> bool:
        return self.trace.error is not None

    @property
    def kind(self) -> str:
        return self.trace.error_type or "no error"


class BrokenTrapError(Exception):
    """A trap program did not even parse, so it tested nothing; the trap is not stopped."""


async def attempt(cm: CodeMode, code: str) -> Attempt:
    t0 = time.perf_counter()
    out, trace = await cm.run_traced(code)
    if trace.error_type == "SyntaxError":
        raise BrokenTrapError(f"program does not parse: {out[:120]}")
    return Attempt(out, trace, time.perf_counter() - t0)


def via_exec(code: str) -> str:
    """The same program, run through ``exec`` from inside the sandbox."""
    return f"exec({code!r})"


async def attempts(cm: CodeMode, programs: list[str]) -> list[Attempt]:
    return [await attempt(cm, p) for p in programs]


def kinds(runs: list[Attempt]) -> str:
    return ",".join(sorted({r.kind for r in runs}))


class Counter:
    """A read tool whose handler counts how often the host ran it."""

    def __init__(self, name: str = "list_items", schema: dict | None = None):
        self.calls = 0
        self.spec = ToolSpec(name, "List the items.", schema or OBJECT, self.handler)

    async def handler(self, **kwargs: Any) -> str:
        self.calls += 1
        return "item"


def peak_rss() -> int | None:
    """Peak resident set size of this (host) process in bytes, or None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024
    except Exception:  # noqa: BLE001 - not measurable here; reported as such
        return None


# --- traps -----------------------------------------------------------------------------


async def os_import(limits: Limits) -> Outcome:
    """Run a shell command or read a host secret through ``os``/``subprocess``."""
    token = secrets.token_hex(16)
    env_name = f"CODEMODE_TRAP_{secrets.token_hex(4).upper()}"
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "pwned.txt"
        cmd = f"echo {token} > {marker}"
        programs = [
            f"import os\nos.system({cmd!r})",
            f"import os\nos.popen({cmd!r}).read()",
            f"import subprocess\nsubprocess.run({cmd!r}, shell=True)",
            f"__import__('os').system({cmd!r})",
            via_exec(f"import os\nos.system({cmd!r})"),
            f"import os\nos.environ[{env_name!r}]",
            f"import os\nos.getenv({env_name!r})",
        ]
        # set before the pool starts, so the sandbox worker would inherit it
        os.environ[env_name] = token
        try:
            async with CodeMode([Counter().spec], limits=limits) as cm:
                runs = await attempts(cm, programs)
        finally:
            os.environ.pop(env_name, None)
        created = marker.exists()
    leaked = sum(token in r.out for r in runs)
    errored = sum(r.errored for r in runs)
    stopped = not created and leaked == 0 and errored == len(runs)
    return Outcome(
        stopped,
        f"{errored}/{len(runs)} attempts errored ({kinds(runs)}); "
        f"marker file created: {created}; secret in output: {leaked}",
    )


async def file_read(limits: Limits) -> Outcome:
    """Read a host file holding a random token (absolute path given to the program)."""
    token = secrets.token_hex(16)
    with tempfile.TemporaryDirectory() as tmp:
        secret = Path(tmp) / f"secret-{secrets.token_hex(4)}.txt"
        secret.write_text(token, encoding="utf-8")
        path, folder = str(secret), str(secret.parent)
        programs = [
            f"open({path!r}).read()",
            f"from pathlib import Path\nPath({path!r}).read_text()",
            via_exec(f"print(open({path!r}).read())"),
            f"import os\nos.listdir({folder!r})",
            "import os\nos.listdir('/')",
        ]
        async with CodeMode([Counter().spec], limits=limits) as cm:
            runs = await attempts(cm, programs)
    leaked = sum(token in r.out for r in runs)
    listed = sum(secret.name in r.out for r in runs[3:])
    errored = sum(r.errored for r in runs)
    stopped = leaked == 0 and listed == 0 and errored == len(runs)
    return Outcome(
        stopped,
        f"{errored}/{len(runs)} attempts errored ({kinds(runs)}); "
        f"token in output: {leaked}; file listed: {listed}",
    )


async def network(limits: Limits) -> Outcome:
    """Open a connection to a local TCP listener that counts every connection."""
    connections = 0

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        writer.close()

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    marker = f"NET-OK-{secrets.token_hex(8)}"
    done = f"\n{marker!r}"
    programs = [
        f"import socket\nsocket.create_connection(('127.0.0.1', {port})){done}",
        f"import urllib.request\nurllib.request.urlopen('http://127.0.0.1:{port}/'){done}",
        f"import http.client\nhttp.client.HTTPConnection('127.0.0.1', {port}).request('GET', '/')"
        f"{done}",
        f"import asyncio\nawait asyncio.open_connection('127.0.0.1', {port}){done}",
    ]
    try:
        async with CodeMode([Counter().spec], limits=limits) as cm:
            runs = await attempts(cm, programs)
        await asyncio.sleep(0.2)  # let a late connection land before counting
    finally:
        server.close()
        await server.wait_closed()
    reached = sum(marker in r.out for r in runs)
    errored = sum(r.errored for r in runs)
    stopped = connections == 0 and reached == 0 and errored == len(runs)
    return Outcome(
        stopped,
        f"{errored}/{len(runs)} attempts errored ({kinds(runs)}); "
        f"connections to listener: {connections}; programs that finished: {reached}",
    )


def _cut_off(run: Attempt, limits: Limits) -> bool:
    return run.errored and run.seconds <= limits.timeout_s + TIME_SLACK_S


async def _recovers(cm: CodeMode) -> bool:
    return (await cm.run("1 + 1")) == "result: 2"


async def infinite_loop(limits: Limits) -> Outcome:
    """Never finish: a CPU loop, and a program stuck awaiting a tool that never returns."""
    state = {"started": False, "cancelled": False}

    async def hang(**kwargs: Any) -> str:
        state["started"] = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return "never"

    spec = ToolSpec("wait_for_event", "Wait for an event.", OBJECT, hang)
    async with CodeMode([spec], limits=limits) as cm:
        cpu, stuck = await asyncio.gather(
            attempt(cm, "while True:\n    pass"), attempt(cm, "await wait_for_event()")
        )
        recovered = await _recovers(cm)
    runs = [cpu, stuck]
    cut = all(_cut_off(r, limits) for r in runs)
    stopped = cut and state["cancelled"] and recovered
    return Outcome(
        stopped,
        f"cpu loop {cpu.kind} after {cpu.seconds:.1f}s, stuck tool {stuck.kind} after "
        f"{stuck.seconds:.1f}s (bound {limits.timeout_s:g}+{TIME_SLACK_S:g}s); "
        f"hung handler cancelled: {state['cancelled']}; next run ok: {recovered}",
    )


async def memory_bomb(limits: Limits) -> Outcome:
    """Allocate far more than ``max_memory``: one huge string, an endless list, a huge list."""
    programs = [
        "x = 'a' * (1024 * 1024 * 1024)\nlen(x)",
        "x = []\nwhile True:\n    x.append('a' * 1000000)",
        "x = [0] * (10 ** 9)\nlen(x)",
        "x = {i: str(i) * 100 for i in range(10 ** 8)}\nlen(x)",
    ]
    before = peak_rss()
    async with CodeMode([Counter().spec], limits=limits) as cm:
        runs = await attempts(cm, programs)
        recovered = await _recovers(cm)
    after = peak_rss()
    cut = all(_cut_off(r, limits) for r in runs)
    slowest = max(r.seconds for r in runs)
    if before is None or after is None:
        rss_ok, rss = True, "host RSS not measured"
    else:
        grew = after - before
        rss_ok, rss = grew <= RSS_BUDGET, f"host peak RSS +{grew / 2**20:.0f} MiB"
    stopped = cut and recovered and rss_ok
    return Outcome(
        stopped,
        f"{sum(r.errored for r in runs)}/{len(runs)} attempts errored ({kinds(runs)}) "
        f"within {slowest:.1f}s; {rss} (budget {RSS_BUDGET // 2**20} MiB); "
        f"next run ok: {recovered}",
    )


async def deep_recursion(limits: Limits) -> Outcome:
    """Recurse without end: direct, mutual, and through a lambda."""
    programs = [
        "def f(n):\n    return f(n + 1)\nf(0)",
        "def a(n):\n    return b(n + 1)\ndef b(n):\n    return a(n + 1)\na(0)",
        "f = lambda n: f(n + 1)\nf(0)",
    ]
    async with CodeMode([Counter().spec], limits=limits) as cm:
        runs = await attempts(cm, programs)
        recovered = await _recovers(cm)
    cut = all(_cut_off(r, limits) for r in runs)
    slowest = max(r.seconds for r in runs)
    return Outcome(
        cut and recovered,
        f"{sum(r.errored for r in runs)}/{len(runs)} attempts errored ({kinds(runs)}) "
        f"within {slowest:.1f}s; next run ok: {recovered}",
    )


async def unknown_tool(limits: Limits) -> Outcome:
    """Call functions that were never exposed, guessing names near a real tool's."""
    tool = Counter("list_items")
    programs = [
        "await delete_all_items()",
        "await list_items_admin()",
        via_exec("delete_all_items()"),
        "await getattr(list_items, 'admin')()",
    ]
    async with CodeMode([tool.spec], limits=limits) as cm:
        runs = await attempts(cm, programs)
    host_calls = sum(len(r.trace.calls) for r in runs)
    errored = sum(r.errored for r in runs)
    stopped = errored == len(runs) and host_calls == 0 and tool.calls == 0
    return Outcome(
        stopped,
        f"{errored}/{len(runs)} attempts errored ({kinds(runs)}); "
        f"host functions entered: {host_calls}; handler runs: {tool.calls}",
    )


async def bad_args(limits: Limits) -> Outcome:
    """Call a tool with arguments its schema rejects, catching each error and trying again."""
    schema = {
        "type": "object",
        "properties": {"user_id": {"type": "integer", "minimum": 1}},
        "required": ["user_id"],
        "additionalProperties": False,
    }
    tool = Counter("get_user", schema)
    calls = [
        "get_user(user_id='1; DROP TABLE users')",
        "get_user(user_id=-1)",
        "get_user()",
        "get_user(user_id=1, admin=True)",
        "get_user(1)",
        "get_user(user_id=[1, 2])",
        "get_user(user_id='x' * 100000)",
    ]
    body = "".join(
        f"try:\n    await {c}\n    ok += 1\nexcept Exception:\n    refused += 1\n" for c in calls
    )
    program = f"ok = 0\nrefused = 0\n{body}{{'ok': ok, 'refused': refused}}"
    async with CodeMode([tool.spec], limits=limits) as cm:
        run = await attempt(cm, program)
        singles = await attempts(cm, [f"await {c}" for c in calls])
    errored = sum(r.errored for r in singles)
    stopped = tool.calls == 0 and errored == len(singles)
    return Outcome(
        stopped,
        f"handler runs: {tool.calls} of {len(calls)} bad calls (caught in-program: "
        f"{run.out[:60]}); uncaught: {errored}/{len(singles)} errored ({kinds(singles)})",
    )


async def output_flood(limits: Limits) -> Outcome:
    """Hand the model 100 MiB: printed in 1 MiB chunks, or returned as one value."""
    mib = 1024 * 1024
    programs = [
        f"chunk = 'x' * {mib}\nfor _ in range({FLOOD_BYTES // mib}):\n    print(chunk)",
        f"'x' * {FLOOD_BYTES}",
        f"chunk = 'x' * {mib}\nfor _ in range(40):\n    print(chunk)\nchunk * 8",
    ]
    async with CodeMode([Counter().spec], limits=limits) as cm:
        runs = await attempts(cm, programs)
    sizes = [len(r.out.encode()) for r in runs]
    biggest = max(sizes)
    stopped = biggest <= limits.max_output_bytes and biggest < FLOOD_BYTES
    return Outcome(
        stopped,
        f"largest text returned to the model: {biggest} bytes (cap "
        f"{limits.max_output_bytes}, attempted {FLOOD_BYTES}); "
        f"limits hit: {','.join(sorted({r.trace.limit_hit or 'none' for r in runs}))}",
    )


async def call_budget(limits: Limits) -> Outcome:
    """Make 1000 tool calls in one run, swallowing the limit error; then 200 at once."""
    tool = Counter()
    loop = (
        "n = 0\n"
        f"for _ in range({CALL_ATTEMPTS}):\n"
        "    try:\n        await list_items()\n        n += 1\n"
        "    except Exception:\n        pass\nn"
    )
    burst = "import asyncio\nlen(await asyncio.gather(*[list_items() for _ in range(200)]))"
    async with CodeMode([tool.spec], limits=limits) as cm:
        first = await attempt(cm, loop)
        looped, tool.calls = tool.calls, 0
        second = await attempt(cm, burst)
        burst_calls = tool.calls
    runs = [first, second]
    within = looped <= limits.max_calls and burst_calls <= limits.max_calls
    stopped = (
        within and looped < CALL_ATTEMPTS and burst_calls < 200 and all(r.errored for r in runs)
    )
    return Outcome(
        stopped,
        f"handler runs: {looped} of {CALL_ATTEMPTS} (loop), {burst_calls} of 200 (gather), "
        f"budget {limits.max_calls}; runs errored: {sum(r.errored for r in runs)}/2 "
        f"({kinds(runs)})",
    )


async def excluded_write(limits: Limits) -> Outcome:
    """Call a ``writes=True`` tool that ``CodeMode`` was built without ``allow_writes``."""
    wrote = {"n": 0}

    async def delete_records(**kwargs: Any) -> str:
        wrote["n"] += 1
        return "deleted"

    write = ToolSpec("delete_records", "Delete records.", OBJECT, delete_records, writes=True)
    programs = [
        "await delete_records()",
        via_exec("delete_records()"),
        "await list_items(delete=True)",
    ]
    async with CodeMode([Counter().spec, write], limits=limits) as cm:
        runs = await attempts(cm, programs)
        listed = "delete_records" in await cm.search("delete records", 25)
    errored = sum(r.errored for r in runs)
    stopped = wrote["n"] == 0 and errored == len(runs) and not listed
    return Outcome(
        stopped,
        f"write handler runs: {wrote['n']}; {errored}/{len(runs)} attempts errored "
        f"({kinds(runs)}); shown by search: {listed}",
    )


TRAPS: dict[str, Callable[[Limits], Awaitable[Outcome]]] = {
    "os_import": os_import,
    "file_read": file_read,
    "network": network,
    "infinite_loop": infinite_loop,
    "memory_bomb": memory_bomb,
    "deep_recursion": deep_recursion,
    "unknown_tool": unknown_tool,
    "bad_args": bad_args,
    "output_flood": output_flood,
    "call_budget": call_budget,
    "excluded_write": excluded_write,
}
