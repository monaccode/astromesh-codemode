"""Every Python snippet in the READMEs and docs runs, in order, per file; the examples compile.

A fenced ```python block runs as written, in one namespace per file, so a later block can use
what an earlier one defined. A first line ``# needs: <module>[, <module>...]`` skips that block
when any of those modules is missing (so a doc can show every framework, and CI's adapters job,
with every extra installed, runs them all). ``# needs: network`` marks a block that
talks to a real server or model: it is never run here (it is only compiled). The snippets
assume one name the surrounding prose explains (``model``, a chat model); it is provided
below as an offline fake.
"""

from __future__ import annotations

import importlib.util
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "README.md",
    "README.es.md",
    "docs/frameworks.md",
    "docs/sandbox.md",
    "docs/mcp-proxy.md",
]
FENCE = re.compile(r"^```(python)\n(.*?)^```", re.S | re.M)
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


def blocks(path: str) -> list[str]:
    return [m.group(2) for m in FENCE.finditer((ROOT / path).read_text("utf-8"))]


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def fake_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return Fake(messages=iter([AIMessage(content="done")]))


@pytest.mark.parametrize("path", FILES)
def test_snippets_run(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    found = blocks(path)
    assert found, f"{path} has no python snippet"
    namespace: dict = {"__name__": "__snippet__"}
    ran, skipped = 0, []
    for i, code in enumerate(found):
        first = code.splitlines()[0]
        needs = (
            [m.strip() for m in first.split(":", 1)[1].split(",") if m.strip()]
            if first.startswith("# needs:")
            else []
        )
        compiled = compile(code, f"{path}[{i}]", "exec")
        if "network" in needs:
            continue
        if not all(has(module) for module in needs):
            skipped.append(f"[{i}] needs {', '.join(needs)}")
            continue
        if "langchain_core" in needs:
            namespace.setdefault("model", fake_model())
        exec(compiled, namespace)
        ran += 1
    if not ran:
        pytest.skip(f"every snippet needs a missing module: {skipped}")


def test_readmes_share_the_same_snippets():
    en, es = blocks("README.md"), blocks("README.es.md")
    assert len(en) == len(es)
    firsts = [[b.splitlines()[0] for b in bs] for bs in (en, es)]
    assert firsts[0] == firsts[1]


def test_readmes_cite_the_measured_results():
    table = (ROOT / "results" / "2026-11-25-L1.md").read_text("utf-8")
    rows = [line for line in table.splitlines() if line.startswith("| 8 ") or "| 512 " in line]
    for readme in ("README.md", "README.es.md"):
        text = (ROOT / readme).read_text("utf-8")
        assert "results/2026-11-25-L1.md" in text
        assert "--level 2 --models gpt-4o-mini,gpt-4o" in text
        for row in rows:
            assert row in text, f"{readme} does not carry the measured row {row!r}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_examples_compile(example):
    py_compile.compile(str(example), doraise=True)


def test_functions_quickstart_runs():
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "functions_quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    assert "codemode_search" in out
    assert "async def list_orders(" in out
    assert "delete_order" not in out.split("result:")[0]
    assert '"customer": "Customer 1"' in out
    assert "PermissionError" in out
