# Frameworks

`CodeMode` itself is framework-neutral: `cm.tool_definitions()` returns two `NeutralTool`s
(`name`, `description`, `parameters` as JSON Schema, an async `handler`). Each adapter turns
them into the framework's own tool type:

```
astromesh_codemode.adapters.<module>.as_tools(cm) -> list[<native tool>]
```

The list is always `[<name>_search, <name>_run]` (`codemode_search`, `codemode_run` with the
default `name`). Each adapter needs its extra; without it, `as_tools` raises
`MissingExtraError` (an `ImportError`) naming the `pip install` line.

| Framework | Module | Extra | What you get | Calls |
|---|---|---|---|---|
| LangChain | `langchain` | `langchain` | `StructuredTool` | `invoke` (sync), `ainvoke` (async) |
| CrewAI | `crewai` | `crewai` | `BaseTool` | `run` (sync), `arun` (async) |
| Google ADK | `adk` | `adk` | `BaseTool` | `run_async` |
| OpenAI Agents SDK | `openai_agents` | `openai-agents` | `FunctionTool` | `on_invoke_tool` (async) |
| PydanticAI | `pydantic_ai` | `pydantic-ai` | `Tool` | async |
| LlamaIndex | `llamaindex` | `llamaindex` | `FunctionTool` | `call` (sync), `acall` (async) |
| smolagents | `smolagents` | `smolagents` | `Tool` | sync |
| Microsoft Agent Framework | `agent_framework` | `agent-framework` | `FunctionTool` | async |

Optional parameters the framework fills with `None` are dropped before they reach the
handler.

## Sync, async, and closing

`CodeMode` starts a pool of Monty worker processes on the first `run`, and that pool
belongs to the event loop it was created on.

- **Async entry points** (`ainvoke`, `arun`, `acall`, ADK, OpenAI Agents, PydanticAI, Agent
  Framework) await the handler on *your* loop. Close with `await cm.aclose()` (or use
  `async with CodeMode(...) as cm:`).
- **Sync entry points** (LangChain `invoke`, CrewAI `run`, LlamaIndex `call`, smolagents)
  go through **one background loop thread per `CodeMode`**, so the pool is created once and
  reused by every sync call. Close with `astromesh_codemode.adapters.close_sync(cm)`, which
  closes the pool and stops the thread.
- **Pick one style per `CodeMode`.** A pool started by sync calls is bound to the background
  loop, so awaiting `run` from your own loop afterwards fails. Build one `CodeMode` per
  style.
- A sync call made from a thread whose event loop is running is refused with a
  `RuntimeError` (it would block that loop). Use the framework's async API there, or
  `asyncio.to_thread`.
- Do not close while runs are in flight. Both `aclose()` and `close_sync(cm)` are safe to
  call twice, and a later call starts a new pool.

`run_sync(cm, coro)` (also in `astromesh_codemode.adapters`) is the bridge the sync adapters
use, if you write your own sync wrapper.

## Setup used below

```python
import asyncio
import json

from astromesh_codemode import CodeMode
from astromesh_codemode.sources import from_function


def get_rows(n: int) -> list[dict]:
    """Return n rows, each with an id and a value."""
    return [{"id": i, "value": i * 10} for i in range(1, n + 1)]


TOOLS = [from_function(get_rows)]
PROGRAM = "rows = await get_rows(n=100)\nsum(r['value'] for r in rows)"
```

Each example calls the `run` tool directly, the way the framework would when the model
asks for it. In a real agent you only pass the list to the framework.

## LangChain

```python
# needs: langchain_core
from astromesh_codemode.adapters import close_sync
from astromesh_codemode.adapters.langchain import as_tools

cm = CodeMode(TOOLS)
tools = as_tools(cm)  # pass to model.bind_tools(tools) or to an agent constructor
print(tools[1].invoke({"code": PROGRAM}))  # sync
close_sync(cm)


async def lc_async():
    async with CodeMode(TOOLS) as acm:
        print(await as_tools(acm)[1].ainvoke({"code": PROGRAM}))  # async


asyncio.run(lc_async())
```

A full agent loop with a real model: [`examples/langchain_crm.py`](../examples/langchain_crm.py).

## CrewAI

```python
# needs: crewai
from astromesh_codemode.adapters import close_sync
from astromesh_codemode.adapters.crewai import as_tools

cm = CodeMode(TOOLS)
tools = as_tools(cm)  # Agent(role=..., goal=..., backstory=..., tools=tools)
print(tools[1].run(code=PROGRAM))
close_sync(cm)
```

## Google ADK

```python
# needs: google.adk
from astromesh_codemode.adapters.adk import as_tools


async def adk_demo():
    async with CodeMode(TOOLS) as cm:
        tools = as_tools(cm)  # LlmAgent(name=..., model=..., tools=tools)
        print(await tools[1].run_async(args={"code": PROGRAM}, tool_context=None))


asyncio.run(adk_demo())
```

## OpenAI Agents SDK

```python
# needs: agents
from agents.tool_context import ToolContext

from astromesh_codemode.adapters.openai_agents import as_tools


async def oai_demo():
    async with CodeMode(TOOLS) as cm:
        tools = as_tools(cm)  # Agent(name=..., instructions=..., tools=tools)
        raw = json.dumps({"code": PROGRAM})
        ctx = ToolContext(
            context=None, tool_name=tools[1].name, tool_call_id="c1", tool_arguments=raw
        )
        print(await tools[1].on_invoke_tool(ctx, raw))


asyncio.run(oai_demo())
```

## PydanticAI

```python
# needs: pydantic_ai
from astromesh_codemode.adapters.pydantic_ai import as_tools


async def pai_demo():
    async with CodeMode(TOOLS) as cm:
        tools = as_tools(cm)  # Agent(model, tools=tools)
        print(await tools[1].function(code=PROGRAM))


asyncio.run(pai_demo())
```

## LlamaIndex

```python
# needs: llama_index.core
from astromesh_codemode.adapters import close_sync
from astromesh_codemode.adapters.llamaindex import as_tools

cm = CodeMode(TOOLS)
tools = as_tools(cm)  # FunctionAgent(tools=tools, llm=...)
print(tools[1].call(code=PROGRAM).content)  # sync; `await tools[1].acall(...)` is async
close_sync(cm)
```

## smolagents

```python
# needs: smolagents
from astromesh_codemode.adapters import close_sync
from astromesh_codemode.adapters.smolagents import as_tools

cm = CodeMode(TOOLS)
tools = as_tools(cm)  # ToolCallingAgent(tools=tools, model=...)
print(tools[1](code=PROGRAM))
close_sync(cm)
```

smolagents tools are sync only, so they always go through the background loop.

## Microsoft Agent Framework

```python
# needs: agent_framework
from astromesh_codemode.adapters.agent_framework import as_tools


async def af_demo():
    async with CodeMode(TOOLS) as cm:
        tools = as_tools(cm)  # pass as the agent's tools=
        print((await tools[1].invoke(arguments=None, code=PROGRAM))[0].text)


asyncio.run(af_demo())
```

## Your own loop

Without a framework, map each `NeutralTool` to your provider's tool format and await its
handler with the arguments the model sent:

```python
async def own_loop():
    async with CodeMode(TOOLS) as cm:
        defs = {t.name: t for t in cm.tool_definitions()}
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in defs.values()
        ]
        # ... the model answers with a tool call: name + JSON arguments
        call_name, call_args = "codemode_run", json.dumps({"code": PROGRAM})
        print(len(openai_tools), await defs[call_name].handler(**json.loads(call_args)))


asyncio.run(own_loop())
```
