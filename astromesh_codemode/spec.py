"""Tool description shared by every source and adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """A tool failed; the message is safe to show to the model."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Awaitable[Any]]
    output_schema: dict | None = None
    writes: bool = False
