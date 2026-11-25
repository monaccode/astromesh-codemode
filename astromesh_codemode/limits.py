"""Per-run resource limits for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass

MIN_OUTPUT_BYTES = 256


@dataclass(frozen=True)
class Limits:
    """Bounds applied to every ``run``.

    ``timeout_s`` is a wall-clock bound on the whole run, time spent waiting on
    tool handlers included. ``max_memory`` caps the sandbox heap in bytes.
    ``max_calls`` caps tool calls per run. ``max_output_bytes`` caps the UTF-8
    size of the text returned to the model.
    """

    timeout_s: float = 30
    max_memory: int = 64 * 1024 * 1024
    max_calls: int = 50
    max_output_bytes: int = 16384

    def __post_init__(self) -> None:
        if not self.timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s!r}")
        if self.max_memory <= 0:
            raise ValueError(f"max_memory must be > 0, got {self.max_memory!r}")
        if self.max_calls <= 0:
            raise ValueError(f"max_calls must be > 0, got {self.max_calls!r}")
        if self.max_output_bytes < MIN_OUTPUT_BYTES:
            raise ValueError(
                f"max_output_bytes must be >= {MIN_OUTPUT_BYTES}, got {self.max_output_bytes!r}"
            )
