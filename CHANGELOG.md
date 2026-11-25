# Changelog

## [Unreleased]

### Added

- Scaffold del paquete con `ToolSpec` y `ToolError`.
- `CodeMode`: las tools se exponen como `codemode_search` (stubs Python tipados) y
  `codemode_run` (programa del modelo en un sandbox Monty), con `Limits` (tiempo de reloj,
  memoria, llamadas, bytes de salida), `Trace` por run y `emit_otel` (extra `otel`).
