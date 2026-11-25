# Changelog

## [Unreleased]

### Added

- Scaffold del paquete con `ToolSpec` y `ToolError`.
- `CodeMode`: las tools se exponen como `codemode_search` (stubs Python tipados) y
  `codemode_run` (programa del modelo en un sandbox Monty), con `Limits` (tiempo de reloj,
  memoria, llamadas, bytes de salida), `Trace` por run y `emit_otel` (extra `otel`).
- Eval de trampas (`python -m astromesh_codemode.eval traps`, `run_traps(limits)`): 11
  programas maliciosos (OS, archivos, red, loop, memoria, recursión, tool inexistente,
  args inválidos, flood de salida, presupuesto de llamadas, tool de escritura) contra
  `CodeMode`; "detenida" se prueba con evidencia del host. 11/11 con `Limits()` y corre en CI.
