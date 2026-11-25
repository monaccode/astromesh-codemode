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
- Eval de costo nivel 1 (`python -m astromesh_codemode.eval cost --level 1`, `run_level1`):
  modelo guionado sobre tools reales, *direct* (todo el catálogo en cada llamada) contra
  Code Mode (`search` + `run` reales), tokens con `tiktoken` `o200k_base`; 6 escenarios
  (catálogo 8/64/512 × 5/500 filas), escribe `results/<fecha>-L1.{json,md}` y corre en CI.
  Extra `eval` (`tiktoken`, `litellm`). Code Mode pierde en salida en todos los escenarios
  y en contexto con 4 tools y 5 filas; ver `docs/eval.md`.
