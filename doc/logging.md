# logging

**Status:** CLI and band-structure runner now emit MPB-style progress logs for long jobs, defaulting to verbose output unless `--quiet` is set.

## What gets logged

- Startup summary: backend type, grid dimensions, polarization, eigensolver knobs, k-point count, and lattice vectors.
- Dielectric/FFT prep: real-space sampling duration plus a note when spectral workspaces/buffers come online.
- Per k-point line: index, `(kx, ky)`, polarization, Lanczos iteration count, number of bands returned, elapsed wall time, and a compact frequency range (`frequencies=[min..max]`) so the console stays readable even when solving dozens of bands.
- Solver state changes such as Γ deflation, deflation store usage, or warm-start counts are captured via metrics rather than inline console tags to keep the human-readable output minimal.
- When metrics are enabled, every k-point still emits `k_point_solve`, and now each solver iteration also triggers an `eigen_iteration` event carrying `(max_residual, avg_residual, block_size, new_directions)` for downstream dashboards. Those events (plus the `EigenDiagnostics` payload) retain the detailed Rayleigh/residual/mass data that was removed from the human log.
- Final recap: total time and aggregate iteration count across the sweep.

All messages go to stderr via `eprintln!`, leaving stdout free for CSV/structured output.

## Implementation Notes

- `EigenResult` now attaches an `EigenDiagnostics` payload (per-band ω/λ/residual/mass norms plus aggregate/iteration stats) that the runner exposes through metrics while keeping the console summary short.
- Instrumentation lives in `bandstructure::run` and is gated by the shared `Verbosity` enum so any caller (CLI/tests) can opt out.
- Formatting is single-line and stable to make it easy to grep or parse.

## Next Steps

1. Add explicit WARN lines or CLI exit codes when iteration residuals stagnate above tolerance for configurable windows.
2. Mirror the same log schema inside upcoming GPU/WASM backends to keep observability uniform.
3. Emit JSON summaries alongside human-readable logs for automated regression harnesses.
