# logging

**Status:** CLI and band-structure runner now emit MPB-style progress logs for long jobs, defaulting to verbose output unless `--quiet` is set.

## What gets logged

- Startup summary: backend type, grid dimensions, polarization, eigensolver knobs, k-point count, and lattice vectors.
- Dielectric/FFT prep: real-space sampling duration plus a note when spectral workspaces/buffers come online.
- Per k-point line: index, `(kx, ky)`, polarization, Lanczos iteration count, number of bands returned, and elapsed wall time.
- Final recap: total time and aggregate iteration count across the sweep.

All messages go to stderr via `eprintln!`, leaving stdout free for CSV/structured output.

## Implementation Notes

- `EigenResult` now tracks the number of Lanczos steps, which feeds the per-k logs.
- Instrumentation lives in `bandstructure::run` and is gated by the shared `Verbosity` enum so any caller (CLI/tests) can opt out.
- Formatting is single-line and stable to make it easy to grep or parse.

## Next Steps

1. Capture additional solver diagnostics (relative residuals) once exposed by the eigensolver.
2. Mirror the same log schema inside upcoming GPU/WASM backends to keep observability uniform.
3. Emit JSON summaries alongside human-readable logs for automated regression harnesses.
