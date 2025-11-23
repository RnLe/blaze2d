# cli

**Status:** Provides the `mpb2d-lite run` command that reads TOML configs, drives the CPU backend, and emits MPB-style CSV output.

## Responsibilities

- Parse `JobConfig` files (geometry/grid/polarization/k-path/eigensolver).
- Accept `--path {square|hexagonal}` + `--segments-per-leg` overrides to avoid hand-writing k-paths.
- Expose `--no-auto-symmetry` to disable lattice-inferred parity projectors without editing the TOML config.
- Invoke `bandstructure::run` with the CPU backend for now, wiring through the requested verbosity.
- Produce CSV rows with `k_index`, `(kx, ky)`, cumulative path length, and normalized (`ω / 2π`) band frequencies.
- Emit stderr logs describing config loading, preset overrides, and CSV destinations before delegating to the core runner (which adds per-k progress lines) unless `--quiet` is supplied.

## Usage

```shell
mpb2d-lite run --config examples/square_air_hole.toml --path square --segments-per-leg 12 --no-auto-symmetry --output bands.csv --quiet
```

Omit `--output` to stream CSV to stdout.

## Gaps / Next Steps

- Allow selecting alternate backends (CUDA, WASM) via CLI flags.
- Support named paths with explicit Γ/X/M labels in output.
- Add optional JSON summary stats (band gaps, min/max frequencies) alongside the CSV stream.
