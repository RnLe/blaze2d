# bandstructure

**Status:** Provides high-level job/result containers and a working runner that drives Θₖ + Lanczos over a supplied k-path.

## Responsibilities

- `BandStructureJob` bundles geometry, grid, polarization, eigensolver options, and k-path data for a sweep.
- `BandStructureResult` stores sampled bands per k-point along with cumulative k-distances.
- `Verbosity` toggles the human-readable stderr logs so callers can select `Quiet` versus `Verbose` output.
- `run` clones a backend per k-point, builds Θₖ, runs the eigensolver, logs progress (startup + per-k, honoring `Verbosity`) and returns raw ω values.

## Usage

```rust
use mpb2d_core::bandstructure::{self, BandStructureJob, Verbosity};
let job: BandStructureJob = /* build or deserialize */;
let result = bandstructure::run(CpuBackend::new(), &job, Verbosity::Verbose);
```

## Gaps / Next Steps

- Provide helpers for common k-path parametrizations and CSV/export utilities beyond the CLI.
- Consider caching FFT plans/backend state across k-points for GPU backends to reduce setup costs.
- Surface richer summary statistics (band gaps, min/max ω) so callers do not have to post-process results.
