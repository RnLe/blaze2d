# backend-cpu

**Status:** Implements the CPU spectral backend using rustfft with optional rayon-parallel FFT passes.

## Responsibilities

- Provide `CpuBackend`, the `SpectralBackend` implementation referenced by the CLI/tests.
- Owns 2D FFT routines (row/column passes with normalization).
- Supplies BLAS-like helpers (`scale`, `axpy`, `dot`).

## Usage

```rust
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::grid::Grid2D;
let backend = CpuBackend::new();
let mut field = backend.alloc_field(Grid2D::new(8, 8, 1.0, 1.0));
backend.forward_fft_2d(&mut field);
backend.inverse_fft_2d(&mut field);

let parallel_backend = CpuBackend::new_parallel();
let mut field = parallel_backend.alloc_field(Grid2D::new(64, 64, 1.0, 1.0));
parallel_backend.forward_fft_2d(&mut field);
parallel_backend.inverse_fft_2d(&mut field);
```

The parallel variant transposes between the row and column passes so that both stages reuse the same rayon chunked kernels, giving us a straightforward way to compare single-threaded vs. multi-threaded performance in Criterion.

## Gaps / Next Steps

- Cache FFT plans per grid size to avoid re-planning on every call (now more visible in the FFT benchmark).
- Improve heuristics for flipping the `parallel_fft` flag (large grids benefit, tiny grids remain faster when left serial).

## Benchmarks

Run `cargo bench -p mpb2d-backend-cpu fft` to compare serial vs. parallel FFT throughput. Criterion stores results under `target/criterion`, so subsequent runs automatically highlight regressions; you can also pin baselines with `cargo bench -- --save-baseline <name>` and compare via `--baseline <name>`.
