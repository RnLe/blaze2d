# backend

**Status:** Trait definitions (with `Field2D` support) shared by all spectral backends.

## Responsibilities

- `SpectralBuffer` abstracts storage owned by a backend; `Field2D` already implements it.
- `SpectralBackend` defines allocation, FFT hooks, and BLAS-like ops (`scale`, `axpy`, `dot`).

## Usage

```rust
use mpb2d_core::backend::SpectralBackend;
fn touch_backend<B: SpectralBackend>(backend: &B) {
    let grid = mpb2d_core::grid::Grid2D::new(8, 8, 1.0, 1.0);
    let mut field = backend.alloc_field(grid);
    backend.scale(num_complex::Complex64::new(0.0, 1.0), &mut field);
}
```

## Gaps / Next Steps

- Consider adding pointwise multiply helpers once operators need them.
- Provide safe plan caching so FFT performance matches MPB expectations.
