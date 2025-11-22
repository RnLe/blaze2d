# field

**Status:** Provides the `Field2D` container for complex data on uniform grids.

## Responsibilities

- Store `Grid2D` metadata alongside contiguous `Vec<Complex64>` samples.
- Offer constructors (`zeros`, `from_vec`) plus indexing helpers.
- Expose slice views for interop with FFT backends or custom kernels.
- Implements `SpectralBuffer`, so backends can use it directly as their storage type.

## Usage

```rust
use mpb2d_core::{field::Field2D, grid::Grid2D};
let grid = Grid2D::new(16, 16, 1.0, 1.0);
let mut field = Field2D::zeros(grid);
field.fill(num_complex::Complex64::new(1.0, 0.0));
```

## Gaps / Next Steps

- Add real/imag view helpers or typed iterators when needed by solvers.
- Consider borrowing slices without copying for backend hand-offs.
