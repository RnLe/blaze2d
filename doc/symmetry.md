# symmetry

**Status:** Provides square/hex presets plus a densifier for high-symmetry paths.

## Responsibilities

- `PathType` enumerates currently supported presets (square, hexagonal, custom).
- `standard_path` emits a densified path (segments-per-leg) along Γ-centered routes.

## Usage

```rust
use mpb2d_core::symmetry::{self, PathType};
let lattice = mpb2d_core::lattice::Lattice2D::hexagonal(1.0);
let path = symmetry::standard_path(&lattice, PathType::Hexagonal, 12);
```

## Gaps / Next Steps

- Encode canonical labels (Γ, X, M, …) alongside coordinates for plotting and CSV headers.
- Implement irreducible Brillouin zone reductions and point-group projections.
