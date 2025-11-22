# dielectric

**Status:** Samples the current geometry on a uniform grid and caches both ε(r) and 1/ε(r).

## Responsibilities

- `Dielectric2D::from_geometry` sweeps grid points (in fractional coordinates) and stores ε(r).
- Automatically caches the reciprocal profile to avoid repeated divisions in TM operators.
- Keeps both sampled arrays and the `Grid2D` metadata for later reuse.

## Usage

```rust
use mpb2d_core::{dielectric::Dielectric2D, geometry::Geometry2D, lattice::Lattice2D, grid::Grid2D};
let geom = Geometry2D::single_air_hole(Lattice2D::square(1.0), 0.2, 12.0);
let grid = Grid2D::new(32, 32, 1.0, 1.0);
let eps = Dielectric2D::from_geometry(&geom, grid);
```

## Gaps / Next Steps

- Allow partial updates (e.g., when only atom radii change) instead of regenerating from scratch.
