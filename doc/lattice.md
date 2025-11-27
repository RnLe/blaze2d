# lattice

**Status:** Provides lattice constructors plus reciprocal and coordinate transforms.

## Responsibilities

- Create common lattices via helpers (`square`, `rectangular`, `hexagonal`, `oblique`).
- Produce `ReciprocalLattice2D` values and expose primitive vectors.
- Convert between fractional and Cartesian coordinates.
- Estimate a characteristic length (currently |a1|) for scaling circle radii.

## Usage

```rust
use mpb2d_core::lattice::Lattice2D;
let lat = Lattice2D::hexagonal(1.0);
let cart = lat.fractional_to_cartesian([0.25, 0.25]);
```

## Gaps / Next Steps

- Consider exposing full metric tensors for more accurate distance calculations.
- Support validation helpers (e.g., ensuring primitive vectors are normalized as expected).
