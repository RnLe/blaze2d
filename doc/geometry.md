# geometry

**Status:** Implements circular basis atoms evaluated in fractional coordinates.

## Responsibilities

- `BasisAtom` stores fractional positions, relative radii, and material contrast.
- `Geometry2D` aggregates atoms, background permittivity, and lattice references.
- Provides `relative_permittivity_at_fractional` / `relative_permittivity_at_cartesian` for sampling.
- Offers convenience constructors (`air_holes_in_dielectric`, `single_air_hole`).

## Usage

```rust
use mpb2d_core::geometry::{Geometry2D, BasisAtom};
use mpb2d_core::lattice::Lattice2D;
let geom = Geometry2D::single_air_hole(Lattice2D::square(1.0), 0.2, 12.0);
let eps = geom.relative_permittivity_at_fractional([0.05, 0.05]);
```

## Gaps / Next Steps

- Extend to mixed-material atoms (priority ordering / blending rules).
- Support non-circular inclusions or anisotropic ε tensors.
