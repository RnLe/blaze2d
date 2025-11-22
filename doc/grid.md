# grid

**Status:** Lightweight wrapper for uniform Nx×Ny sampling domains.

## Responsibilities

- Stores dimensions and physical spans (`lx`, `ly`).
- Supplies helper methods like `idx(ix, iy)` and `len()` for flattened indexing.
- Provides serde integration so grids can be configured via TOML.

## Usage

```rust
use mpb2d_core::grid::Grid2D;
let grid = Grid2D::new(64, 64, 1.0, 1.0);
let center_idx = grid.idx(32, 32);
```

## Gaps / Next Steps

- Surface spacing (`dx`, `dy`) helpers to avoid recomputing in downstream modules.
- Consider enforcing that `lx`/`ly` default to lattice magnitudes rather than 1.0.
