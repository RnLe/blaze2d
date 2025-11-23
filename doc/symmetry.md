# symmetry

**Status:** Provides square/hex presets plus a densifier for high-symmetry paths, and mirror-parity projectors that can be opt-in (automatic lattice-driven inference is available but disabled by default while we debug the solver).

## Responsibilities

- `PathType` enumerates currently supported presets (square, hexagonal, custom).
- `standard_path` emits a densified path (segments-per-leg) along Γ-centered routes.

## Usage

```rust
use mpb2d_core::symmetry::{self, PathType};
let lattice = mpb2d_core::lattice::Lattice2D::hexagonal(1.0);
let path = symmetry::standard_path(&lattice, PathType::Hexagonal, 12);
```

### Parity projectors

`SymmetryOptions` describes a list of mirror constraints, each given by a `ReflectionConstraint { axis, parity }`. When you pass the options through `EigenOptions.symmetry`, the solver projects every search and residual vector with

```text
P_± = ½ (I ± S)
```

where `S` reflects about the requested axis (x or y). This keeps the Lanczos subspace inside the requested irreducible representation while still composing cleanly with Γ-deflation and the general B-orthogonal projector.

Example (TOML) – enforce odd symmetry across x:

```toml
[eigensolver.symmetry]
reflections = [
  { axis = "x", parity = "odd" },
]
```

Rust-side construction:

```rust
use mpb2d_core::symmetry::{Parity, ReflectionAxis, ReflectionConstraint, SymmetryOptions, SymmetryProjector};

let options = SymmetryOptions {
  reflections: vec![ReflectionConstraint {
    axis: ReflectionAxis::X,
    parity: Parity::Odd,
  }],
  ..Default::default()
};
let projector = SymmetryProjector::from_options(&options).unwrap();
```

### Auto-selecting mirror axes from the lattice

`SymmetryOptions::default()` starts with **no** automatic reflections so fresh jobs explore the full eigenspace (matching the current debugging posture). If you want MPB-style parity projections, explicitly supply the `auto` block:

```toml
[eigensolver.symmetry]
auto = { parity = "even" }
```

- Square / rectangular lattices ⇒ mirrors across `x` and `y`.
- Triangular lattices ⇒ mirrors across `x` and `y` (aligned with the default basis).
- Oblique lattices ⇒ no mirrors (options stay empty).

Manual `reflections` always take precedence; `auto` only fires when the list is empty. At solve time, auto reflections are **gated per Bloch point** via `SymmetryOptions::selection_for_bloch`: a mirror is enabled only when the corresponding Bloch component is within `auto.bloch_tolerance` (default `1e-6`) relative to the orthogonal component. Off-axis `k` points therefore skip parity projectors automatically, and the pipeline records both the applied and skipped counts in `MetricsEvent::KPointSolve` (`symmetry_reflections` / `symmetry_reflections_skipped`).

To disable the auto inference without editing the TOML, pass `--no-auto-symmetry` to the CLI (still useful when a config explicitly enables `auto`).

## Gaps / Next Steps

- Encode canonical labels (Γ, X, M, …) alongside coordinates for plotting and CSV headers.
- Extend symmetry support beyond axis mirrors (e.g., rotations, glide planes) and automatically infer valid constraints from the lattice/geometry.
