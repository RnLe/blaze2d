# operator

**Status:** Implements both TM and TE flavors of Θₖ plus the toy Laplacian operator for validation/regression tests.

## Responsibilities

- `ThetaOperator` stores references to the chosen backend, dielectric snapshot, polarization, and Bloch vector, and implements:
	- TM matvec −∇·(1/ε∇) using the usual ∇/ε/∇ pipeline.
	- TE matvec −(1/ε)(∇ + ik)² by multiplying spectral components by |k+G|² and scaling in real space by 1/ε.
- `LinearOperator` trait abstracts matvec semantics so eigensolvers can target either Θₖ or the toy Laplacian interchangeably.
- `ToyLaplacian` applies a periodic −Δ via FFTs (used for validation / power iteration).

## Usage

```rust
use mpb2d_core::operator::ThetaOperator;
use mpb2d_core::operator::ToyLaplacian;
// Constructed with a SpectralBackend + Grid2D; see backend-cpu tests for an example.
```

## Gaps / Next Steps

- Generalize toy operator helpers so they work with future backends (CUDA/WebGPU).
- Consider exposing precomputed |k+G|² factors to avoid per-apply recomputation in TE mode once profiling says it matters.
