# operator

**Status:** Implements both TM and TE flavors of Θₖ plus the toy Laplacian operator for validation/regression tests.

## Responsibilities

- `ThetaOperator` stores references to the chosen backend, dielectric snapshot, polarization, and Bloch vector, and implements:
  - TM matvec −∇·(1/ε∇) using the usual ∇/ε/∇ pipeline.
  - TE matvec −(∇ + ik)² by multiplying spectral components by |k+G|²; the TE dielectric only appears through the B/mass application.
  - Mass-operator helpers so the eigensolver can evaluate Bx alongside Θₖx (identity for TM, ε scaling for TE).
  - Preconditioner builder for the Fourier-diagonal variant (|k+G|² for TE, |k+G|²/ε_eff for TM) used by LOBPCG.
  - Cached `|k+G|²` spectra plus Bloch-shifted `kx_shifted/ky_shifted` axes so TE matvecs, TM gradient/divergence kernels, and Fourier-diagonal preconditioners reuse the same data every apply.
- `LinearOperator` trait abstracts matvec semantics so eigensolvers can target either Θₖ or the toy Laplacian interchangeably.
- `ToyLaplacian` applies a periodic −Δ via FFTs (used for validation / power iteration).

## Usage

```rust
use mpb2d_core::operator::ThetaOperator;
use mpb2d_core::operator::ToyLaplacian;
// Constructed with a SpectralBackend + Grid2D; see backend-cpu tests for an example.
```

## Validation

- `_tests_operator.rs` uses the slow DFT `TestBackend` to compare Θₖ against analytic expectations.
- Bloch-shifted plane waves on rectangular grids assert the |k+G|² eigenvalues (TE) and |k+G|²/ε scaling (TM).
- Hermitian symmetry is exercised in patterned dielectrics by verifying ⟨x, Θₖy⟩ = ⟨Θₖx, y⟩ for TM/TE fields.
- Mass-operator tests ensure TE multiplies by ε(x,y) exactly while TM remains the identity.
- Fourier-diagonal preconditioner tests confirm the TE scaling factor 1/(|k+G|² + σ) and the TM scaling 1/(|k+G|²/ε_eff + σ).

## Gaps / Next Steps

- Generalize toy operator helpers so they work with future backends (CUDA/WebGPU).
- Evaluate whether additional derived spectra (e.g., cached reciprocals or |k+G|⁻²) would further reduce TM preconditioner cost once profiling says it matters.
- Plumb the cached spectra through backend-neutral APIs so future GPU implementations can upload them once per k-point instead of rebuilding on device.
