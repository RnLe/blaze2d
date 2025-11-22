# Eigensolver Improvements

Living notes on the Lanczos/LOBPCG plumbing in `mpb2d-core`—what exists today, goals for parity with MPB, and concrete work items.

## Current State (2025-11-22)

- Algorithm: lightweight Lanczos variant inside `solve_lowest_eigenpairs`, no restarting, minimal re-orthogonalization, Jacobi diagonalization of the tridiagonal.
- Backend: single-threaded `CpuBackend` feeding Θ matvecs; FFT path is forced serial because parallel FFT never wins below 1024².
- Benchmarks: Criterion suites cover FFT vs. serial and real hex TM/TE workloads (Γ/M/K) so we can measure true eigensolver cost in ~1 s per solve.
- Limits: Lanczos routinely hits `max_iter` (default 200) on realistic grids, so iterations per k-point are effectively capped and throughput matches across k-path samples.

## High-Leverage Levers

1. **Parallelization**
   - *Goal*: exploit Rayon (CPU) and future GPU backends to batched matvecs and orthogonalization.
   - *Today*: Θ matvecs run serially even when backend could parallelize (FFT path disabled for perf). Iterative loop itself single-threaded.
   - *Ideas*:
     - Re-enable parallel FFT for large grids (>1024²) or offload to GPU backend when available.
     - Parallelize vector ops (dot/axpy) via Rayon to keep Krylov steps multi-core even if FFT stays serial.
     - Batch multiple k-points or multiple Lanczos vectors when building Krylov subspaces (requires API changes).

2. **Preconditioning**
   - *Goal*: reduce iteration count by approximating Θ⁻¹ or diagonal dominance, similar to MPB's preconditioned LOBPCG.
   - *Today*: Fourier-diagonal preconditioner (|k+G|² for TE, |k+G|²/ε_eff for TM plus σ shift) ships as the default, replacing the earlier real-space Jacobi experiment.
   - *Ideas*:
     - Extend the diagonal model with polarization-aware damping or k-dependent scaling (e.g., MPB’s |k+G|⁻² heuristics).
     - Reintroduce a more principled real-space smoother (multigrid-inspired Jacobi on inverse ε) if profiling shows value.
     - Use existing dielectric/grid metadata to adapt σ and ε_eff automatically per k-point.

3. **Conditioning After Eigenpairs**
   - *Goal*: deflate converged eigenpairs to keep remaining spectrum well-conditioned.
   - *Today*: No explicit deflation; solver seeds each run with deterministic vector and relies on Lanczos ordering.
   - *Ideas*:
     - Implement selective reorthogonalization against stored Ritz vectors.
     - Introduce shift-and-invert or spectral transformation once a mode converges.
     - Apply MPB-like "operator improvement" (Θ → (1/ε)Θ(1/ε)) per polarization to tighten conditioning.

## Next Steps

- [ ] Identify quick wins for **parallel vector ops** inside `SpectralBackend` (Rayon-enabled dot/axpy/scale) without reintroducing slow FFTs.
- [ ] Instrument the Fourier-diagonal preconditioner (log iteration deltas, residual drops) so we can tune σ/ε_eff heuristics per workload.
- [ ] Add instrumentation to eigensolver benchmark output (iterations, residuals) so we can see convergence behavior alongside wall-clock time.
- [ ] Survey MPB papers/code for specific deflation/conditioning techniques worth porting (notes + tasks TBD).

> Keep this file concise and action-oriented; update it whenever we land an improvement or reprioritize.
