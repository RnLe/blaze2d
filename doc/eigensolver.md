# eigensolver

**Status:** Implements a block LOBPCG solver for Θₖ operators (with Rayleigh–Ritz subspace solves, deflation, and symmetry projection) plus the legacy power-iteration helper.

## Responsibilities

- `EigenOptions` captures band counts, iteration limits, tolerances, warm-start controls, and block-related knobs (e.g. `block_size`, preconditioner kind, Γ handling, deflation limits, symmetry constraints). The tolerance is now interpreted as a Rayleigh-relative stop (`‖r‖ / (|λ|‖x‖)` with a `‖Θx‖` fallback) and defaults to `1e-6`, while a fixed `1e-8` absolute guard remains for debugging. `JobConfig → BandStructureJob` conversion now **enforces** the safe defaults: `block_size` is clamped to `n_bands + 2` (never smaller than the requested band count) and the preconditioner is forced to `PreconditionerKind::StructuredDiagonal` unless runs are assembled programmatically. Users should only raise `max_iter` above the default `200` for dedicated stress tests—short, well-conditioned solves converge faster once preconditioning/deflation stay on.
- `EigenResult` reports converged ω values, iteration counts, whether the Γ constant mode was deflated, and the associated `Field2D` modes for downstream analysis.
- `EigenDiagnostics` (carried inside `EigenResult`) records the Rayleigh quotient λ, ω, residual norm, Rayleigh-relative residual, and B-normalization for every returned band plus aggregate stats (max/avg residual, max/avg relative residual, duplicate skips, sorting tolerance) and a full `IterationDiagnostics` history so downstream logs/metrics can replay solver health across iterations.
- `solve_lowest_eigenpairs` seeds a block of search directions, applies Θₖ and B repeatedly, enforces symmetry/deflation constraints, preconditions the residuals, and performs Rayleigh–Ritz on the `[X | P | W]` subspace before deduplicating nearly-equal ω and returning normalized modes.
- Preconditioning is configurable: `PreconditionerKind` now exposes three modes. `None` disables the helper entirely, `HomogeneousJacobi` applies the classic MPB diagonal inverse (ε_eff / |k+G|² with Γ bins zeroed) for a single FFT pair, and `StructuredDiagonal` (the default) multiplies the Jacobi result by dielectric weights (ε for TE, ε⁻¹ for TM) to echo MPB’s ε(r)-aware option without another FFT. TOML configs can pick any of the string aliases (`"homogeneous_jacobi"`, `"structured_diagonal"`, `"none"`; legacy `"fourier_diagonal"` / `"real_space_jacobi"` still select the homogeneous mode) so it is easy to benchmark “light” vs. “full-MPB” behaviour on demand.
- `build_deflation_workspace` and `DeflationWorkspace::project` keep subsequent solves B-orthogonal to previously converged bands (including the analytic Γ mode).
- `WarmStartOptions` controls whether the solver seeds its initial block from previously converged `Field2D` modes, matching MPB’s subspace recycling so consecutive k-point solves converge faster.
- `power_iteration` remains available for quick validation of toy operators.

## Usage

```rust
use mpb2d_core::eigensolver::{
    build_deflation_workspace,
    EigenOptions,
    GammaContext,
    solve_lowest_eigenpairs,
};

let mut opts = EigenOptions::default();
opts.block_size = 4; // request a 4-vector block regardless of n_bands

let bloch = [0.0, 0.0];
let gamma = GammaContext::new(opts.gamma.should_deflate(
    (bloch[0].powi(2) + bloch[1].powi(2)).sqrt(),
));

let mut operator = make_theta_operator();
let deflation = build_deflation_workspace::<_, _>(&mut operator, []);
let result = solve_lowest_eigenpairs(
    &mut operator,
    &opts,
    None,
    gamma,
    None,
    Some(&deflation),
    None,
);
```

## Gaps / Next Steps

- Add explicit stagnation alarms (e.g., WARN when residuals plateau above tolerance for N iterations) now that iteration histories are recorded.
- Finish the Step 10 cleanup story by emitting per-band residual norms and normalization diagnostics so the pipeline’s reporting guarantees are testable.
