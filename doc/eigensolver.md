# eigensolver

**Status:** Provides a lightweight Lanczos/Krylov routine for Θₖ plus the shared power-iteration helper.

## Responsibilities

- `EigenOptions` captures solver settings (band count, iteration cap, tolerance) plus Γ handling knobs (enable/disable + tolerance).
- `EigenResult` stores computed eigenfrequencies, the number of Lanczos steps executed, and whether the Γ constant mode was deflated.
- `solve_lowest_eigenpairs` performs a Lanczos sweep, diagonalizes the tridiagonal proxy, and reports the lowest bands while optionally projecting out the Γ constant mode.
- `power_iteration` offers a simple validation utility on top of any `LinearOperator` implementation.

## Usage

```rust
use mpb2d_core::eigensolver::{EigenOptions, GammaContext, solve_lowest_eigenpairs};

let opts = EigenOptions::default();
let bloch = [0.0, 0.0];
let gamma = GammaContext::new(opts.gamma.should_deflate((bloch[0].powi(2) + bloch[1].powi(2)).sqrt()));
let result = solve_lowest_eigenpairs(&mut operator, &opts, None, gamma);
```

## Gaps / Next Steps

- Add block-LOBPCG support to converge larger band bundles faster and to emit mode shapes.
- Return mode shapes when requested to enable field post-processing.
