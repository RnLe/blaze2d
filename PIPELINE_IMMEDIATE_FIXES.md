Eigensolver Health Check

Residual report pipeline_report_square_eps13_r0p3_res24_k6_b8.txt shows every TE/TM k-point stopping at 101 iters with max_residual still 8e2–1.6e4 and max_relative_residual ≈2–3; the zig-zag ratios (~0.44–0.56) mean the loop keeps undoing its own progress. Preconditioner snapshots inside the same report confirm it does cut raw residuals by 1–2 orders (e.g. iteration-trace row 0: preconditioner_avg_before=2.33e3, after=8.25e1), so the stall isn’t caused by the diagonal solve.

Looking directly at operator_iteration_trace_te_k000.csv highlights the real instability: after ~20 iterations of steady decay, max_residual explodes to 4.7e7 at iteration 24, 7.9e7 at iteration 30, 5.7e8 at iteration 34, etc., before dropping back to O(1e2) on the very next step. These spikes coincide with max_relative_scale jumping to 1e7–3e8 and preconditioner_avg_before jumping by 4–5 orders, which can only happen if Rayleigh–Ritz just injected Ritz pairs with |λ| ≫ physical values. The pattern repeats every few steps, giving the ≈0.5 zig-zag ratio reported above and guaranteeing non-convergence.

Code audit: the mass-orthogonalization bug noted in EIGENSOLVER_FIXES.md is fixed (reorthogonalize_with_mass now uses backend.dot(target, mass_basis).conj() at crates/core/src/eigensolver.rs:719, and DeflationWorkspace::project mirrors that at line 458). In the pipeline run you’re inspecting, deflation isn’t even wired (crates/core/src/bandstructure.rs:616-632 hard-code workspace_ref = None), so the culprit must be elsewhere.

The remaining weak spot is the Rayleigh–Ritz stage (eigensolver.rs:1120-1360). After every residual phase you assemble [X|P|W], build mass_proj = XᵀBX, “stabilize” it via jacobi_eigendecomposition, and then demand want = block_size eigenpairs no matter how ill-conditioned the mass matrix becomes. Once W fills up (history factor = 2, so up to ~30 columns) and everything has been symmetry/gamma projected, mass_proj frequently loses rank—the stabilizer keeps any column whose eigenvalue is merely > max·1e-10. When those eigenvalues span >1e12 (which they will after repeated projections), the whitening picks up columns with enormous scaling, the generalized eigen solve dutifully spits out λ≈1e7–1e9, and combine_entries replaces the entire block with those garbage vectors. The very next residual computation therefore starts from scratch (huge norms, Rayleigh quotients miles away), which matches the spikes you see in the traces.

What to try next

Instrument/guard the Rayleigh–Ritz projection

Log prepared.reduced_dim, min_mass_eig, max_mass_eig, and cond = max/min. If cond blows past, say, 1e8 or reduced_dim < want, skip W entirely or fall back to a safer subspace (e.g., [X|P] only).
Reject Ritz vectors whose |λ| is, e.g., > 1e3× the largest eigenvalue seen so far; keep the old x_entries for that slot instead of overwriting them with nonsense.
Consider shrinking history_factor (set [eigensolver.debug] history_multiplier = 0 in the TOML) to verify that W is the trigger—if the residual spikes disappear, you’ve isolated the instability.
Stress-test without optional machinery

Run one config with gamma.enabled = false and another with preconditioner = "homogeneous" (or none) just to prove the clog isn’t tied to the constant-mode projector or structured weights. If those variants behave but the full stack doesn’t, you know exactly which feature is interacting badly.
Temporarily reduce block_size to n_bands so the projected system never exceeds ~20 columns; if convergence returns, it confirms the root cause is the oversized subspace plus ill-conditioned mass matrix.
Add diagnostics to stabilize_projected_system

Right now the filter only drops columns below max_eig·1e-10. Bump that tolerance (e.g., max_eig·1e-6) or explicitly cap the kept columns at the actual numerical rank reported by the Jacobi sweep, and record the selected.len() inside your iteration diagnostics so you can correlate small ranks with the spikes.
Tolerance sanity

Even after you stabilize Rayleigh–Ritz, note that tol = 1e-6 with scale = max(|λ|‖x‖, ‖Θx‖₂) forces B-norm residuals down to ~1e-5 absolute, which is far beyond what MPB itself targets for band diagrams. Once the solver is stable, consider relaxing to 1e-4…1e-3 to keep iteration counts manageable.
If you add those guards/instrumentation and re-run the inspection script, you should see whether the catastrophic λ injections disappear. Once that’s tamed, the preconditioner’s current 10–30× reduction ought to be enough for the block solver to meet any realistic tolerance.