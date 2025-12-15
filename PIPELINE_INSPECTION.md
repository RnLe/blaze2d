# Pipeline Inspection Playbook

The solver pipeline touches far more than the eigensolver loop. To shorten the
"black box" chain we can enumerate each stage, what data it produces, and which
artifacts/metrics make the most sense to capture when debugging tough runs.

## 1. Geometry → Dielectric Sampling

- **Inputs**: `Geometry2D`, lattice vectors, atom list, background ε.
- **Processing**: walk the uniform grid, sample ε(x,y), build `Dielectric2D` with
  both ε and ε⁻¹ slices.
- **What to inspect**:
  - Real-space ε(x,y) heatmap or contour plot to confirm geometry alignment.
  - Histogram/statistics (min/max) to catch accidental negative/zero values.
  - Fourier spectrum ε(G) to ensure periodicity and symmetry look right.
- **Artifacts now available**: enable `--dump-pipeline <dir>` (or set
  `inspection.dump_eps_*` in the TOML) to emit `epsilon_real.csv` and
  `epsilon_fourier.csv` for every run.
  - `epsilon_real.csv`: columns `ix,iy,x,y,epsilon`.
  - `epsilon_fourier.csv`: columns `ix,iy,gx,gy,real,imag` (raw FFT, no extra
    normalization).

## 2. FFT Workspace Prep (per k-point)

- **Inputs**: dielectric slices, Bloch wavevector, backend plans.
- **Processing**: instantiate `ThetaOperator`, allocate spectral buffers, cache
  |k+G|² and parity/deflation helpers.
- **What to inspect**:
  - Time-to-build per k-point (already logged via `FftWorkspace`).
  - Optional dump of |k+G|² arrays or the actual k+G lists for first k-point to
    confirm indexing—CSV with `ix,iy,kx_plus_G,ky_plus_G` would help diagnose
    dispersion mismatches.
  - Memory footprint of cached buffers (esp. when GPU backend lands).
- **Artifacts now available** (enable via `[inspection] dump_fft_workspace_raw = true`
  / `dump_fft_workspace_report = true` or just pass `--dump-pipeline`):
  - `fft_workspace_raw_k000.csv` — raw per-cell snapshot of
    `(ix,iy,kx_plus_G,ky_plus_G,|k+G|²)` for the first Bloch point.
  - `fft_workspace_report_k000.json` — compact summary with grid size,
    polarization, Bloch norm, min/max/mean |k+G|², shifted-k extents, and the
    estimated workspace buffer footprint.

## 3. Preconditioner Snapshot

- **Inputs**: Θ symbol, ε statistics, polarization.
- **Processing**: build Fourier diagonal (|k+G|² / ε_eff + σ).
- **What to inspect**:
  - Dump the diagonal used for the first k-point (CSV or histogram) so we can
    see whether σ dominates or ε_eff drifts.
  - Already instrumented via `preconditioner_*` metrics; pairing that with an
    actual diagonal dump makes it easier to explain stalled reductions.

## 4. Warm-Start Cache & Symmetry Projector

- **Inputs**: prior modes, requested symmetry reflections.
- **Processing**: sort by ω, trim, project.
- **What to inspect**:
  - Current metrics (`seed_count`, `warm_start_hits`, symmetry skip counts)
    explain how many modes survive; to debug *why* a mode is rejected we could
    dump the B-norm of each seed before/after projection or write parquet traces
    for a single k-point on demand.

## 5. Eigensolver Loop

- Already heavily instrumented (per-iteration residuals, preconditioner stats,
  symmetry/deflation bookkeeping). Remaining knobs:
  - Expose min/max Rayleigh quotient per iteration for spotting eigenvalue
    drift.
  - Emit random-projection sanity checks ( ⟨xᵢ,εxᵢ⟩ ) to catch norm loss.

## Additional Artifact Ideas

1. **Grid sanity dump**: CSV of fractional vs. Cartesian coordinates to confirm
   `Grid2D` spacing.
2. **Operator spot checks**: record Θ·plane-wave responses for a small set of
   (mx,my) indices and compare to analytical eigenvalues (tiny table per run).
3. **Preconditioner diagonal**: optional `preconditioner_diag.csv` storing
   `(ix,iy,kxG,kyG,diag_value)`.
4. **Residual spectra**: when an iteration stagnates, dump the residual block as
   a binary/CSV snapshot for offline SVD (gated behind a `max_iterations` guard).
5. **Theta workspace hash**: compute SHA256 of ε(x,y) and cached |k+G|² arrays so
   we can detect accidental differences between supposedly identical configs.

## Usage Summary

- **Real-space / Fourier dielectric dumps**: `mpb2d-lite --dump-pipeline artifacts/run001`
  or in TOML:

  ```toml
  [inspection]
  output_dir = "artifacts/run001"
  dump_eps_real = true
  dump_eps_fourier = true
  ```

- Extend `inspection` with future booleans (`dump_kplusg`, `dump_precond_diag`)
  so that each dial corresponds to a concrete file. The CSV layout should always
  include the integer indices plus the physical coordinate to keep downstream
  plotting dead simple.

The guiding principle: every stage that mutates data should have a cheap,
optional "snapshot" tap that produces deterministic artifacts. Start with the
geometry/FFT black boxes (now covered), then layer in smaller taps for the
preconditioner, projector, and residual blocks as we discover new failure modes.
