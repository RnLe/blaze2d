# Build Phases Checklist

Lightweight progress tracker mirroring the implementation roadmap (see `PROJECT_IMPLEMENTATION.md` for full context).

## Phase 0 – Skeleton & Config

- [x] Initialize Cargo workspace and crate directories.
- [x] Stub `Lattice2D` and `ReciprocalLattice2D` types.
- [x] Implement `Geometry2D` with a single circular atom on a square lattice.
- [x] Implement `Grid2D` with indexing helpers.
- [x] Add CLI config parsing stub that echoes parsed TOML.

## Phase 1 – CPU Backend + Toy Operator

- [x] Introduce `Field2D` storage abstraction.
- [x] Define the `SpectralBackend` trait.
- [x] Implement `backend-cpu` powered by `rustfft` (forward/inverse FFT smoke tests).
- [x] Add toy periodic −Δ operator using the backend.
- [x] Run naive power iteration to match analytic eigenvalues for a test lattice.

## Phase 2 – TE/TM Operator with ε(x, y)

- [x] Build `Dielectric2D::from_geometry` for circular air holes.
- [x] Implement TM operator Θ = −∇·(1/ε ∇).
- [x] Wire Lanczos/LOBPCG eigensolver scaffolding.
- [x] Validate bands for a known square lattice vs. MPB reference data.
- [x] Implement TE operator Θ = −(1/ε)∇² for Bloch-periodic fields.
- [x] Generate MPB/Meep TE reference data for a square lattice.
- [x] Validate TE bands for the same lattice against the new reference dataset.

## Phase 3 – Band-Structure Pipeline

- [x] Implement `BandStructureJob` / `BandStructureResult` APIs.
- [x] Provide standard k-path generators for square and hex lattices.
- [x] Extend CLI with `run` command (`config` → bands, CSV output).

## Phase 3.5 – Pipeline Observability

- [x] Emit MPB-like logs describing backend, lattice, polarization, and solver knobs so users can see what the CLI is doing before the heavy work begins.
- [x] Surface dielectric/FFT preparation milestones with elapsed timings.
- [x] Log a compact line per k-point with iteration counts and runtimes, plus a closing summary (k-count + total iterations + elapsed time) for long sweeps.
- [x] Wire a `--quiet` CLI flag (defaulting to verbose) so automation can suppress the progress stream without losing human-friendly defaults.

## Phase 3.9 – Testing & Optimization

**Module-specific `_tests_*.rs` files (ordered from most fundamental to most dependent):**

- [x] `_tests_grid.rs` – validate indexing, wrapping, and serialization helpers.
- [x] `_tests_units.rs` – confirm constants/scaling helpers stay normalized.
- [x] `_tests_lattice.rs` – reciprocal construction, fractional/cartesian round-trips.
- [x] `_tests_geometry.rs` – atom placement, fractional wrapping, overlap resolution.
- [x] `_tests_dielectric.rs` – sampling from geometry, 1/ε invariants, boundary cases.
- [x] `_tests_field.rs` – contiguous storage behavior, indexing, conversions.
- [x] `_tests_backend.rs` – FFT round-trips, Bloch phases, parallel vs. serial parity.
- [x] `_tests_polarization.rs` – TE/TM enum ergonomics and serde.
- [x] `_tests_operator.rs` – TM/TE matvec smoke tests and hermitian checks.
- [x] `_tests_eigensolver.rs` – convergence on analytic and degenerate spectra.
- [x] `_tests_bandstructure.rs` – short hex path smoke test against reference JSON.
- [x] `_tests_io.rs` – config parsing/serialization fixtures.
- [x] `_tests_reference.rs` – reference-data loading and schema validation.
- [x] `_tests_symmetry.rs` – k-path builders, label ordering, irreducible path slicing.

**Broader regression coverage:**

- [ ] Cover FFT/backend plumbing with round-trip, Bloch phase, and parallel vs. serial regression tests.
- [ ] Add geometry/lattice/grid fixtures that confirm reciprocal metrics, atom placement, and dielectric sampling accuracy.
- [ ] Add operator/bandstructure smoke tests that run a short hex path and compare against reference JSON within tolerance.

## Phase 3.99 – Instrumentation & Benchmarks

- [x] Add opt-in metrics logging that captures setup/dielectric/FFT/k-solve timings in a machine-readable file without affecting quiet/verbose logs.
- [x] Ensure the metrics pipeline has zero cost when disabled and is configurable per evaluation/config run.
- [x] Introduce FFT micro-benchmarks (criterion) to compare serial vs. parallel plans; results show serial wins up to at least 256², so we now pin production workloads to the serial backend until >1024² grids justify revisiting parallel.
- [x] Add eigensolver micro-benchmarks so regressions can be isolated outside the full band-structure sweep (Criterion suite now solves real hex TM/TE workloads at Γ/M/K using the CPU backend).

## Phase 4 – 2-Atomic & Oblique Lattices

- [ ] Support multiple basis atoms in `Geometry2D`.
- [ ] Add hexagonal and oblique lattice helpers.
- [ ] Validate parameter sweeps (e.g., two-atom geometries) against MPB.

## Phase 5 – WASM Build + Web Viewer

- [ ] Create wasm-friendly backend wrapper (single-threaded CPU backend).
- [ ] Expose minimal WASM bindings via `wasm-bindgen`.
- [ ] Build lightweight JS/TS demo to compute & plot small-grid bands in-browser.

## Phase 6 – GPU Backend (CUDA)

- [ ] Scaffold CUDA backend crate with `cudarc` plumbing.
- [ ] Wrap cuFFT for 2D transforms + vector ops (dot/axpy/scale).
- [ ] Cross-check eigenfrequencies vs. CPU backend.
- [ ] Benchmark throughput across grids and k-point sweeps.
