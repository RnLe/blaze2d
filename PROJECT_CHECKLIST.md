# Build Phases Checklist

Lightweight progress tracker mirroring the implementation roadmap (see `PROJECT_IMPLEMENTATION.md` for full context).

## Phase 0 – Skeleton & Config

- [ ] Initialize Cargo workspace and crate directories.
- [ ] Stub `Lattice2D` and `ReciprocalLattice2D` types.
- [ ] Implement `Geometry2D` with a single circular atom on a square lattice.
- [ ] Implement `Grid2D` with indexing helpers.
- [ ] Add CLI config parsing stub that echoes parsed TOML.

## Phase 1 – CPU Backend + Toy Operator

- [ ] Introduce `Field2D` storage abstraction.
- [ ] Define the `SpectralBackend` trait.
- [ ] Implement `backend-cpu` powered by `rustfft` (forward/inverse FFT smoke tests).
- [ ] Add toy periodic −Δ operator using the backend.
- [ ] Run naive power iteration to match analytic eigenvalues for a test lattice.

## Phase 2 – TE/TM Operator with ε(x, y)

- [ ] Build `Dielectric2D::from_geometry` for circular air holes.
- [ ] Implement TM operator Θ = −∇·(1/ε ∇).
- [ ] Wire Lanczos/LOBPCG eigensolver scaffolding.
- [ ] Validate bands for a known square lattice vs. MPB reference data.

## Phase 3 – Band-Structure Pipeline

- [ ] Implement `BandStructureJob` / `BandStructureResult` APIs.
- [ ] Provide standard k-path generators for square and hex lattices.
- [ ] Extend CLI with `run` command (`config` → bands, CSV output).

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
