1. **Numerical core in 2D (what we’re actually solving)**
2. **Tech stack (crates & why)**
3. **Workspace & module structure**
4. **Lattice & geometry design (incl. 2-atomic + overlaps – big remark)**
5. **Backend abstraction (CPU now, GPU / WASM later)**
6. **“Phase-2” high-leverage extras (symmetry, etc.)**
7. **Suggested implementation roadmap**

---

## 1. Numerical core (2D MPB-lite)

We’ll implement a **2D plane-wave / FFT-based Maxwell eigenproblem** similar in spirit to MPB, but restricted to:

* 2D periodic structures with scalar ε(x, y),
* TE and TM polarizations (no full 3D vector Maxwell),
* Uniform grid, operator applied via FFTs and pointwise operations.

### 1.1 Physics model

For 2D photonic crystals (invariant in z, periodic in x,y):

* **TM** (E_z out of plane) or **TE** (H_z out of plane) polarization.
* Time-harmonic fields with Bloch condition:
  [
  f(\mathbf{r} + \mathbf{R}) = e^{i \mathbf{k}\cdot \mathbf{R}} f(\mathbf{r})
  ]

Standard scalar eigenproblem (TM case as example):

[
\nabla \cdot \left(\frac{1}{\varepsilon(\mathbf{r})} \nabla H_z(\mathbf{r})\right) + \frac{\omega^2}{c^2} H_z(\mathbf{r}) = 0
]

On a uniform grid, we represent (H_z(\mathbf{r})) and (\varepsilon(\mathbf{r})); we use FFTs to handle derivatives / G-space operations (like MPB: ε(r) local in r, derivatives simple in k+G).

So: **one matvec** =

1. field in **real space**,
2. FFT → **G-space**, apply derivative factors (k+G),
3. inverse FFT → **real space**, multiply by 1/ε(r),
4. FFT again if needed,
5. combine to give Θₖ f.

We never form a big dense matrix; we just provide an `apply(k, v) -> w` operator to an iterative eigensolver (Lanczos/LOBPCG).

---

## 2. Tech stack (Rust 2024, crates & rationale)

We want:

* **Pure Rust CPU implementation first**, portable to WASM.
* Later: **GPU backends** (CUDA via Rust, and potentially WebGPU).

### 2.1 Language & edition

* Rust stable with **edition 2024** (`edition = "2024"` in Cargo.toml). Rust 2024 is now stable in Rust 1.85.0.

### 2.2 Core numeric crates

**FFT**

* `rustfft` – high-performance, SIMD-accelerated, pure Rust FFT.

  * Works on stable Rust.
  * Well-maintained, used in multiple ecosystems.
  * AVX-optimized on x86_64 automatically.

(**Alternative**: `PhastFT` is interesting but requires nightly; I’d stick with `rustfft` for now. )

**Linear algebra / BLAS-like**

For this project, most operations are:

* vector axpy,
* inner products,
* norms,
* small dense matrices (e.g. 2×2 metric, k-path transforms, etc.).

We actually don’t *need* a full BLAS (we can hand-roll loops), but for future extensibility:

* `faer` – high-performance linear algebra library for medium/large matrices; pure Rust, modern API.

We can use `faer` for:

* small dense decompositions if needed,
* convenience in some transforms,
* potential future dense eigenanalysis in small auxiliary problems.

**Array / grid handling**

For 2D grids:

* You can either:

  * Use `Vec<Complex<f64>>` + manual indexing (fastest, easiest for WASM), or
  * Use `ndarray` for more ergonomic 2D indexing.

I’d suggest:

* **Core storage** = contiguous `Vec<Complex<f64>>` (or `Vec<f64>` interleaved) in a `Field2D` struct, with manual indexing and row-major layout.
* Optionally: use `ndarray` only in test/prototype code.

This keeps the core **backend-friendly** (easy to port to CUDA/WebGPU).

**Complex numbers**

* `num-complex` for `Complex<f64>`.

**Parallelism**

* `rayon` – CPU parallelism for loops (e.g. batched FFTs, multiple k-points) in native builds.
* Gate it using features (so WASM builds can be single-threaded).

### 2.3 Config & bindings

* `serde` + `serde_json`/`toml` – config files.
* `clap` – CLI argument parsing.
* `thiserror` – typed error handling.
* `pyo3` – Python bindings (future crate `mpblite-py`).
* `wasm-bindgen` – WASM bindings (for a browser demo / interactive viewer).

### 2.4 GPU ecosystem (for later)

Not used in v1, but we want to design for it:

* **CUDA backend (desktop)**:

  * `cudarc` – modern, high-level CUDA bindings; widely used and recommended by people doing real scientific CUDA in Rust.
  * Alternatively `cust` from the Rust CUDA project.

* **WebGPU backend (for browser GPU)**:

  * `wgpu` – cross-platform implementation of WebGPU; can do compute shaders and runs both native and in browser.

We’ll design the **FFT + vector ops behind a trait**, so CPU uses `rustfft`, CUDA backend uses cuFFT (via `cudarc`), and WebGPU backend uses compute shaders.

---

## 3. Workspace & module structure

Use a **Cargo workspace** with multiple crates:

```text
mpb2d-lite/
├─ Cargo.toml            # workspace
├─ crates/
│  ├─ core/              # mpb2d-core: math, physics, operators, eigensolver
│  │   └─ src/
│  ├─ backend-cpu/       # mpb2d-backend-cpu: FFT + BLAS on CPU (rustfft + rayon)
│  ├─ backend-wasm/      # mpb2d-backend-wasm: thin wrapper for wasm32 (optional)
│  ├─ backend-cuda/      # mpb2d-backend-cuda: future CUDA backend (cudarc)
│  ├─ cli/               # mpb2d-cli: command line interface
│  └─ python/            # mpb2d-py: pyo3 bindings
└─ examples/
   ├─ square_air_hole.toml
   ├─ hex_air_hole.toml
   └─ oblique_two_atom.toml
```

### 3.1 `mpb2d-core` crate – logical modules

Inside `crates/core/src`:

```text
core/src/
├─ lib.rs
├─ units.rs              # physical units & constants
├─ lattice.rs            # Bravais lattice, reciprocal lattice, k-mesh, paths
├─ geometry.rs           # basis, inclusions (circles), 2-atom, overlaps
├─ dielectric.rs         # builds ε(x,y) field on grid
├─ grid.rs               # Grid2D struct, indexing, metrics
├─ polarization.rs       # TE/TM enums, equations
├─ backend.rs            # traits for FFT + vector ops
├─ operator.rs           # Θ_k operator (matvec)
├─ eigensolver.rs        # Lanczos/LOBPCG on top of Operator
├─ bandstructure.rs      # high-level "run band calculation" API
├─ symmetry.rs           # (Phase-2) BZ paths, point-group symmetry helpers
└─ io.rs                 # config structs (serde), results (HDF5/CSV/etc.)
```

> 🔎 **Module layout note**: we stick to Rust's modern module style (each module uses `name.rs` alongside an optional `name/` folder) and avoid legacy `mod.rs` files throughout the workspace.

And then each of the support crates implements the backend trait(s):

* `backend-cpu/src/lib.rs` – implements `Backend` using `rustfft` + `rayon`.
* `backend-wasm` – reuses CPU backend but disables rayon or uses wasm-thread helpers.
* `backend-cuda` – future: implements `Backend` using CUDA + cuFFT.

---

## 4. Lattice & geometry design (big remark section)

This is where we define:

* 2D Bravais lattice, reciprocal lattice,
* basis atoms,
* simple shapes (circles in 2D),
* how to handle 2-atomic basis and overlapping geometries.

### 4.1 Bravais lattice abstraction

We want **all lattice types** (square, rectangular, hexagonal, oblique) with one data structure:

```rust
pub struct Lattice2D {
    pub a1: [f64; 2],   // primitive vectors in Cartesian coords
    pub a2: [f64; 2],
}
```

Provide convenience constructors:

```rust
impl Lattice2D {
    pub fn square(a: f64) -> Self { ... }
    pub fn rectangular(a: f64, b: f64) -> Self { ... }
    pub fn hexagonal(a: f64) -> Self { ... }  // 60° angle
    pub fn oblique(a1: [f64; 2], a2: [f64; 2]) -> Self { ... }
}
```

Then define **reciprocal lattice**:

```rust
pub struct ReciprocalLattice2D {
    pub b1: [f64; 2],
    pub b2: [f64; 2],
}
```

and a function `Lattice2D::reciprocal()`.

This gives you:

* General Bravais lattice,
* Hard-coded BZ paths for familiar types (square: Γ–X–M–Γ; hex: Γ–M–K–Γ; etc.) in `symmetry.rs`.

### 4.2 Basis & inclusions (2-atomic etc.)

Basis positions: fractional coordinates inside unit cell:

```rust
pub struct BasisAtom {
    pub pos: [f64; 2],   // fractional coords in [0,1)x[0,1)
    pub radius: f64,     // hole radius in units of |a1|, for example
    pub eps_inside: f64, // permittivity inside (air hole = 1.0)
}
```

A geometry is then:

```rust
pub struct Geometry2D {
    pub lattice: Lattice2D,
    pub eps_bg: f64,
    pub atoms: Vec<BasisAtom>, // 1-atomic, 2-atomic, ... arbitrary
}
```

**2-atomic basis** is just `atoms.len() == 2`. Overlapping disks are simply multiple atoms whose circles intersect.

We’ll **evaluate ε(r)** on a real-space grid:

* For each grid point r(x,y):

  * Convert to *fractional* coordinates (u,v) such that `r = u a1 + v a2`.
  * Bring (u,v) into [0,1)² (mod 1).
  * For each atom:

    * compute real position `r_atom = atom.pos[0]*a1 + atom.pos[1]*a2`,
    * compute distance |r – r_atom| in Cartesian coords,
    * if inside radius → treat as inside hole.

**How to combine multiple atoms at one point?**

Most straightforward for air holes in dielectric background:

* Assume all holes are **same material** (air) with ε_in = 1.
* Evaluate all atoms; if point is inside at least one hole → `eps = eps_hole` (1.0), else `eps = eps_bg`.

Then overlapping just means: union of disks; no extra complexity.

If later you want **different materials / overlapping materials**:

* You can define a priority or functional combination, e.g. last atom wins, or use a list of materials and some mixing rule.
* But overlapped circles themselves are trivial in this union model.

> 🔴 **Big remark about complexity**
>
> * **Most straightforward v1**:
>
>   * Only one “hole material” (air) in one “background material”.
>   * Any number of holes (basis atoms), all with same ε_inside = 1.
>   * Overlaps are just union; simple “inside any hole → air”.
> * **Future extension** (moderate complexity):
>
>   * Multiple materials (e.g. two different hole materials in same cell).
>   * Define a material **priority** or layering model:
>
>     * e.g. evaluate atoms in order and override ε(r) each time.
>   * Still easy if geometries are all circles.
> * **Hard extensions** (much more complex):
>
>   * Arbitrary polygonal shapes, curved boundaries, anisotropic ε tensors.
>   * Multiple layers in z (even though you stay 2D), e.g. effective anisotropy.
>   * Accurate treatment of subpixel positioning (smearing / smoothing).
>
> Overlapping circles with 2-atomic basis is **not** the hard part. The difficulty starts when you want many different materials and complex shapes.

---

## 5. Backend abstraction (CPU first, GPU/WASM later)

You want:

* CPU implementation now (for validation, WASM),
* GPU later without rewriting physics.

So we define a **backend trait** that abstracts FFT + vector ops.

### 5.1 Backend traits

In `core/backend.rs`:

```rust
use num_complex::Complex64;

pub struct Field2D<'a> {
    pub data: &'a mut [Complex64],
    pub nx: usize,
    pub ny: usize,
}

pub trait SpectralBackend {
    type Buffer;    // backend-specific buffer handle

    fn alloc_field(&self, nx: usize, ny: usize) -> Self::Buffer;

    fn forward_fft_2d(&self, field: &mut Self::Buffer);
    fn inverse_fft_2d(&self, field: &mut Self::Buffer);

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64;
    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer);
    fn scale(&self, alpha: Complex64, x: &mut Self::Buffer);

    // Potentially: pointwise_mul, etc.
}
```

Then the **operator** and **eigensolver** are generic over `B: SpectralBackend`.

### 5.2 CPU backend (v1)

`backend-cpu` implements `SpectralBackend`:

* `Self::Buffer` = `FieldCpu { data: Vec<Complex64>, nx, ny }`.
* `forward_fft_2d`:

  * Use `rustfft` to do FFT along x, then along y (or use a wrapper that does 2D).
* `dot`, `axpy`, `scale`: simple loops, optionally parallelized with `rayon`.

Because `rustfft` is pure Rust, this backend is **WASM-friendly**; we just disable `rayon` for wasm32.

### 5.3 CUDA backend (later)

`backend-cuda`:

* `Self::Buffer` = device pointer plus metadata.
* `forward_fft_2d`:

  * Use cuFFT via `cudarc` to plan batched 2D FFTs.
* `dot`, `axpy`, `scale`: use cuBLAS or custom kernels.

Interface to higher layers stays identical.

### 5.4 WASM

For WASM, we just compile with:

* `backend-cpu` but single-threaded.
* Build target `wasm32-unknown-unknown` + `wasm-bindgen` in `crates/backend-wasm` or `crates/cli` to expose a JS/TS API.

---

## 6. Operator, eigensolver, and band-structure pipeline

### 6.1 Grid & indexing

`grid.rs`:

```rust
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub lx: f64,    // physical lengths in some units
    pub ly: f64,
    // might store lattice, or just the real-space step
}

impl Grid2D {
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }
}
```

### 6.2 Dielectric field

`dielectric.rs`:

```rust
pub struct Dielectric2D {
    pub eps_r: Vec<f64>, // length = nx * ny
    pub grid: Grid2D,
}

impl Dielectric2D {
    pub fn from_geometry(geom: &Geometry2D, grid: &Grid2D) -> Self {
        // Loop over grid points, evaluate eps(r)
    }
}
```

Later we can also store `1/eps` to avoid divisions.

### 6.3 Polarization & operator

`polarization.rs`:

```rust
pub enum Polarization {
    TE,
    TM,
}
```

`operator.rs` defines:

```rust
pub struct ThetaOperator<B: SpectralBackend> {
    pub backend: B,
    pub dielectric: Dielectric2D,
    pub pol: Polarization,
    pub kx: f64,
    pub ky: f64,
    // maybe store Gx/Gy factors etc.
}

impl<B: SpectralBackend> ThetaOperator<B> {
    pub fn apply(&self, x: &B::Buffer, y: &mut B::Buffer) {
        // 1. copy x -> temp real-space
        // 2. FFT, apply k+G factors
        // 3. inverse FFT, multiply by 1/eps
        // 4. final operations to form Θ_k x
    }
}
```

We keep this **matrix-free**.

### 6.4 Eigensolver

`eigensolver.rs`:

Implement something like a **shifted, normalized Lanczos** or **LOBPCG**:

```rust
pub struct EigenOptions {
    pub n_bands: usize,
    pub max_iter: usize,
    pub tol: f64,
}

pub struct EigenResult {
    pub omegas: Vec<f64>,       // eigenfrequencies (or ω²)
    pub modes: Option<Vec<B::Buffer>>, // optionally store eigenvectors
}

pub fn solve_lowest_eigenpairs<B: SpectralBackend>(
    op: &ThetaOperator<B>,
    opts: &EigenOptions,
) -> EigenResult {
    // Lanczos/LOBPCG using op.apply
}
```

You can design it so you can plug in different solvers later (even external ones, e.g. wrap ARPACK via FFI if you really want).

### 6.5 Band-structure pipeline

`bandstructure.rs`:

```rust
pub struct BandStructureJob {
    pub geom: Geometry2D,
    pub grid: Grid2D,
    pub pol: Polarization,
    pub k_path: Vec<[f64; 2]>,    // list of k-points
    pub eigen_opts: EigenOptions,
}

pub struct BandStructureResult {
    pub k_path: Vec<[f64; 2]>,
    pub bands: Vec<Vec<f64>>,     // bands[n_k][n_band] = ω
}

pub fn run_bandstructure<B: SpectralBackend>(
    backend: &B,
    job: &BandStructureJob,
) -> BandStructureResult {
    let dielectric = Dielectric2D::from_geometry(&job.geom, &job.grid);

    let mut bands_for_all_k = Vec::new();

    for (kx, ky) in job.k_path.iter().copied() {
        let op = ThetaOperator {
            backend: backend.clone_or_handle(),
            dielectric: dielectric.clone_shallow(), // or ref
            pol: job.pol,
            kx, ky,
        };

        * Emit MPB-like logs describing backend, lattice, polarization, and solver knobs so users can see what the CLI is doing before the heavy work begins.
        * Surface dielectric/FFT preparation milestones with elapsed timings.
        * Log a compact line per k-point with iteration counts and runtimes, plus a closing summary (k-count + total iterations + elapsed time) for long sweeps.
        * Wire a `--quiet` CLI flag (defaulting to verbose) so automation can suppress the progress stream without losing human-friendly defaults.
    BandStructureResult {
        k_path: job.k_path.clone(),
        bands: bands_for_all_k,
    }
}
```

(Details around ownership and cloning can be designed nicely; above is schematic.)

---

## 7. “Phase-2” high-leverage extras (symmetry, etc.)

These are **not required** in v1 but give big leverage once the core is stable.

### 7.1 Symmetry and irreducible Brillouin zone

* Implement lattice-specific **standard k-paths**:

  * Square: Γ(0,0) → X(½,0) → M(½,½) → Γ.
  * Rectangular: Γ → X → S → Y → Γ.
  * Hexagonal: Γ → M → K → Γ.

* Provide a helper:

  ```rust
  pub enum PathType { Standard, Custom(Vec<[f64; 2]>) }

  pub fn generate_k_path(
      lattice: &Lattice2D,
      path_type: PathType,
      n_segments: usize,
  ) -> Vec<[f64; 2]> { ... }
  ```

* Later: implement **point-group Irreps** (C4v, C6v) and project fields onto symmetry subspaces to separate mode families. This is more advanced but gives:

  * Cleaner identification of degeneracies.
  * Potential factor ~2× speedups per k in some cases.

### 7.2 TE/TM decomposition and mode labeling

* You already have TE/TM as separate runs.
* Later: add small routines to classify modes by symmetry (e.g. even/odd wrt mirror planes).

### 7.3 Real/complex symmetries in Fourier space

For certain k-points / lattices, fields can be chosen real up to phase. You can exploit that to:

* Halve storage,
* Speed up some operations.

This is subtle and not necessary at first, but can be added later.

### 7.4 Caching & parameter sweeps

* Caching of:

  * dielectric field ε(r) when only k changes,
  * FFT plans,
  * precomputed (k+G) factors for a given k-path.

* Use this to sweep:

  * radius r/a,
  * ε_bg,
  * etc., with minimal recomputation.

### 7.5 Analytic structure factors (very advanced)

For simple geometries (circular rods), one can compute analytic **ε(G)** or structure factors and sometimes design better preconditioners.

This is deep and optional, but conceptually:

* Use analytic Fourier coefficients for rods to generate ε(r) or build preconditioners.

---

## 8. Implementation roadmap (how to actually build this)

A realistic plan that yields working artifacts at each stage:

### Phase 0 – Skeleton & config

* Setup workspace & crates.
* Implement:

  * `Lattice2D`, `ReciprocalLattice2D`.
  * `Geometry2D` with one circular atom, square lattice.
  * `Grid2D`.
* Implement simple config parsing (TOML) and CLI stub that prints parsed config.

### Phase 1 – CPU backend + trivial operator

* Implement `Field2D` and `SpectralBackend` trait.

* Implement `backend-cpu` with `rustfft`:

  * Basic 2D FFT under test: forward then inverse yields original.

* Implement a **toy operator**:

  * For example, just −Δ on a periodic domain (no ε yet).
  * Implement a naive power iteration to compute the lowest mode; compare to known analytic eigenvalues for a simple domain as a sanity check.

### Phase 2 – TE/TM operator with ε(x,y)

* Implement `Dielectric2D::from_geometry` for circular air holes.

* Implement TM operator in MPB style:

  * Θ = −∇·(1/ε ∇).

* Use Lanczos (or simple LOBPCG) eigensolver.

* Validate by:

  * Reproducing a known 2D band diagram (e.g. square lattice rods) from MPB paper / user guide.
  * Using the Python `generate_square_tm_bands.py` helper to capture MPB/Meep TM bands (`python/reference-data/square_tm_uniform_mpb.json`) and comparing the Rust TM operator against those values inside `tm_operator_tracks_uniform_reference_data`.
  * Mirroring the same workflow for TE polarization by implementing the TE branch of `ThetaOperator`, generating `python/reference-data/square_te_uniform_mpb.json`, and adding the `te_operator_tracks_uniform_reference_data` regression test.

### Phase 3 – Full band-structure pipeline

* Implement `BandStructureJob` and `BandStructureResult`.
* Implement k-path generator for square and hex lattices.
* CLI:

  * `mpb2d-lite run --config config.toml --output bands.csv`.

> **Current status:** The codebase now includes a working `BandStructureJob` runner, MPB-style CLI output, and preset-based k-path generation (square/hex) that can be invoked via config or `--path`. Remaining work in this phase is primarily richer CLI ergonomics (alternate backends, automatic labeling, summary stats).

### Phase 3.5 – Pipeline observability

* Emit MPB-like logs describing backend, lattice, polarization, and solver knobs so users can see what the CLI is doing before the heavy work begins.
* Surface dielectric/FFT preparation milestones with elapsed timings.
* Log a compact line per k-point with iteration counts and runtimes, plus a closing summary (k-count + total iterations + elapsed time) for long sweeps.
* Wire a `--quiet` CLI flag (defaulting to verbose) so automation can suppress the progress stream without losing human-friendly defaults.

### Phase 3.9 – Testing & optimization

* Stand up dedicated `_tests_*.rs` companions next to every major module (eigensolver, operator, backend FFT plumbing, lattice/geometry, etc.) and migrate existing inline tests there.
* Expand eigensolver coverage with deterministic fixtures that validate convergence on simple Hermitian operators, noisy inputs, and edge conditions (degenerate spectra, zero/negative shifts).
* Add FFT/back-end regression tests that round-trip random fields, stress Bloch phase factors, and verify parallel execution matches serial results.
* Introduce lightweight benchmarks or profiled smoke tests to track per k-point iteration counts and spot regressions when tweaking precision/parallelism knobs.

> **Current status:** Every module-specific companion now exists, including fresh suites for `io`, `reference`, and `symmetry` that exercise config serde fallbacks, reference JSON loading, and k-path densification. The remaining open work in this phase is the broader regression/benchmarking bullets above.

### Phase 3.99 – Instrumentation & benchmarks

The current solver spends most of its wall time inside FFT prep and the Lanczos loop, and we lack structured metrics to understand why. This phase adds two complementary guardrails:

1. **Opt-in metrics stream.** A new config block enables JSONL metrics that mirror the existing human-friendly logs but capture richer context (backend type, grid, k-index, elapsed seconds, iteration counts, etc.) for every heavyweight stage: setup, dielectric sampling, FFT workspace allocation, and each k-point solve. Metrics default to fully disabled and short-circuit at compile time so there is effectively zero runtime overhead unless an output file is requested.
2. **Targeted micro-benchmarks.** Lightweight benchmarks (criterion or `cargo bench`) will cover the spectral backend/FFT helpers and the eigensolver loop separately from the full pipeline, making it easy to bisect regressions or compare serial vs. parallel backends without running full band diagrams.

Together these tools let us spot misconfigurations (e.g., non-converging eigensolver) quickly and focus optimization work where it matters.

> **Current status:** Metrics logging is landed and can be toggled per config to stream JSONL events for setup, dielectric sampling, FFT prep, each k-point solve, and pipeline completion. Criterion benches now cover both hotspots: the FFT suite stresses serial vs. parallel plans, and the eigensolver suite instantiates real hex TM/TE workloads (Gamma/M/K) so we can time the actual Lanczos solves without running a full band sweep.

> **FFT benchmark decision:** Criterion runs up to 256×256 grids show the "parallel" FFT path never outruns the serial path—the Rayon + transpose overhead dominates until well beyond 1024² points. To avoid wasting milliseconds per matvec, we now force the CPU backend to stay on the serial plan for production workloads and retain the benchmark purely as a regression guardrail. Re-enable/retune the parallel variant only if we add much larger grids or a GPU backend that can actually amortize the extra orchestration.

### Phase 4 – 2-atomic basis & oblique lattice

* Extend `Geometry2D` to multiple basis atoms.
* Add hex and oblique convenience constructors.
* Validate parameter sweeps vs MPB for a few 2-atomic geometries.

### Phase 5 – WASM build + minimal web viewer

* Add `mpb2d-wasm` or integrate into `cli` + `wasm-bindgen`.
* Build a small JS/TS demo that:

  * Accepts simple parameters (lattice type, r/a),
  * Calls WASM function to compute a few bands for a *small grid*,
  * Plots band diagram in browser.

### Phase 6 – GPU backend (CUDA)

* Create `backend-cuda`:

  * Use `cudarc` to allocate device buffers, transfer ε(r).
  * Wrap cuFFT for 2D FFT.
  * Implement dot, axpy with cuBLAS or custom kernels.
* Benchmark vs CPU backend:

  * Check correctness (ω differences).
  * Measure speedup for large grids and many k-points.

---

### Living module documentation

This repository keeps per-module development notes under `doc/`. Each file (for example `doc/geometry.md`) describes the code that exists **today**: key types, available APIs, and known gaps. When you modify a module, update its companion document so the snapshots remain accurate. Treat these notes as short-form dev logs; broader intent still lives in this implementation guide.

### Testing conventions

All Rust unit and integration tests live in standalone files prefixed with `_tests_` within each crate (for example `crates/core/src/_tests_eigensolver.rs`). Each test file mirrors a production module and re-exports only the public APIs it exercises. When you add or modify a module:

1. Create/extend the matching `_tests_*.rs` file instead of sprinkling `#[cfg(test)]` blocks inside the implementation.
2. Keep fixtures small and deterministic—use analytic fields, MPB reference JSON, or hand-rolled operators that complete in milliseconds.
3. Document tricky invariants at the top of the test file so future contributors understand what “correct” means (e.g., normalization, Hermitian symmetry, FFT round-trip tolerances).

This layout keeps production modules free of test-only imports, simplifies feature gating (especially for WASM), and makes it obvious where to add new coverage.
