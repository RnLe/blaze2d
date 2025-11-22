Nice, this is exactly the level where “copying MPB’s brain” pays off.

I’ll treat your notes as the skeleton and turn them into a **solver design doc**:

* First: **physics** – what A and B actually are in 2D TE/TM.
* Then: **step-by-step MPB-style pipeline** (your 1–10), adapted to 2D.
* For each step: what it does, why it matters, and CPU vs GPU comments.
* I’ll highlight which pieces are *essential optimizations* you really want in `mpb2d-lite`.

---

## 0. Physics: the generalized Maxwell eigenproblem in 2D

MPB works in **H-field formulation**, solving

[
\nabla \times \varepsilon^{-1}(\mathbf r), \nabla \times \mathbf H(\mathbf r)
= \left(\frac{\omega}{c}\right)^2 \mu(\mathbf r), \mathbf H(\mathbf r)
]

with Bloch boundary conditions (\mathbf H(\mathbf r + \mathbf R)=e^{i\mathbf k\cdot \mathbf R}\mathbf H(\mathbf r)).

This can be written as a Hermitian generalized eigenproblem

[
A \mathbf H = \lambda, B \mathbf H, \quad
\lambda = \left(\frac{\omega}{c}\right)^2,
]

with

* (A = \nabla \times \varepsilon^{-1} \nabla \times),
* (B = \mu) (often just the identity if (\mu=1)).

### 2D TE / TM specialization

For a truly 2D structure (invariant in z, ε = ε(x,y), μ = μ(x,y)):

* **TM polarization** (out-of-plane (H_z)):

  * Unknown: scalar (H_z(x,y)).
  * Equation (using (H_z) formulation):

    [
    \nabla\cdot\left(\frac{1}{\varepsilon}\nabla H_z\right)

    * \left(\frac{\omega}{c}\right)^2 H_z = 0.
      ]

    So (A = -\nabla\cdot \varepsilon^{-1}\nabla), (B = 1).

* **TE polarization** (out-of-plane (E_z)):

  * Unknown: scalar (E_z(x,y)).
  * Equation:

    [
    \nabla\cdot\left(\frac{1}{\mu}\nabla E_z\right)

    * \left(\frac{\omega}{c}\right)^2 \varepsilon E_z = 0.
      ]

    If μ = 1, you can choose either E- or H-formulation; the generalized eigenproblem typically becomes (A E_z = \lambda B E_z) with (A=-\nabla^2), (B=\varepsilon).

For your 2D solver it’s clean to keep the **generalized Hermitian form**:

[
A u = \lambda B u, \quad \lambda = (\omega/c)^2,
]

with:

* TM: (A = -\nabla\cdot\varepsilon^{-1}\nabla,\ B = I).
* TE: (A = -\nabla^2,\ B = \varepsilon).

Numerically you’ll:

* represent (u(x,y)) and ε(x,y) on a uniform grid,
* implement ∇ / ∇· via FFT (pseudo-spectral) and multiply by ε, ε⁻¹ in real space.

Everything in your notes (deflation, parity, preconditioning) sits on *top* of this generalized eigenproblem.

---

## 1. High-level MPB-style pipeline for **solve_k_point**

This is the implementation checklist, mapped to your steps 1–10.

I’ll assume:

* we already have a **geometry** → ε(x,y) on grid,
* we have a list of **k-points** in the first Brillouin zone,
* we handle one k at a time.

---

### Step 1 — Γ-point special case & analytic curl-free / constant modes

> Your note: `||k|| < 10^-10 → Γ point; exploit analytic curl-free modes`

**Concept**

At k = 0 in Maxwell’s equations, there are trivial “curl-free” or “constant” modes that correspond to **ω = 0** (gauge-like modes).

In the 2D scalar formulation:

* The constant function (u(x,y)=\text{const}) satisfies
  (\nabla u = 0 \Rightarrow A u = 0).
* This gives a zero eigenvalue λ=0 that we **know analytically** and rarely care about.

Keeping that mode in the Krylov subspace:

* pollutes the spectrum (massive eigenvalue separation),
* slows convergence,
* may cause numerical issues.

**What to do**

* Detect when (|\mathbf k| < k_{\text{tol}}) (e.g. 10⁻¹⁰).

* Define a **constant vector** (u_0) with all entries 1 on the grid, then B-normalize:

  [
  u_0 \leftarrow u_0 / \sqrt{u_0^* B u_0}.
  ]

* Treat (u_0) as an **analytically known eigenvector with λ=0**.

* Project all search vectors and residuals onto the orthogonal complement of (u_0) in the B-inner product:

  [
  v \leftarrow v - u_0 (u_0^* B v).
  ]

This is a special case of deflation (steps 4–5), but worth keeping explicit because it always applies at Γ.

**Can we use curl-free modes beyond constant?**

For the full 3D H formulation there’s a larger nullspace of gradient fields; in 2D TE/TM scalar you essentially only get **constant** (and trivial phase factors from Bloch). So for `mpb2d-lite`, “analytic curl-free modes” = *just handle the constant mode at Γ*.

**CPU vs GPU**

* This is **small dense linear algebra** on 1–few vectors: best on CPU.
* GPU is pointless here; keep deflated modes in host memory (or mirror them) and apply projection on CPU or with very small GPU kernels.

---

### Step 2 — Reset H, E, D (fields & buffers)

> “Reset H, E, D” – purely technical.

Implementation meaning:

* Zero or re-initialize all working vectors for the eigensolver at this k:

  * block of approximate eigenvectors (X),
  * residuals (R),
  * preconditioned residuals (P),
  * temporary workspaces for FFTs.

This is mostly **bookkeeping**.

**CPU vs GPU**

* Small overhead; where these arrays live is dictated by backend:

  * CPU backend: allocate & reset on host.
  * GPU backend: allocate & reset on device; minimize host/device copies.

---

### Step 3 — Precompute all **k + G** (filling reciprocal space up to cut-off)

> “Calculate all k + G_m = k + u₁b₁ + u₂b₂ (+ u₃b₃)… (fills entire space up to a cut-off G²)”

**Concept**

In plane-wave / pseudo-spectral methods:

* Fourier modes are labelled by reciprocal lattice vectors (G = m_1 b_1 + m_2 b_2).
* For Bloch wavevector k, derivatives become:

  [
  \nabla \leftrightarrow i (\mathbf k + \mathbf G).
  ]

The **symbol** of the operator and the preconditioner use (|k+G|^2), etc.

You have two layers:

1. Real-space grid (for FFT): indices (i,j) = 0..N_x-1, 0..N_y-1.
2. Corresponding “Fourier indices” (k_x index, k_y index) representing G.

You precompute:

* arrays `kx_plus_G[i,j]`, `ky_plus_G[i,j]`,
* `kG_norm_sq[i,j] = |k+G|²`.

These are used in both:

* operator application A u,
* preconditioner M⁻¹ u.

**Algorithm**

For each grid Fourier index `(ix, iy)`:

* Compute integer wave numbers (m_x, m_y) (with FFT frequency mapping).
* Compute (G = m_x b_1 + m_y b_2).
* Set `kG = k + G`.

Store:

* `kGx[ix,iy], kGy[ix,iy]`,
* optionally `1/|k+G|²` for preconditioner.

**CPU vs GPU**

* CPU: small cost, done once per k; vectorizable.
* GPU: you can precompute on CPU and upload, or compute on GPU once and reuse.
* It’s not performance-critical; just do it where convenient.

---

### Step 4 — Γ-point constant band deflation

> “For k=0, constant bands are deflated so solver only works on non-trivial bands”

This is the same as Step 1, but emphasised:

* At k=0, constant eigenvector (u_0) has λ=0.
* You **don’t want the solver to converge to this trivial band** when you ask e.g. for “lowest 4 bands”.

So:

* Include (u_0) in the deflation subspace Y (see next step).
* Or special-case: “we never include band 0, we always start from band index ≥ 1” and always project out `u0`.

---

### Step 5 — General deflation of previously found bands

> “Deflation: Make new search directions orthogonal to earlier bands
> (X \leftarrow (I - Y (Y^T B Y)^{-1} Y^T B) X)”

**Concept**

When computing multiple eigenpairs (bands) sequentially or in blocks, deflation ensures:

* New search directions are orthogonal (in B-inner product) to previously converged eigenvectors.
* Prevents the solver from re-finding the same bands.
* Improves convergence for higher bands.

Mathematically:

* Let (Y) be a matrix whose columns are the **deflated eigenvectors** (including constant mode u₀, possibly previous k-point eigenvectors if you use warm starts).

* Define the **B-orthogonal projector** onto the complement:

  [
  P = I - Y (Y^* B Y)^{-1} Y^* B.
  ]

* If Y is B-orthonormal ( (Y^* B Y = I) ), this simplifies to (P = I - Y Y^* B).

You apply P to:

* Initial guess block X,
* Residuals R,
* Preconditioned residuals P (the letter clash is unfortunate; use different symbol in code).

**Algorithm**

1. Maintain a block Y of B-orthonormal eigenvectors (size N×p).

2. For any block X:

   * Compute `C = Y^* B X` (small p×m dense matrix).
   * Compute `X <- X - Y C`.

3. This guarantees B-orthogonality: (Y^* B X = 0).

**CPU vs GPU**

* The “large” operations (Y^* B X, Y C, X − Y C) are **matrix–block-vector multiplies**:

  * Good for GPU: high flops, large arrays.
  * But also fine on CPU; for now you can implement them with simple loops or `rayon` parallelism.

* The “small” inverse ((Y^* B Y)^{-1}) is p×p, done on CPU with `faer` or similar.

In v1 you can:

* Always B-orthonormalize Y → (Y^* B Y)=I, no need to invert.
* Keep deflation strictly **within one k-point** to avoid complexity.

---

### Step 6 — Parity constraints (mirror symmetries)

> “Parity constraints: enforces required mirror symmetry (e.g. even/odd modes z → -z)”

**Concept**

If the structure and boundary conditions are symmetric under some symmetry S (mirror, rotation), eigenmodes can be classified by **irreducible representations** of that symmetry:

* For a mirror symmetry S (e.g. (x \to -x)), eigenmodes satisfy

  [
  S u = \pm u.
  ]

* You can restrict your search space to the **even** ( + ) or **odd** ( − ) subspace:

  * Even projector: (P_{+} = \frac{1}{2}(I + S)),
  * Odd projector:  (P_{-} = \frac{1}{2}(I - S)).

Benefits:

* Cuts the effective space dimension (roughly by 2 for a simple mirror).
* Separates mode families (even vs odd), avoids near-degenerate mixing.
* Improves convergence and interpretability.

For 2D:

* You can exploit mirrors in x and/or y, depending on lattice and geometry.
* E.g. square lattice with symmetric inclusion ⇒ symmetry plane at x=0 and y=0.

**Implementation**

* Define a linear operator S acting on grid fields:

  * For each grid point (i,j), there is a symmetric point (i',j').
  * S u(i,j) = u(i',j') (or with a sign if needed).
* Choose parity (even or odd) based on which band family you want.
* Project **all** search vectors into the parity subspace at every iteration:

  * Set (u \leftarrow P_{\pm} u = \frac{1}{2}(u \pm S u)).

This works nicely together with deflation: alternate projection onto parity subspace and deflated subspace.

**CPU vs GPU**

* S is a simple index permutation + add/sub, perfectly GPU-friendly.
* For v1, implement on CPU; later you can fuse parity projection with other vector operations in GPU kernels.

---

### Step 7 — Eigenvalue problem: block-LOBPCG on (A x = \lambda B x)

> “Block-LOBPCG iteratively minimizes Rayleigh quotient
> (\lambda = (X^* A X)/(X^* B X)), with (A = \nabla\times \varepsilon^{-1}\nabla\times,\ B = \mu^{-1}) (or I).”

You don’t *have* to implement LOBPCG specifically, but the key properties to keep:

* **Generalized Hermitian problem** (A x = \lambda B x),
* Work in the **B-inner product**:

  * inner product: ((x,y)_B = x^* B y),
  * Rayleigh quotient: (\lambda(x) = (x, A x)_B / (x,x)_B),
* Solve for blocks of eigenpairs simultaneously (for multiple bands, good for GPU and deflation).

**Block LOBPCG structure (high level)**

For each k:

1. Initialize X (n×m matrix of m trial eigenvectors).
2. Loop until convergence:

   * Apply operator: (W = A X).
   * Form residuals: (R = W - B X \Lambda), where Λ are current Rayleigh quotients (diag).
   * Apply deflation projector P and parity projector.
   * Precondition: (P_R = M^{-1} R).
   * Build a subspace spanned by columns of {X, P_R, maybe old directions}.
   * Solve **small dense generalized eigenproblem** in that subspace to update X and Λ.

As long as you maintain:

* B-orthonormality of X,
* deflation (Y^* B X = 0),
* parity constraints (S X = ± X),

you’ll converge to the desired bands.

**CPU vs GPU**

* Big matvecs A X and B X and preconditioner M⁻¹ X → great for GPU.
* Small subspace dense eigenproblem → use `faer` on CPU.
* Or do small dense work on GPU later; not necessary in v1.

---

### Step 8 — Maxwell operator application (A u)

> “maxwell operator: computes ΔX for each operation
> ΔH = (k+G)× ε⁻¹ · [ (k+G)×H ]”

For full MPB in 3D H-formulation the operator in Fourier space looks like:

[
(\Theta_{\mathbf{k}} \mathbf H)*{\mathbf G}
= (k+G) \times \sum*{G'} \varepsilon^{-1}(G-G') \left[(k+G') \times \mathbf H_{G'}\right].
]

But thanks to FFT, they do:

1. In Fourier space: compute ((k+G)\times H_G),
2. FFT → real space,
3. multiply by ε⁻¹(r),
4. FFT back,
5. multiply by (k+G)× again.

In 2D scalar TE/TM this simplifies.

#### 2D TM (H_z) operator

Using pseudo-spectral derivatives and Bloch shift:

* Gradient: (\nabla H_z \leftrightarrow i (k+G) H_G).
* Divergence: (\nabla\cdot \mathbf v \leftrightarrow i(k+G)\cdot V_G).

Algorithm for (A H_z = -\nabla\cdot \varepsilon^{-1}\nabla H_z):

1. Start with H in real space.

2. FFT → H_G.

3. Compute components of gradient in Fourier space:

   [
   \widehat{\partial_x H} = i (k_x + G_x) H_G, \quad
   \widehat{\partial_y H} = i (k_y + G_y) H_G.
   ]

4. iFFT → get (\partial_x H, \partial_y H) in real space.

5. Multiply each component by ε⁻¹(x,y): (f_x = ε^{-1} \partial_x H), (f_y = ε^{-1} \partial_y H).

6. FFT f_x, f_y.

7. Compute divergence symbolically:

   [
   \widehat{\nabla\cdot f} = i (k_x + G_x) \hat f_x + i(k_y + G_y) \hat f_y.
   ]

8. iFFT back if needed; or stay in Fourier space depending on how you integrate with B.

Then

[
A H_z = -\nabla\cdot(\varepsilon^{-1}\nabla H_z).
]

#### 2D TE (E_z) operator

For TE with B = ε:

* A can simply be −∇² (since μ=1 and curl-curl on a scalar out-of-plane field becomes Laplacian in 2D).
* Implementation:

  * FFT E,
  * Multiply by (|k+G|^2),
  * iFFT.

No ε in A; ε appears in B (mass matrix) via inner products and residuals.

#### CPU vs GPU

* All these steps are **FFT + pointwise multiplies + simple linear combinations**:

  * The **heavy part**: FFT (O(N log N)) → perfect for GPU (cuFFT / wgpu).
  * Pointwise multiplications (ε⁻¹, k+G, etc.) → embarrassingly parallel; also perfect for GPU.

* CPU:

  * Totally fine for moderate grids (2D).
  * Use `rustfft` + `rayon` for multi-threaded FFTs or batched k-points.

This is the **core performance hotspot**: in a GPU version this is where most of the time goes.

---

### Step 9 — Preconditioning (spectral, diagonal in Fourier space)

> “Preconditioning: solves (ε/μ) diagonal in Fourier space to speed convergence.”

MPB uses a **Fourier-space diagonal preconditioner** derived from an approximate symbol of A:

* Replace spatially varying ε⁻¹(r) by a simpler “average” or smooth approximation.
* In Fourier space the operator becomes **diagonal**:

  * For TM: symbol ≈ (|k+G|^2 / \varepsilon_{\text{eff}}).
  * For TE: symbol ≈ (|k+G|^2) or adjusted by μ.

Then preconditioner (M^{-1}) acts as:

[
(M^{-1} v)*G \approx \frac{1}{|k+G|^2 / \varepsilon*{\text{eff}} + \sigma} v_G,
]

where σ is a small shift to avoid division by zero (especially near Γ).

**Algorithm**

To apply (z = M^{-1} r):

1. FFT r → r_G.
2. For each Fourier mode (ix,iy):

   * Compute diag entry

     [
     d_{G} = |k+G|^2 / \varepsilon_{\text{eff}} + \sigma.
     ]

   * Compute `z_G = r_G / d_G`.
3. iFFT z_G → z.

The key idea: this is a **cheap approximate inverse of A** that captures the dominant |k+G|² behaviour and removes stiffness from the Krylov iterations.

You can choose ε_eff as:

* volume average of ε(r),
* or harmonic average (depends on polarization).

**CPU vs GPU**

* Again: FFT + pointwise multiplies → GPU gold.
* On CPU, use same backend as operator; cost similar to one operator application.

**Why “Fourier space”?**

Because with constant ε_eff the operator is diagonal in the plane-wave basis; inversion is trivial there, but not in real space.

---

### Step 10 — Eigenpair assembly & cleanup

> “Eigen-pair assembly and clean-up”

Once block-LOBPCG converges:

1. Have approximate eigenvectors X and eigenvalues Λ.

2. B-orthonormalize X one last time.

3. Sort by λ, compute ω = c√λ or frequencies in units of (2πc/a) etc.

4. Store:

   * for band structure: pair (k, {ωₙ(k)}),
   * optionally eigenvectors in real or Fourier space for analysis.

5. Clean up:

   * free workspaces & temporary buffers,
   * keep Y (deflation subspace) if you want to warm-start neighbouring k.

**CPU vs GPU**

* Sorting, storing, small orthogonalization → CPU tasks.
* You might copy eigenvectors from GPU to CPU only at the end (for each k) if you don’t need them further on device.

---

## 2. CPU vs GPU overview (where each shines)

**CPU serial**

* Geometry → ε(x,y) generation.
* Lattice & k-path construction.
* Small dense ops:

  * (Y^* B Y) inverse,
  * small subspace eigenproblems in LOBPCG,
  * final sorting / bookkeeping.
* Useful for WASM builds (no GPU).

**CPU parallel (threads via rayon)**

* For modest 2D problems, CPU is fine:

  * Parallel over k-points.
  * Parallel over multiple vectors in block (X columns).
  * Multi-threaded FFT (if you implement batched FFTs with rayon yourself; `rustfft` is single-threaded but you can parallelize across batches or across polarizations).

**GPU parallel**

Main speedups:

* FFTs:

  * batched 2D FFT for each field component and each block/vector.
* Maxwell operator:

  * gradient/divergence in Fourier space (multiply by k+G).
  * pointwise multiplication by ε⁻¹ in real space.
* Preconditioner:

  * FFT → scale → iFFT.
* Block ops:

  * computing B X, Y^* B X etc. as large GEMM-like operations.

On GPU you want to:

* Keep ε, k+G, all field vectors X, R, P *resident* on device.
* Only copy back converged eigenvectors / eigenvalues.
* Do deflation and parity as simple kernels on device (or fused into existing kernels).

---

## 3. Minimal “must-have” list for `mpb2d-lite`

From all of this, if I had to mark *critical* things:

1. **Generalized eigenproblem** (A u = \lambda B u) with:

   * TM: (A = -\nabla\cdot\varepsilon^{-1}\nabla,\ B=I).
   * TE: (A = -\nabla^2,\ B=\varepsilon).
2. **FFT-based operator application** (Step 8).
3. **Fourier-space diagonal preconditioner** (Step 9).
4. **B-orthogonal deflation** for:

   * constant mode at Γ,
   * previously converged bands.
5. **Block eigensolver** minimizing Rayleigh quotient in B-inner product (e.g. block-LOBPCG).
6. Optional but highly beneficial:

   * **Parity projectors** when symmetry is present (Step 6).
   * Warm-start from neighbouring k (store previous k’s eigenvectors as initial X).

If you implement *just* those pieces on CPU first, your solver will already feel “MPB-like” in behaviour and performance. Then the GPU port is essentially:

* replace `rustfft` + loops by cuFFT + CUDA kernels (or wgpu shaders),
* keep **all the algorithmic structure identical**.

A dev reading this should be able to sketch the full pipeline:

1. Build ε(x,y) on grid from lattice+geometry.
2. For each k:

   * build k+G arrays,
   * set up deflation subspace (include constant mode at Γ),
   * run block-LOBPCG with projector (deflation + parity), Maxwell operator, preconditioner.
   * assemble eigenpairs and store bands.

---

## Implementation Checklist (mpb2d-lite vs MPB)

### Critical pipeline (Steps 0–10)

* [x] **Step 0 — Enforce the generalized eigenproblem.** `ThetaOperator` now separates A vs. B (TE keeps ε in `apply_mass`) and the Lanczos/power iterations use B-normalization, B-inner products, and B-orthogonal reorthogonalization so Rayleigh quotients/residuals match MPB’s generalized Hermitian structure.
* [x] **Step 1 — Γ-point detection.** `EigenOptions.gamma` now exposes an `enabled + tolerance` pair, `bandstructure::run` computes `||bloch||` for each k-point, and every solver call receives a `GammaContext` so Γ awareness is automatic and configurable.
* [x] **Step 4 — Γ constant-mode deflation.** `solve_lowest_eigenpairs` constructs the B-normalized constant vector when `GammaContext::is_gamma`, reorthogonalizes every search/residual vector against it, and marks the metadata flag so logging/tests can tell the trivial λ=0 band was removed.
* [ ] **Step 5 — General B-orthogonal deflation.** Maintain a block `Y` of converged eigenvectors, keep `Y^T B X = 0` via projection, and optionally recycle bands from neighbouring k-points. Nothing in `crates/core/src/eigensolver.rs` stores or reuses converged vectors yet.
* [ ] **Step 6 — Parity / symmetry projectors.** Implement the MPB-style `(I ± S)/2` projectors for mirror symmetries and gate them with config toggles so benchmarking stays modular; search-space symmetry is currently unchecked.
* [ ] **Step 7 — Block LOBPCG with Rayleigh–Ritz.** Replace the single-vector Lanczos chain with a block solver that forms residuals, applies the preconditioner, and solves the dense subspace problem in the B-inner product. This is the prerequisite for meaningful deflation and preconditioning.
* [ ] **Step 8 — Operator validation.** The FFT-based Θₖ implementations exist (`crates/core/src/operator.rs`), but we still need analytic tests (e.g., compare against the toy Laplacian, MPB references, and Hermitian symmetry checks) to ensure TM gradients/divergence and TE mass matrices are correct for all lattices.
* [ ] **Step 9 — Fourier-diagonal preconditioner.** Implement MPB’s |k+G|²/ε_eff diagonal preconditioner (with small σ shift) and expose it as another `PreconditionerKind`, alongside the existing real-space Jacobi scaler. This should finally help TM convergence.
* [ ] **Step 10 — Eigenpair cleanup & reporting.** Expand the current per-k logging to include residual norms, Rayleigh quotients, and explicit sorting/normalization diagnostics so we can trust the reported ω even when convergence is shaky.
* [x] **Step 2 — Workspace reset.** `ThetaOperator::new` allocates fresh scratch/gradient buffers per k-point, and the Lanczos loop zeroes/normalizes working vectors before use.
* [ ] **Step 3 — k+G bookkeeping (partial).** `build_k_vector` precomputes the FFT frequencies, but (k+G) magnitudes and reciprocals are reassembled inside tight loops every iteration. Cache `k_plus_G` and `|k+G|²` once per k so the operator and preconditioner share the data.
* [x] **Step 8 foundation — FFT-based Θₖ apply.** The TM pseudo-spectral gradient/divergence cycle and TE Laplacian path already mirror MPB’s pattern and reuse backend FFT plans.
* [ ] **Step 9 foundation — Toggleable Jacobi preconditioner (partial).** The config-driven `PreconditionerKind` switch exists, but it only scales by ε in real space; keep it as a baseline once Fourier-diagonal variants land.

### Search-space control & toggles

* [ ] **Warm-start / subspace recycling.** Use converged bands from kᵢ as the initial block for kᵢ₊₁ (after parity/deflation), controlled via config so we can benchmark warm vs cold starts. Today every solve seeds with `seed_vector`.
* [ ] **Feature toggle registry.** Extend `EigenOptions` (and the CLI TOML schema) with switches for Γ handling, deflation, parity, and warm-starts so each optimization can be toggled independently like the preconditioner.
* [ ] **Residual monitoring.** Surface per-iteration residual norms and stagnation detection (maybe through metrics) to explain non-convergence; current logs only state the final iteration count per k-point.

### Validation & reference tracking

* [ ] **MPB regression comparisons.** Use the `python/reference-data` helpers to harvest MPB TE/TM bands and add tests (e.g., `*_tracks_uniform_reference_data`) that assert our spectra stay within tolerance.
* [ ] **Unit / property tests per module.** The `_tests_*.rs` scaffolding exists, but the eigensolver/operator modules still lack cases covering Hermitian symmetry, Γ handling, and parity projections.
* [ ] **Benchmarks for solver loops.** Criterion benches cover FFT prep, but we still need ones that isolate the Lanczos/LOBPCG loop (with and without preconditioning) so we can quantify gains from each toggle.

### Roadmap phases (PROJECT_IMPLEMENTATION.md)

* [x] **Phase 0 – Skeleton & config.** Workspace, config parsing, and CLI stubs are finished.
* [x] **Phase 1 – CPU backend + toy operator.** `mpb2d-backend-cpu` (rustfft + optional Rayon) and the toy Laplacian operator compile and are used in tests.
* [x] **Phase 2 – TE/TM Θₖ operator + Lanczos.** The pseudo-spectral TM/TE operator plus the current Lanczos solver exist, but they still need the generalized B-formulation noted above.
* [x] **Phase 3 – Band-structure pipeline.** `BandStructureJob`/`run_with_metrics` orchestrate dielectric sampling, Θₖ instantiation, and logging per k-point.
* [x] **Phase 3.5 – Pipeline observability.** Verbose CLI logs, k-point iteration summaries, and `Verbosity::Quiet` satisfy the logging goals (quiet mode already forwards to stdout only).
* [ ] **Phase 3.9 – Testing & optimization (partial).** Module-specific `_tests_*.rs` files exist, but regression suites and deterministic solver fixtures are still outstanding.
* [ ] **Phase 3.99 – Instrumentation & benchmarks (partial).** Metrics logging and Criterion FFT/eigensolver benches landed, yet we still need solver-specific guardrails (residual plots, warm-start stats) before touching GPU work.
* [ ] **Phase 4 – 2-atomic basis & oblique lattice validation.** Geometry supports arbitrary atoms, but we still need configs/tests demonstrating 2-atomic hex lattices and overlap edge cases.
* [ ] **Phase 5 – WASM build + web viewer.** No wasm32 target or bindings exist yet.
* [ ] **Phase 6 – GPU backend (CUDA/WebGPU).** Backend traits are ready, but there is no device implementation or cuFFT integration yet.

### Foundations already in place

* [x] **Uniform grid + dielectric sampling.** `Grid2D`, `Geometry2D`, and `Dielectric2D::from_geometry` already generate ε(x,y) on a uniform mesh (air holes + wraps).
* [x] **Spectral backend abstraction.** `SpectralBackend` / `Field2D` encapsulate FFT + BLAS-lite routines, paving the way for future CUDA/WebGPU ports.
* [x] **CLI + config plumbing.** The `mpb2d-lite` CLI parses TOML configs, supports preset k-path overrides, toggles verbosity, and writes CSV/metrics streams.
* [x] **Metrics + logging.** `bandstructure::run_with_metrics` emits setup/dielectric/FFT/k-point timing events, and the CLI exposes a `--quiet` mode for automation.
