Alright, let's dig into this.

From what you showed + the code in `eigensolver.rs`, `operator.rs`, `preconditioner.rs` and the MPB notes, the picture is:

* The operator and ε(r) look sane.
* The Fourier preconditioner *does* reduce raw residuals (your snapshots confirm that).
* But the LOBPCG loop stalls:
  residuals stay at **10³–10⁴**, zig-zag ratio ≈ 0.4–0.55, and you always hit `max_iter = 101`.

That’s exactly the signature of:

> “Preconditioner helps locally, but Rayleigh–Ritz + constraints keep throwing me back into a subspace where the Rayleigh quotient can’t improve.”

Below I’ll go through:

1. What the report actually says numerically.
2. Concrete code-level suspects (including one real bug in the projection/deflation).
3. How the constraint machinery (Γ-mode + symmetry + deflation) can kill convergence.
4. A prioritized set of experiments / patches to try.

---

## 1. What your report is actually telling us

For the TE run at Γ (`square_te_eps13_r0p3_res24_k6_b8`):

* All k’s hit **101 iterations** and stop with

  * final residual ∼ **10³–10⁴**
  * `rel ≈ 2.6–3.0`
  * `scale ≈ 10³` (from your iteration trace summary)
  * **no thresholds hit**: the solver *never* sees `rel ≤ tol` or `residual ≤ 1e-8`.
* `zig-zag ratio` ≈ 0.44–0.55 for all k:

  * About half the iterations *increase* the residual ceiling.
  * So LOBPCG is essentially wandering around instead of homing in.
* Residual snapshots:

  * `raw` max |r| ~ 1.5×10²
  * `projected` ~ similar
  * `preconditioned` ~ O(10¹–10²), so the **preconditioner does reduce the residual magnitude** (your MPB notes say 1–2 orders of magnitude in some configs).
* TE/TM operator snapshots show |Θu| in the **O(1–10)** range – nothing insane.

So: preconditioner is *not* completely broken; it just doesn’t translate into steadily decreasing Rayleigh residuals inside the LOBPCG loop.

---

## 2. Concrete code-level suspects

### 2.1. Inner-product mismatch in mass-orthogonal projections (real bug)

You consistently define the B-inner product as

[
\langle x, y \rangle_B := x^* (B y)
]

and implement it as:

```rust
fn mass_norm<B: SpectralBackend>(backend: &B, vector: &B::Buffer, mass_vec: &B::Buffer) -> f64 {
    backend.dot(vector, mass_vec).re.max(0.0).sqrt()
}
```

i.e. `dot(vector, mass_vec)` = `x^* (B y)` with `y=x` for norms. 

But **your projections use the reversed order**:

```rust
fn reorthogonalize_with_mass<B: SpectralBackend>(
    backend: &B,
    target: &mut B::Buffer,
    basis: &B::Buffer,
    mass_basis: &B::Buffer,
) {
    let coeff = backend.dot(mass_basis, target); // <--- HERE
    backend.axpy(-coeff, basis, target);
}
```

and in deflation: 

```rust
pub fn project(&self, backend: &B, vector: &mut B::Buffer, mass_vector: &mut B::Buffer) {
    for entry in &self.entries {
        let coeff = backend.dot(&entry.mass_vector, vector); // <--- HERE
        backend.axpy(-coeff, &entry.vector, vector);
        backend.axpy(-coeff, &entry.mass_vector, mass_vector);
    }
}
```

Mathematically, what you *want* is:

[
\alpha = \langle x, v \rangle_B = x^* (B v)
]

which corresponds to `dot(target, mass_basis)` with your dot convention, *not* `dot(mass_basis, target)`.

For *real* fields, this difference is harmless because
`dot(mass_basis, target) = conj(dot(target, mass_basis))` and everything stays real.
But for **complex fields (generic k)** this is genuinely wrong:

* You are using ⟨v, x⟩₍ᴮ₎ instead of ⟨x, v⟩₍ᴮ₎, i.e. the complex conjugate.
* The projection step no longer guarantees ⟨x_new, v⟩₍ᴮ₎ = 0; you only kill the real part and leave an imaginary component.
* Over many iterations, this breaks B-orthogonality of:

  * deflation subspace,
  * Γ-mode constraint,
  * warm-start modes,
  * and the block basis in LOBPCG.

That is exactly the kind of thing that shows up as:

* “preconditioner works locally, but Rayleigh–Ritz fights it,”
* residual zig-zag ratios near 0.5,
* block subspaces acquiring nearly linearly dependent directions and poor conditioning in the projected mass matrix.

**Patch (minimal, and I would definitely do it):**

* In `reorthogonalize_with_mass`:

```rust
let coeff = backend.dot(target, mass_basis); // instead of dot(mass_basis, target)
```

* In `DeflationWorkspace::project`:

```rust
let coeff = backend.dot(vector, &entry.mass_vector); // instead of dot(&entry.mass_vector, vector)
```

This aligns all your B-orthogonalization with the same inner-product convention used in `mass_norm`, `normalize_with_mass`, and `build_projected_matrices`.

Even though your current problematic run is at Γ (fields are mostly real), this bug *will* bite you at nonzero k. I’d fix it now; it’s small and conceptually necessary.

---

### 2.2. Convergence metric is still extremely strict

Your relative residual is (correctly) defined as

```rust
scale = max( |λ| * ||x||_B , ||Θx||_2 )
rel   = ||r||_B / scale;
```

with an absolute guard at `1e-8`.

For low bands in a high-index structure:

* λ ≈ ω² is **small** (say 0.05–0.3).
* ‖x‖_B is ≈ 1 after normalization.
* ‖Θx‖_2 is O(10–30) from your snapshots.

So the denominator `scale` is O(10–30) for the *first* bands, **not** O(10³.**).
Yet your trace reports `scale ≈ 10³` and `residual ≈ 2.4×10³`, giving `rel ≈ 2.6`. That suggests:

* either the scale is dominated by much higher bands (large λ, large Θx), or
* `entry.applied` for the first bands is large enough that `||Θx||` is ~10³ (unexpected given snapshots), or
* something in your Python aggregation mismatches bands vs. rows.

Even if we assume the code is now correct, with `tol = 1e-6` you are asking for:

[
|r|_B \lesssim 10^{-6} \cdot \max(|λ| |x|_B, |\Theta x|)
]

so in practice you want B-norm residuals around 10⁻⁵–10⁻⁶ relative to operator norm. That’s *much* stricter than what MPB itself uses for band-diagram work (you usually care about ~1e-7 in **frequency**, not necessarily in Rayleigh residual).

So there is a dual effect here:

1. Residual **plateaus** around O(10³) because something in the iteration is off.
2. Your **tolerance** is so strict that even a healthy solver might not get there for the lowest bands without more aggressive preconditioning.

For diagnosis, I would *definitely* do one “sanity check run” with:

```toml
[eigensolver]
tol = 1e-3
max_iter = 200
preconditioner = "structured"
symmetry = { disable_auto = true }
gamma = { enabled = true }
deflation = { enabled = false }
```

If residual now **drops monotonically** and stabilizes around `rel ~ 1e-3` in < 50 iterations, then:

* The core LOBPCG machinery is basically fine.
* What is killing you at `tol=1e-6` is either:

  * the strict metric, or
  * the constraint stack (see next section) + square Γ peculiarity.

---

### 2.3. Preconditioner is strong but can “fight” constraints

From `preconditioner.rs` and MPB notes:

* You construct a **Fourier-diagonal TE/TM preconditioner** using:

  * a clamped `|k+G|²` (`K_PLUS_G_NEAR_ZERO_FLOOR`),
  * an ε-aware structured weight,
  * a small shift.
* Tests enforce that for a high-frequency plane wave, the preconditioner reduces the residual by ≥10×.

The residual snapshots confirm that: max |r| shrinks from ~O(10²) to O(10¹) after preconditioning.

Then you do:

```rust
operator.apply_mass(&vector, &mut mass);
enforce_constraints(... gamma, deflation, symmetry);
project_against_entries(... x,p,w);
```

**twice**, once before and once after the preconditioner. 

The second pass is necessary because M⁻¹ does not preserve:

* symmetry irreps,
* Γ-constant deflation,
* B-orthogonality.

But the combination:

* preconditioner built in Fourier (no knowledge of symmetry or deflation),
* then hard projection back into a very thin subspace (square Γ, with parity + Γ-mode deflation),
* plus B-orthogonalization against X/P/W,

means it’s entirely possible that:

> the “good” components created by the preconditioner lie mostly in directions that are *then projected away* by symmetry + deflation + Gram–Schmidt.

That gives exactly the pattern:

* preconditioner trials not zero,
* preconditioned residual amplitude smaller *before projection*,
* but after Rayleigh–Ritz, residual ceiling doesn’t move much, and zig-zag ~0.5.

You have some notes to this effect in `MPB_PIPELINE.md`, especially about **symmetry projectors staying on at Γ** and killing viable directions. 

Now that you also deflate Γ mode and carry a deflation workspace, the risk of over-projecting is even larger.

---

## 3. Constraint stack: Γ-mode + symmetry + deflation

For this particular run:

* Γ point (`Bloch |k| = 0`).
* `GammaHandling.enabled = true` and tolerance 1e-10 ⇒ you **build and deflate the constant mode**. 
* You also have:

  * auto-inferred symmetry reflections (for square lattice),
  * potentially manual symmetry,
  * a per-solve `DeflationWorkspace` (warm-start or previously converged modes). 

The enforcement pipeline looks like:

```rust
// inside enforce_constraints
symmetry.apply(vector);
apply_mass(...);

gamma_mode.reorthogonalize_with_mass(...);
apply_mass(...);

deflation.project(backend, &mut vector, &mut mass);
```

followed by projections against X, P, W. 

For Γ:

1. **Symmetry**: restricts you to a specific parity subspace (or two) of the full physical eigenspace.
2. **Γ-mode**: removes the constant eigenvector (λ=0) from that subspace.
3. **Deflation** (within the same k): removes modes already converged in this LOBPCG run.

That’s a *lot* of projections stacked on top of each other. If these subspaces get even mildly misaligned with the numerically best preconditioned directions (for example because of the inner-product bug above), LOBPCG can very easily end up in a small, badly conditioned corner of the space where Rayleigh–Ritz barely improves anything.

This is why, diagnostically, you should:

* Run with **symmetry disabled** (`--no-auto-symmetry`, or `SymmetryOptions::disable_auto()`).
* Or at least with `deflation.enabled = false` for a test solve.
* And see whether the plateau disappears.

If turning off symmetry+deflation suddenly makes the residual behave as in your hexagonal runs, then the *algorithmic* core is fine and the culprit is the constraint stack.

---

## 4. Suggested experiments / patches (in order)

Here’s how I’d systematically attack this, without changing physics:

### Step 1 – Fix the B-inner-product bug

This is a clean mathematical bug, and it’s small.

* Change:

```rust
let coeff = backend.dot(mass_basis, target);
```

to

```rust
let coeff = backend.dot(target, mass_basis);
```

both in `reorthogonalize_with_mass` and in `DeflationWorkspace::project`.

Rebuild, rerun `square_*_eps13_r0p3_res24_k6_b8`, and re-check:

* `iterations`,
* `max_residual` vs. iteration,
* zig-zag ratio.

Even for Γ (mostly real fields) this will at least make the entire projection logic internally consistent.

---

### Step 2 – Strip down constraints and see if solver is intrinsically OK

Run the same square TE/TM job with:

* `deflation.enabled = false` (or `opts.debug.disable_deflation = true`),
* `gamma.enabled = true` (you still want to deflate the constant),
* `--no-auto-symmetry` so the `SymmetryProjector` is `None` for that solve, or explicitly pass `symmetry_override = None`.

If with those settings:

* the residual suddenly *decreases monotonically* and reaches `rel ~ 1e-3` in, say, 20–30 iterations,
* and zig-zag ratio drops well below 0.5,

then the core LOBPCG implementation is fine, and the “killer” is the combination of:

* symmetry projector,
* per-k deflation workspace,
* structured preconditioner.

You can then reintroduce them one by one:

1. add symmetry back, keep deflation off;
2. add deflation back, keep symmetry off;
3. then combine both once you know each is safe.

---

### Step 3 – Try a “naive” preconditioner as a control

Set `preconditioner = "homogeneous"` (or `None`) in `EigenOptions` and rerun the same configuration.

* If **no preconditioner** gives *better* convergence behaviour (even if slower), the Fourier diagonal is probably too aggressive in directions that are then projected away.
* If homogeneous Jacobi behaves fine, but structured diagonal does not, I’d inspect:

  * how you blend `|k+G|²` and ε(x,y),
  * whether you are over-amplifying the clamped bin at `|k+G|² ≈ floor` at Γ.

This would also show up in the preconditioner diagnostics you already added (average norms before/after, directions accepted). 

---

### Step 4 – Relax tolerance purely for debugging

Temporarily use `tol = 1e-3` (relative) and `max_iter = 200`.

* If residual curves now behave sensibly (decay to ~1e-2 or ~1e-3, then plateau), the algorithm is okay; you’re just pushing it into a regime where the diagonal preconditioner plus projections can’t achieve 1e-6.
* If even at tol=1e-3 you still see flat ~10³ residuals and zig-zag ~0.5, then there’s still something more fundamental (e.g. deflation workspace, Rayleigh–Ritz, or operator).

---

### Step 5 – Double-check Rayleigh–Ritz consistency

I didn’t see a blatant bug in `build_projected_matrices` / `generalized_eigen` / `combine_entries`, but a few things are worth sanity-testing:

* Verify that for a **tiny synthetic problem** (e.g. 3×3 SPD A,B) your `generalized_eigen` returns the same eigenpairs as a reference solver (you already have some tests, but I’d add one where B is far from identity).
* Confirm in a log:

  * The eigenvalues from `rayleigh_ritz` decrease monotonically over iterations for at least one k-point when you use a diagonal A,B test (where preconditioner is identity and constraints disabled).
* Check that `mass_proj` is indeed SPD in practice (no `cholesky_decompose` failures, no NaNs) – your code already returns `None` when Cholesky fails; make sure that path is not silently taken.

---

## 5. What I’d bet on right now

Given:

* the **known** B-inner-product bug in deflation/Γ reorthogonalization,
* the **heavy constraint stack** at Γ (symmetry + Γ-mode + deflation),
* the **good behaviour of the preconditioner** in isolation,
* and the **zig-zag ratios ~ 0.5**,

my best guess is:

1. The inner-product inconsistency is slowly degrading the B-orthogonality of both the deflation workspace and the active block, especially when warm starts are involved.
2. At Γ, the combination of:

   * symmetry projector into a tight irrep,
   * Γ-mode deflation,
   * plus slightly mis-orthogonal deflation vectors
     leaves LOBPCG searching in a narrow, badly conditioned subspace where Rayleigh–Ritz can’t reduce the residual further.
3. The strict relative tolerance (`1e-6`) makes that plateau look like “no convergence at all” even though your physics might already be okay within e.g. 1e-3 relative error.

So: fix the inner-product, temporarily simplify constraints and loosen `tol`, and see if the residual curves start making sense. If they do, you can then gradually re-enable the “nice” features (symmetry, deflation, tight tol) one by one until you see what reintroduces the stall.
