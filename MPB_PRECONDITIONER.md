MPB’s preconditioner is **not exotic magic** – it is basically the inverse of the *homogeneous-medium* Maxwell operator in the plane-wave basis, applied to the residual. For 2D TE/TM, you can implement a very close analogue with just a diagonal operator in G-space. Below is how it works in the Johnson–Joannopoulos paper and MPB.

---

## 0. The eigenproblem MPB actually solves

In H–formulation, μ=1, periodic ε(r):

[
\hat A_{\mathbf k},\mathbf H_{\mathbf k} \equiv
\nabla\times\left[\varepsilon^{-1}(\mathbf r),\nabla\times\mathbf H_{\mathbf k}(\mathbf r)\right]
= \left(\frac{\omega^2}{c^2}\right)\mathbf H_{\mathbf k}(\mathbf r)
]

In a plane-wave basis with Bloch wavevector **k**, coefficients (\mathbf h_m) for G-vectors (\mathbf G_m), application of (A_{\mathbf k}) is (their Eq. (10))([MIT Mathematics][1])

[
(A_{\mathbf k} \mathbf h)*\ell
= -(\mathbf k + \mathbf G*\ell)\times
\text{IFFT}\Big[
\varepsilon_g^{-1}(\mathbf r),
\text{FFT}\big[(\mathbf k+\mathbf G_m)\times \mathbf h_m\big]
\Big]
]

so one “matvec” (y = A_{\mathbf k} x) is:

1. Cross with (\mathbf q_m = \mathbf k+\mathbf G_m) in G-space.
2. FFT to real space.
3. Multiply by the *effective* inverse dielectric (\varepsilon_g^{-1}(\mathbf r)) (the smoothed one you already implemented).([MIT Mathematics][1])
4. IFFT back to G-space.
5. Cross again with each (\mathbf q_\ell).

They then apply a **block LOBPCG / block-conjugate-gradient** eigensolver to this Hermitian positive-definite operator with preconditioning.

---

## 1. What MPB wants from the preconditioner

For the generalized eigenproblem (A h = \lambda B h) with (B=I), the ideal preconditioner is (M \approx A^{-1}). In practice, they want:

* Very cheap to apply: ideally **O(N)** or **O(N log N)** like a matvec.
* Captures the high-frequency / stiff part of the spectrum (large (|\mathbf k+\mathbf G|) modes).
* Remains Hermitian and positive definite so CG/LOBPCG theory still applies.

They construct (M^{-1}) from the **homogeneous Maxwell operator** in the same plane-wave basis, i.e. “what A would be if ε were constant”.

For a homogeneous medium with ε₀ and transverse fields:

[
\nabla\times\left(\varepsilon_0^{-1}\nabla\times\mathbf H\right)
= \varepsilon_0^{-1}(-\nabla^2)\mathbf H
;\Rightarrow;
\lambda_m = \varepsilon_0^{-1}|\mathbf k+\mathbf G_m|^2
]

The exact inverse is diagonal in the plane-wave basis:

[
(A_{\text{hom}}^{-1} r)_m
= \frac{\varepsilon_0}{|\mathbf k+\mathbf G_m|^2} , r_m
\quad(\mathbf k+\mathbf G_m\neq 0).
]

MPB’s preconditioners are essentially **variants of this diagonal inverse**, with sensible handling of ε and of the Γ-point singularity.

---

## 2. MPB’s two preconditioners (paper level)

Section 2.4 of Johnson & Joannopoulos describes two preconditioners: a **diagonal/Jacobi** one and a more refined **“transverse” preconditioner**.

### 2.1 Diagonal (Jacobi) preconditioner

Mathematical idea:

* Approximate A by its homogeneous-medium version with some scalar ε̄ (an appropriate average of ε or ε⁻¹).
* In the transverse plane-wave basis (they already rewrote h in the two polarizations orthogonal to (\mathbf q_m)), that operator is diagonal.

For a residual vector r (in the 2N-dimensional transverse plane-wave basis), define:

[
z_m = (M_{\text{diag}}^{-1} r)*m
= \frac{\varepsilon*{\text{eff}}}{|\mathbf k + \mathbf G_m|^2} , r_m,
\quad \mathbf k+\mathbf G_m\neq 0.
]

Key points:

* (\varepsilon_{\text{eff}}) is essentially an average effective dielectric, consistent with the smoothing formula they use for ε_g. In 2D scalar TE/TM, this just reduces to some scalar average over the cell.([MIT Mathematics][1])
* For the Γ point, (\mathbf k=0), (G=0) gives (|\mathbf q|=0) → they **deflate** those constant modes anyway (your step 4); the preconditioner simply sets that component to zero or leaves it untouched, and the eigensolver never asks it to converge.

Algorithmically, the diagonal preconditioner is just:

```text
for each G:
    if |k+G| > 0:
        z(G) = (eps_eff / |k+G|^2) * r(G)
    else:
        z(G) = 0   # or r(G), but that component is deflated anyway
```

Complexity: O(N), trivially parallel (CPU or GPU).

### 2.2 “Transverse” / improved preconditioner

The second preconditioner in the paper refines this by using a **vector Laplacian restricted to transverse fields** (in spirit, “inverse of −∇² in the transverse subspace”), again in the homogeneous limit. It’s still diagonal in the **transverse plane-wave basis**, but you can think of it as

[
M_{\text{T}}^{-1} \approx P_T A_{\text{hom}}^{-1} P_T,
]

where (P_T) is the projector onto fields perpendicular to (\mathbf k+\mathbf G) (but in MPB this is effectively absorbed into the choice of basis vectors (\hat u_m, \hat v_m)).([MIT Mathematics][1])

In practice, for the implementation you care about:

* In **3D vector case**, they store each G-mode as a 3-vector and project out the longitudinal part via:

  [
  \mathbf r^\perp_m = \mathbf r_m - \frac{\mathbf q_m(\mathbf q_m\cdot\mathbf r_m)}{|\mathbf q_m|^2},
  \quad \mathbf q_m = \mathbf k+\mathbf G_m.
  ]

  Then apply the same diagonal factor ε_eff / |q|² to the transverse part.

* In **2D TE/TM**, the field is already scalar and automatically transverse, so the transverse projection step is trivial; you effectively just get the diagonal preconditioner.

So, for your purposes, MPB’s “improved” preconditioner still looks like:

```text
for each G:
    q = k + G
    if |q| == 0:  # Γ null-space
        z(G) = 0
        continue
    # (3D only) project residual onto transverse plane:
    r_perp(G) = r(G) - q * (dot(q, r(G)) / |q|^2)
    z(G) = (eps_eff / |q|^2) * r_perp(G)
```

Again O(N), massively parallel.

---

## 3. How this plugs into your MPB-2D-lite

### 3.1 2D physics (TE/TM)

For a 2D structure invariant in z, with μ=1:

* **TE (Hz)**: scalar eigenproblem

  [
  \nabla\cdot\left(\varepsilon^{-1}(\mathbf r)\nabla H_z\right)
  = -\frac{\omega^2}{c^2} H_z
  ]

  In plane waves:

  [
  A_{GG'} = (\mathbf k+\mathbf G)\cdot(\mathbf k+\mathbf G'),\varepsilon^{-1}_{G-G'}
  ]

  Preconditioner from homogeneous limit:

  [
  (M^{-1} r)*G = \frac{\varepsilon*{\text{eff}}}{|\mathbf k+\mathbf G|^2},r_G,
  \quad G\neq -k.
  ]

* **TM (Ez)**: similarly,

  [
  \nabla\cdot\left(\varepsilon(\mathbf r)\nabla E_z\right)
  = -\frac{\omega^2}{c^2} E_z
  ]

  But if you implement TM in the same FFT-Maxwell style as MPB, you can stick to the “curl ε⁻¹ curl” formulation with different ε_g smoothing; the preconditioner form is the same: diagonal in G with ε_eff replaced appropriately (average of ε instead of ε⁻¹ depending on polarization, consistent with your smoothing).

In other words: your **2D TE/TM preconditioner can be literally:**

```text
eps_eff_TE = <1/eps>_cell^{-1}   # or something consistent with your smoothing
eps_eff_TM = <eps>_cell         # likewise for TM

for each G:
    q = k + G
    if |q| == 0: z(G) = 0
    else:
        z(G) = (eps_eff_pol / |q|^2) * r(G)
```

where “pol” is TE or TM. This is exactly what you get from “homogeneous Maxwell operator inverse”. It is extremely close in spirit to what MPB uses.

### 3.2 Why your eigensolver might be failing

Given that your ε(r)/FFT/operator code “looks good” but convergence is catastrophic, typical culprits:

1. **No preconditioner** or a preconditioner that is effectively identity.
   → Implement the diagonal one above first; it often changes CG from “doesn’t converge at all” to “converges in O(100) iterations”.

2. **Wrong sign / scaling of A.**

   * The operator must be **Hermitian positive definite** for CG/LOBPCG in the form they use.
   * Check that with a homogeneous test case (single ε, simple lattice) you get eigenvalues (\omega^2/c^2 \approx |\mathbf k+\mathbf G|^2 / \varepsilon).

3. **Γ point null space not deflated correctly.**

   * At k=0, constant modes are true zero-eigenvalue modes and must be removed from the Krylov space (deflation), just like your note 4.
   * If they’re not, CG will chase a zero eigenvalue while your target frequency is >0.

4. **TE/TM operator mismatch with eigenproblem.**

   * Make sure your TE/TM operator is exactly the one your physics says (e.g. minus signs, factors of ε or ε⁻¹ in the right places).
   * A quick test: homogeneous slab, compute a few modes and compare to analytical dispersion.

---

## 4. CPU vs GPU notes (for the preconditioner)

* **Diagonal preconditioner (the MPB one above)**

  * CPU: trivially vectorized, O(N).
  * GPU: embarrassingly parallel – a single kernel with 1 thread per G-mode.
  * Memory footprint negligible.

* **More fancy preconditioners (e.g. including some ε(r) structure)**

  * If you tried to do something fancier than diagonal (e.g. approximate A using a few FFTs in the preconditioner itself), cost becomes comparable to a matvec. That’s only worth it if diagonal preconditioning is insufficient.
  * On GPU, such a “FFT-based preconditioner” is still fine, but then each preconditioner application is: 2 FFTs + some pointwise multiplies + scalar factors. On CPU, also okay, but extra overhead.

Given MPB’s experience and your 2D scope, I’d **start with the diagonal homogeneous-medium preconditioner** as above. It is almost exactly what MPB uses conceptually, and it’s the easiest thing to wire into your existing operator + LOBPCG pipeline.