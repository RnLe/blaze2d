# LOBPCG Optimization Checklist

## Profile Summary (64×64, 12 bands, TE mode)

| Component | Time | % Total | Calls |
|-----------|------|---------|-------|
| `apply_te` | 1.075s | 28% | 12,201 |
| `compute_aq_block` | 1.065s | 28% | 350 |
| `orthonormalize_subspace` | 1.055s | 28% | 350 |
| `svqb::gemm` | 602ms | 16% | 350 |
| `fft_2d::inverse` | 623ms | 16% | 49,428 |
| `fft_2d::forward` | 586ms | 15% | 49,428 |
| `update_ritz_vectors` | 415ms | 11% | 350 |
| `svqb::gram_matrix` | 267ms | 7% | 350 |
| `project_operator` | 267ms | 7% | 350 |

---

## 🔴 Critical Bottlenecks

### 1. ~~Batched Operator Applications in `compute_aq_block`~~ ❌ NOT BENEFICIAL
- **File**: `crates/core/src/eigensolver.rs` L1048-1060
- **Tested**: 2026-01-06
- **Finding**: Batched approach is **slower** (TE: +16.4%, TM: +2.5%)
- **Root cause**: CPU backend processes buffers sequentially, not parallel
- **Future**: True Rayon parallelization across vectors could help

### 2. ~~Redundant Vector Cloning in `collect_subspace_with_mass`~~ ❌ NOT BENEFICIAL
- **File**: `crates/core/src/eigensolver.rs` L919-954
- **Tested**: 2026-01-06
- **Finding**: Moving vectors instead of cloning caused **regression** (TE: +4.6%, TM: +0%)
- **Root cause**: Moving vectors causes fresh allocations in `update_ritz_vectors`.
  The clones are expensive, but keeping buffers in-place provides cache locality
  that outweighs the clone cost. New allocations disrupt cache patterns.
- **Attempted**: `take_subspace_for_orthonormalization` draining x_block/w_block
- **svqb::gemm** regressed 24.6% due to allocation churn and cache effects

### 3. ~~Symmetry-based Reduction (Deep Symmetry)~~ ⚠️ MIXED RESULTS
- **Tested**: 2026-01-06
- **Finding**: **Regression** for small band counts (N=12)
  - Full Symmetry (8 sectors): **+35% slowdown** (2.42s -> 3.27s)
  - Simplified (Even/Odd): **+13% slowdown** (2.39s -> 2.70s)
- **Root Cause**: Overhead of managing multiple `Eigensolver` instances and projectors dominates.
  The $O(N^2)$ gain from solving smaller subspaces is negligible for small $N$.
- **Status**: Implemented but **disabled by default**. Recommended only for large-scale calculations ($N \ge 50$).

### 4. Matrix Building Overhead in GEMM Operations
- **File**: `crates/core/src/eigensolver.rs` L1266-1286, `normalization.rs`
- **Issue**: Builds n×r matrices element-by-element 6× per iteration
- **Fix**: Store vectors in contiguous column-major format
- **Expected**: ~5-10% speedup in GEMM-heavy operations

### 5. Unnecessary B*P and B*W in SVQB
- **File**: `crates/core/src/eigensolver.rs` L1006-1010
- **Issue**: Computes 24 fresh mass applications (FFTs) for P and W blocks
- **Fix**: Compute only needed Gram matrix blocks using available B*X
- **Expected**: Save 24 FFTs per iteration

### 6. Deflation Projection is O(k×n) Dot Products
- **File**: `crates/core/src/eigensolver/deflation.rs` L318-335
- **Issue**: Serial dot products instead of batched GEMM
- **Fix**: `C = V^H × BY` then `V -= Y × C^H` (single GEMM)
- **Expected**: Matters as more bands lock (late iterations)

---

## 🟡 Physics Optimizations

### 6. W Block Size Reduction
- **Current**: W has m vectors (same as X)
- **Literature**: m/2 or fewer W directions often sufficient
- **Expected**: ~15% memory reduction, faster SVQB

### 7. Adaptive Subspace Pruning
- **Current**: Always 3m subspace after iter 0
- **Idea**: Drop P directions for nearly-converged bands
- **Expected**: Smaller dense eigenproblems

### 8. Early Termination per K-point
- **Current**: Checks convergence after each iteration
- **Idea**: Skip remaining work once converged (already done ✅)

---

## 🟢 Already Optimized

- ✅ A*X via GEMM reusing A*Q (no fresh operator applies)
- ✅ B*X precomputed and reused in SVQB
- ✅ Eigenvalue-based convergence (faster than residual-based)
- ✅ Lazy residual B-norm computation (skipped on ~90% of iterations)
- ✅ Batched FFT infrastructure exists

---

## Implementation Checklist

### Phase 1: Low-Hanging Fruit
- [x] **1.1** ~~Use batched operator apply in `compute_aq_block`~~ ❌ NOT BENEFICIAL
- [ ] **1.2** Batch deflation projection with GEMM
- [x] **1.3** ~~Reduce unnecessary vector cloning~~ ❌ NOT BENEFICIAL (cache effects)

### Phase 2: Memory Layout
- [ ] **2.1** Store X block as contiguous n×m matrix
- [x] **2.2** ~~Eliminate cloning in `collect_subspace_with_mass`~~ ❌ NOT BENEFICIAL
- [ ] **2.3** Pre-allocate GEMM workspace buffers

### Phase 3: Algorithm Refinements
- [ ] **3.1** Test reduced W block size (m/2)
- [ ] **3.2** Selective Gram matrix computation in SVQB
- [ ] **3.3** Adaptive subspace pruning for converged bands

### Phase 4: Parallelization (Future)
- [ ] **4.1** True parallel operator applications with Rayon
- [ ] **4.2** Parallel GEMM for large matrices
- [ ] **4.3** SIMD optimization for dot products

### Phase 5: Profiling & Validation
- [ ] **5.1** Verify correctness after each change (eigenvalue accuracy)
- [ ] **5.2** Run profiling suite after each phase
- [ ] **5.3** Document performance delta in profile_history.txt

---

## Validation Commands

```bash
# Build with profiling
make build

# Run quick validation
cargo test --release -p blaze2d-core

# Run profiling
make profile

# Compare to baseline
make profile  # Appends to profile_history.txt
```
