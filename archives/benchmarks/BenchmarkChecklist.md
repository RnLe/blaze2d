# Blaze2D Benchmark Suite

**Goal**: Rigorous performance comparison between Blaze2D (Rust) and MPB (Python/Scheme)

## Test Configurations (Joannopoulos 1997)

| Config | Lattice | Background | Rod ε | Radius | Polarization |
|--------|---------|------------|-------|--------|--------------|
| A | Square | Air (ε=1) | 8.9 | 0.2a | TM, TE |
| B | Hexagonal | ε=13 | Air (ε=1) | 0.48a | TM, TE |

**Common parameters**: 12 bands, 20 k-points per segment, 32×32 resolution

## Fairness Considerations

| Parameter | MPB | Blaze2D | Notes |
|-----------|-----|---------|-------|
| Tolerance | 1e-7 (f64) | 1e-4 (mixed-precision f32) | Different due to precision |
| Resolution | 32×32 | 32×32 | ✓ |
| Bands | 12 | 12 | ✓ |
| K-points | 61 (20/segment) | 61 (20/segment) | ✓ |
| Threading | Single/Multi | Single/Multi | Controlled via env vars |

---

## Benchmark Checklist

### 1. Speed Comparison

#### Single-Core (fair comparison)
- [ ] MPB single-core (OMP_NUM_THREADS=1)
- [ ] Blaze2D single-core (-j 1 via bulk driver)
- [ ] Config A TM/TE + Config B TM/TE
- [ ] 10 jobs × 10 iterations each

#### Multi-Core (16 threads)
- [ ] MPB multi-core (all available)
- [ ] Blaze2D multi-core (bulk driver, -j 16)
- [ ] Config A TM/TE + Config B TM/TE
- [ ] 100 jobs × 10 iterations each

#### Analysis
- [ ] Compute speedup with error bars
- [ ] Generate comparison plots
- [ ] Summary report

### 2. Scaling Analysis (TODO)
- [ ] Resolution 16×16
- [ ] Resolution 32×32
- [ ] Resolution 64×64
- [ ] Resolution 128×128
- [ ] Resolution 256×256
- [ ] Generate scaling plots

### 3. Memory Usage (TODO)
- [ ] Peak memory measurement (MPB)
- [ ] Peak memory measurement (Blaze2D)
- [ ] Memory vs resolution scaling

### 4. Multi-core Performance (TODO)
- [ ] Single-thread baseline
- [ ] 2, 4, 8, 16 thread scaling
- [ ] Parallel efficiency analysis

---

## Benchmark Matrix

| Run | Solver | Config | Pol | Cores | Jobs | Iterations |
|-----|--------|--------|-----|-------|------|------------|
| 1 | MPB | A | TM | 1 | 100 | 10 |
| 2 | MPB | A | TE | 1 | 100 | 10 |
| 3 | MPB | B | TM | 1 | 100 | 10 |
| 4 | MPB | B | TE | 1 | 100 | 10 |
| 5 | MPB | A | TM | 16 | 100 | 10 |
| 6 | MPB | A | TE | 16 | 100 | 10 |
| 7 | MPB | B | TM | 16 | 100 | 10 |
| 8 | MPB | B | TE | 16 | 100 | 10 |
| 9 | Blaze2D | A | TM | 1 | 100 | 10 |
| 10 | Blaze2D | A | TE | 1 | 100 | 10 |
| 11 | Blaze2D | B | TM | 1 | 100 | 10 |
| 12 | Blaze2D | B | TE | 1 | 100 | 10 |
| 13 | Blaze2D | A | TM | 16 | 100 | 10 |
| 14 | Blaze2D | A | TE | 16 | 100 | 10 |
| 15 | Blaze2D | B | TM | 16 | 100 | 10 |
| 16 | Blaze2D | B | TE | 16 | 100 | 10 |

**Total**: 16 distinct runs, 1,000 samples each = 16,000 band structure calculations

---

## Results Summary

| Benchmark | Status | Speedup | Notes |
|-----------|--------|---------|-------|
| Speed (single-core) | - | - | |
| Speed (multi-core) | - | - | |
| Scaling | - | - | |
| Memory | - | - | |

## Optimization Opportunities (Identified via Audit)

### High Priority
- [x] **Batched Operator Application**: Fixed in `Eigensolver` methods:
    - `compute_aq_block`: Uses `operator.batch_apply`.
    - `precondition_residuals`: Uses `preconditioner.batch_apply`.
    - `orthonormalize_subspace`: Uses `operator.batch_apply_mass`.
- [x] **Symmetry-based Reduction**: Implement "Parity Projector" to decompose the problem into even/odd sectors, effectively reducing the search space size and ensuring correct mode classification.

### Medium Priority
- [ ] **Batched Orthogonalization**: Ensure SVQB uses batched BLAS Level 3 operations (GEMM) for inner products instead of looped dot products.

