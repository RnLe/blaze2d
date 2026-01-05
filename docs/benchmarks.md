# Benchmarks

## Methodology

All benchmarks compare Blaze2D against [MIT MPB](https://github.com/NanoComp/mpb) under matched conditions to ensure fairness.

### Test Configurations (Joannopoulos 1997)

We utilize standard test cases from the photonic crystal literature:

| Config | Lattice | Background | Rod ε | Radius | Polarization |
|--------|---------|------------|-------|--------|--------------|
| **A** | Square | Air (ε=1) | 8.9 | 0.2a | TM, TE |
| **B** | Hexagonal | ε=13 | Air (ε=1) | 0.48a | TM, TE |

**Common Parameters:**
- **Bands**: 12
- **K-points**: 61 (20 per segment)
- **Resolution**: 32×32 (unless scaling is being tested)
- **Tolerance**: 1e-7

### Fairness Considerations

To ensure a rigorous comparison:
- **Tolerance**: Both solvers are strictly set to `1e-7`.
- **Parallelism**: MPB is run using Python `multiprocessing` to match Blaze2D's job-level parallelism, avoiding the known overhead of MPB's internal OpenMP threading on small grids.

---

## 1. Speed Comparison

This benchmark measures the raw execution time for band structure calculations.

### Single-Core Performance
- **Objective**: Measure the baseline efficiency of the core solvers without parallel overhead.
- **Setup**: `OMP_NUM_THREADS=1` for MPB; single-threaded bulk driver for Blaze2D.
- **Workload**: 100 jobs per configuration (A/B, TM/TE).

### Multi-Core Throughput
- **Objective**: Evaluate scaling efficiency in a high-throughput parameter sweep scenario.
- **Setup**: 16 concurrent workers (matching the host CPU core count).
- **Workload**: 100 jobs per configuration, distributed across workers.

*Results pending.*

---

## 2. Scaling Analysis

This benchmark evaluates how solver performance scales with grid resolution.

- **Resolutions**: 16×16, 32×32, 64×64, 128×128, 256×256.
- **Objective**: Determine the crossover point where algorithmic differences (e.g., FFT overhead vs. dense matrix ops) become dominant.

*Results pending.*

---

## 3. Memory Usage

This benchmark tracks the peak resident set size (RSS) during execution.

- **Objective**: Compare the memory footprint of the Rust-based architecture vs. the Python/Scheme/C++ stack of MPB.
- **Metric**: Peak MB per worker thread.

*Results pending.*

---

## 4. Parallel Efficiency

This benchmark analyzes the speedup factor as the number of threads increases.

- **Threads**: 1, 2, 4, 8, 16.
- **Objective**: Quantify the overhead of the job distribution system and identify any bottlenecks in the bulk driver.

*Results pending.*

---

## Reproducing Benchmarks

The benchmark suite is automated via the `benchmarks/` directory.

```bash
cd benchmarks

# Run the full comparison suite
make bulk-compare
```
