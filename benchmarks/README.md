# Blaze2D Benchmark Suite

Rigorous performance comparison between Blaze2D (Rust) and MPB (Python/Scheme).

## Test Configurations

Based on Joannopoulos et al., Nature 1997:

| Config | Lattice | Background | Rod ε | Radius |
|--------|---------|------------|-------|--------|
| A | Square | Air (ε=1) | 8.9 | 0.2a |
| B | Hexagonal | ε=13 | Air (ε=1) | 0.48a |

**Parameters**: 12 bands, 20 k-points per segment, 32×32 resolution

## Benchmark Matrix

| Solver | Core Mode | Description |
|--------|-----------|-------------|
| MPB | Single | `OMP_NUM_THREADS=1` |
| MPB | Multi | All available cores |
| Blaze2D | Single | Sequential CLI calls |
| Blaze2D | Multi | Bulk driver (16 threads) |

**Total runs**: 2 solvers × 2 lattices × 2 polarizations × 2 core-modes = **16 runs**  
**Samples per run**: 1000 jobs × 10 iterations = **10,000**

## Quick Start

```bash
# Quick test (fast validation, ~5 minutes)
make quick

# Full benchmark (1000 runs × 10 iterations, ~hours)
make all
```

## Individual Benchmarks

```bash
# Single-core comparison
make mpb-single      # Requires mpb-reference conda env
make blaze-single

# Multi-core comparison (16 threads)
make mpb-multi
make blaze-multi

# Analysis and plotting
make analyze
make plot
```

## Output Files

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `mpb_speed_single_results.json` | MPB single-core timing |
| `mpb_speed_multi_results.json` | MPB multi-core timing |
| `blaze2d_speed_single_results.json` | Blaze2D single-core timing |
| `blaze2d_speed_multi_results.json` | Blaze2D multi-core timing |
| `speed_comparison.json` | Speedup analysis |
| `speed_benchmark_report.png` | Summary figure |

## Requirements

- **MPB**: `mpb-reference` conda environment with meep/mpb
- **Blaze2D**: Rust toolchain (uses `cargo build --release`)
- **Plotting**: matplotlib, numpy
