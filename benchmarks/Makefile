# Blaze2D Benchmark Suite Makefile
#
# Speed benchmarks comparing MPB vs Blaze2D
# - Single-core: Fair 1-thread comparison
# - Multi-core: 16 threads parallel comparison

.PHONY: all quick single multi analyze plot clean help
.PHONY: mpb-single mpb-multi blaze-single blaze-multi
.PHONY: single-mpb single-blaze multi-mpb multi-blaze

# ============================================================================
# Main targets
# ============================================================================

# Full benchmark: all modes (takes ~hours)
all: single multi analyze plot

# Quick test (10 runs × 2 iterations) - for validation
quick: quick-single quick-multi analyze plot

# Single-core benchmarks only
single: single-mpb single-blaze

# Multi-core benchmarks only  
multi: multi-mpb multi-blaze

# Individual solver targets (alternative naming convention)
single-mpb: mpb-single
single-blaze: blaze-single
multi-mpb: mpb-multi
multi-blaze: blaze-multi

# ============================================================================
# MPB Benchmarks (requires mpb-reference conda environment)
# ============================================================================

# Set ALL threading env vars for single-core mode
SINGLE_CORE_ENV := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 GOTO_NUM_THREADS=1

mpb-single:
	@echo "========================================"
	@echo "MPB Single-Core Benchmark"
	@echo "========================================"
	$(SINGLE_CORE_ENV) python bench_mpb_speed.py --cores 1 --output results

mpb-multi:
	@echo "========================================"
	@echo "MPB Multi-Core Benchmark"
	@echo "========================================"
	python bench_mpb_speed.py --output results

mpb-single-quick:
	$(SINGLE_CORE_ENV) python bench_mpb_speed.py --cores 1 --quick --output results

mpb-multi-quick:
	python bench_mpb_speed.py --quick --output results

# ============================================================================
# Blaze2D Benchmarks (using unified bulk-driver script)
# ============================================================================

blaze-single:
	@echo "========================================"
	@echo "Blaze2D Single-Core Benchmark (bulk-driver -j 1)"
	@echo "========================================"
	python bench_blaze2d.py --mode single --output results

blaze-multi:
	@echo "========================================"
	@echo "Blaze2D Multi-Core Benchmark (bulk-driver -j 16)"
	@echo "========================================"
	python bench_blaze2d.py --mode multi --threads 16 --output results

blaze-single-quick:
	python bench_blaze2d.py --mode single --quick --output results

blaze-multi-quick:
	python bench_blaze2d.py --mode multi --quick --threads 16 --output results

# ============================================================================
# Quick test targets
# ============================================================================

quick-single: mpb-single-quick blaze-single-quick

quick-multi: mpb-multi-quick blaze-multi-quick

# ============================================================================
# Analysis and plotting
# ============================================================================

analyze:
	@echo "========================================"
	@echo "Analyzing Results"
	@echo "========================================"
	python analyze_speed.py --output results

plot:
	@echo "========================================"
	@echo "Generating Plots"
	@echo "========================================"
	python plot_results.py --output results --individual

# ============================================================================
# Bulk Comparison: Fair MPB multiprocessing vs Blaze2D bulk-driver
# Config A TM, 64×64, 16 workers, 100 jobs × 5 iterations
# ============================================================================

MPB_PYTHON := /home/renlephy/.local/share/mamba/envs/mpb-reference/bin/python

bulk-compare: bulk-mpb bulk-blaze
	@echo "========================================"
	@echo "BULK COMPARISON COMPLETE"
	@echo "========================================"
	@echo "Results in: results/"

bulk-mpb:
	@echo "========================================"
	@echo "MPB Multiprocessing (16 workers, 100 jobs × 5 iter)"
	@echo "========================================"
	$(MPB_PYTHON) bench_mpb_multiprocess.py

bulk-blaze:
	@echo "========================================"
	@echo "Blaze2D Bulk Driver (16 threads, 100 jobs × 5 iter)"
	@echo "========================================"
	python bench_blaze2d.py --mode multi --threads 16 --config config_a_tm --output results

# ============================================================================
# Utility targets
# ============================================================================

clean:
	rm -rf results/

# Show results summary
summary:
	@if [ -f results/speed_comparison.csv ]; then \
		echo "Speed Comparison Results:"; \
		cat results/speed_comparison.csv; \
	else \
		echo "No results found. Run benchmarks first."; \
	fi

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Blaze2D Benchmark Suite"
	@echo ""
	@echo "Main Targets:"
	@echo "  all          - Run full benchmark suite (single + multi core)"
	@echo "  quick        - Quick validation test (5 jobs × 2 iterations)"
	@echo "  single       - Single-core benchmarks only (MPB + Blaze2D)"
	@echo "  multi        - Multi-core benchmarks only (MPB + Blaze2D)"
	@echo "  analyze      - Analyze existing results"
	@echo "  plot         - Generate plots from results"
	@echo ""
	@echo "Individual Solver Benchmarks:"
	@echo "  single-mpb   - MPB single-core only"
	@echo "  single-blaze - Blaze2D single-core only"
	@echo "  multi-mpb    - MPB multi-core only"
	@echo "  multi-blaze  - Blaze2D multi-core only"
	@echo ""
	@echo "Alternative names (same as above):"
	@echo "  mpb-single, mpb-multi, blaze-single, blaze-multi"
	@echo ""
	@echo "Utility:"
	@echo "  clean        - Remove results directory"
	@echo "  summary      - Show results summary"
	@echo ""
	@echo "Benchmark Samples:"
	@echo "  Single-core: 10 jobs × 10 iterations = 100 samples per config"
	@echo "  Multi-core: 100 jobs × 10 iterations = 1000 samples per config"
	@echo ""
	@echo "Requirements:"
	@echo "  - MPB: 'mpb-reference' conda environment"
	@echo "  - Blaze2D: Rust toolchain"
	@echo "  - Plotting: matplotlib, numpy"
