# Blaze2D Benchmark Suite Makefile
#
# Speed benchmarks comparing MPB vs Blaze2D
# - Single-core: Fair 1-thread comparison
# - Multi-core: 16 threads parallel comparison

.PHONY: all quick single multi analyze plot clean help
.PHONY: mpb-single mpb-multi blaze-single blaze-single-full blaze-multi blaze-multi-full
.PHONY: single-mpb single-blaze multi-mpb multi-blaze
.PHONY: series1 series1-quick plot-series1
.PHONY: series2 series2-quick plot-series2
.PHONY: series3 series3-quick plot-series3
.PHONY: series4 plot-series4
.PHONY: series5 plot-series5
.PHONY: series6 series6-quick plot-series6
.PHONY: series7 series7-low series7-high plot-series7

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
multi: mpb-multi-native mpb-multi-process blaze-multi

# Individual solver targets
single-mpb: mpb-single
single-blaze: blaze-single
multi-mpb: mpb-multi-native mpb-multi-process  # Runs both MPB variants
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

mpb-multi-native:
	@echo "========================================"
	@echo "MPB Multi-Core Benchmark (Native Threads)"
	@echo "========================================"
	OMP_NUM_THREADS=16 python bench_mpb_speed.py --runs 32 --iterations 2 --output results --tag multi-native

mpb-multi-process:
	@echo "========================================"
	@echo "MPB Multi-Core Benchmark (Multiprocessing)"
	@echo "========================================"
	$(MPB_PYTHON) bench_mpb_multiprocess.py --jobs 32 --workers 16 --iterations 2 --output results --tag multi-process

mpb-single-quick:
	$(SINGLE_CORE_ENV) python bench_mpb_speed.py --cores 1 --quick --output results

mpb-multi-quick:
	OMP_NUM_THREADS=16 python bench_mpb_speed.py --runs 32 --iterations 2 --output results --tag multi-native
	$(MPB_PYTHON) bench_mpb_multiprocess.py --jobs 32 --workers 16 --iterations 2 --output results --tag multi-process

# ============================================================================
# Blaze2D Benchmarks (using unified bulk-driver script)
# ============================================================================

blaze-single:
	@echo "========================================"
	@echo "Blaze2D Single-Core Benchmark (bulk-driver -j 1)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	python bench_blaze2d.py --mode single --output results

blaze-single-full:
	@echo "========================================"
	@echo "Blaze2D Single-Core Benchmark (Full Precision f64)"
	@echo "========================================"
	@echo "Building with --no-default-features --features native-linalg"
	python bench_blaze2d.py --mode single --output results --full-precision

blaze-multi:
	@echo "========================================"
	@echo "Blaze2D Multi-Core Benchmark (bulk-driver -j 16)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	python bench_blaze2d.py --mode multi --threads 16 --jobs 32 --iterations 2 --output results

blaze-multi-full:
	@echo "========================================"
	@echo "Blaze2D Multi-Core Benchmark (Full Precision f64)"
	@echo "========================================"
	@echo "Building with --no-default-features --features native-linalg"
	python bench_blaze2d.py --mode multi --threads 16 --jobs 32 --iterations 2 --output results --full-precision

blaze-single-quick:
	python bench_blaze2d.py --mode single --quick --output results

blaze-multi-quick:
	python bench_blaze2d.py --mode multi --threads 16 --jobs 32 --iterations 2 --output results

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
	@echo "Generating All Plots"
	@echo "========================================"
	@echo "[1/4] Main speed benchmark report..."
	python analyze_speed.py --output results
	python plot_results.py --output results --individual
	@echo ""
	@echo "[2/4] Series 1 (Epsilon Sweep)..."
	@if [ -f results/series1_epsilon/series1_epsilon_results.json ]; then \
		python plot_series1_epsilon.py --output results/series1_epsilon --individual; \
	else \
		echo "  Series 1 results not found - skipping (run 'make series1' first)"; \
	fi
	@echo ""
	@echo "[3/4] Series 2 (Bands Sweep)..."
	@if [ -f results/series2_bands/series2_bands_results.json ]; then \
		python plot_series2_bands.py --results-dir results/series2_bands; \
	else \
		echo "  Series 2 results not found - skipping (run 'make series2' first)"; \
	fi
	@echo ""
	@echo "[4/5] Series 3 (Resolution Sweep)..."
	@if [ -f results/series3_resolution/series3_resolution_results.json ]; then \
		python plot_series3_resolution.py --input results/series3_resolution; \
	else \
		echo "  Series 3 results not found - skipping (run 'make series3' first)"; \
	fi
	@echo ""
	@echo "[5/6] Series 4 (Iteration Count)..."
	@if [ -f results/series4_iterations/series4_iterations_results.json ]; then \
		python plot_series4_iterations.py --input results/series4_iterations; \
	else \
		echo "  Series 4 results not found - skipping (run 'make series4' first)"; \
	fi
	@echo ""
	@echo "[6/7] Series 5 (Memory Usage)..."
	@if [ -f results/series5_memory/series5_memory_results.json ]; then \
		python plot_series5_memory.py --input results/series5_memory; \
	else \
		echo "  Series 5 results not found - skipping (run 'make series5' first)"; \
	fi
	@echo ""
	@echo "[7/8] Series 6 (Accuracy Comparison)..."
	@if [ -f results/series6_accuracy/series6_accuracy_results.json ]; then \
		python plot_series6_accuracy.py --output results/series6_accuracy/plots; \
	else \
		echo "  Series 6 results not found - skipping (run 'make series6' first)"; \
	fi
	@echo ""
	@echo "[8/8] Series 7 (Scaling)..."
	@if [ -f results/series7_scaling/series7_scaling_results.json ]; then \
		python plot_series7_scaling.py --output results/series7_scaling/plots; \
	else \
		echo "  Series 7 results not found - skipping (run 'make series7' first)"; \
	fi
	@echo ""
	@echo "Done!"

# ============================================================================
# Series 1: Epsilon Sweep on Square Lattice
# Varies epsilon from 2 to 13 in 0.5 steps
# Square lattice, r=0.2a, 64×64 resolution, 8 bands, single-core only
# ============================================================================

series1:
	@echo "========================================"
	@echo "Series 1: Epsilon Sweep (Full)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series1_epsilon.py --output results/series1_epsilon
	python plot_series1_epsilon.py --output results/series1_epsilon

series1-quick:
	@echo "========================================"
	@echo "Series 1: Epsilon Sweep (Quick)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series1_epsilon.py --quick --output results/series1_epsilon
	python plot_series1_epsilon.py --output results/series1_epsilon

plot-series1:
	@echo "========================================"
	@echo "Plotting Series 1 Results"
	@echo "========================================"
	python plot_series1_epsilon.py --output results/series1_epsilon --individual

# ============================================================================
# Series 2: Number of Bands Sweep on Square Lattice
# Varies bands from 4 to 20 in steps of 1
# Square lattice, ε=8.9 rods, r=0.2a, 32×32 resolution, single-core only
# ============================================================================

series2:
	@echo "========================================"
	@echo "Series 2: Bands Sweep (Full)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series2_bands.py --output results/series2_bands
	python plot_series2_bands.py --results-dir results/series2_bands --summary

series2-quick:
	@echo "========================================"
	@echo "Series 2: Bands Sweep (Quick)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series2_bands.py --quick --output results/series2_bands
	python plot_series2_bands.py --results-dir results/series2_bands --summary

plot-series2:
	@echo "========================================"
	@echo "Plotting Series 2 Results"
	@echo "========================================"
	python plot_series2_bands.py --results-dir results/series2_bands --summary

# ============================================================================
# Series 3: Resolution Sweep on Square Lattice
# Varies resolution from 16 to 512
# Square lattice, ε=8.9 rods, r=0.2a, 10 bands, single-core only
# ============================================================================

series3:
	@echo "========================================"
	@echo "Series 3: Resolution Sweep (Full)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series3_resolution.py --output results/series3_resolution
	python plot_series3_resolution.py --input results/series3_resolution

series3-quick:
	@echo "========================================"
	@echo "Series 3: Resolution Sweep (Quick)"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-bulk-driver
	$(SINGLE_CORE_ENV) $(MPB_PYTHON) bench_series3_resolution.py --quick --output results/series3_resolution
	python plot_series3_resolution.py --input results/series3_resolution

plot-series3:
	@echo "========================================"
	@echo "Plotting Series 3 Results"
	@echo "========================================"
	python plot_series3_resolution.py --input results/series3_resolution

# ============================================================================
# Series 4: Iteration Count per K-Point
# Compares number of LOBPCG iterations needed by MPB vs Blaze2D
# Square lattice, ε=8.9 rods, r=0.2a, 32×32, 8 bands
# ============================================================================

series4:
	@echo "========================================"
	@echo "Series 4: Iteration Count Comparison"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-cli
	$(MPB_PYTHON) bench_series4_iterations.py --output results/series4_iterations
	python plot_series4_iterations.py --input results/series4_iterations

plot-series4:
	@echo "========================================"
	@echo "Plotting Series 4 Results"
	@echo "========================================"
	python plot_series4_iterations.py --input results/series4_iterations

# ============================================================================
# Series 5: Memory Usage Comparison
# Compares Peak RSS memory usage between MPB and Blaze2D
# Three sweeps: resolution, bands, k-points per segment
# ============================================================================

series5:
	@echo "========================================"
	@echo "Series 5: Memory Usage Comparison"
	@echo "========================================"
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-cli
	$(MPB_PYTHON) bench_series5_memory.py --output results/series5_memory
	python plot_series5_memory.py --input results/series5_memory

plot-series5:
	@echo "========================================"
	@echo "Plotting Series 5 Results"
	@echo "========================================"
	python plot_series5_memory.py --input results/series5_memory

# ============================================================================
# Series 6: Accuracy Comparison
# Compares eigenvalue accuracy between MPB, Blaze-f32, Blaze-f64
# Square lattice, ε=8.9 rods, r=0.2a, 32×32, 8 bands
# Uses Hungarian algorithm for optimal band matching
# ============================================================================

series6:
	@echo "========================================"
	@echo "Series 6: Accuracy Comparison (Full)"
	@echo "========================================"
	$(MPB_PYTHON) bench_series6_accuracy.py --output results/series6_accuracy
	python plot_series6_accuracy.py --output results/series6_accuracy/plots

series6-quick:
	@echo "========================================"
	@echo "Series 6: Accuracy Comparison (Skip f64)"
	@echo "========================================"
	$(MPB_PYTHON) bench_series6_accuracy.py --skip-f64 --output results/series6_accuracy
	python plot_series6_accuracy.py --output results/series6_accuracy/plots

plot-series6:
	@echo "========================================"
	@echo "Plotting Series 6 Results"
	@echo "========================================"
	python plot_series6_accuracy.py --output results/series6_accuracy/plots

# ============================================================================
# Series 7: Multi-Threading Scaling
# Measures scaling with thread count: 1, 2, 4, 8, 12, 16
# Two resolutions: low (16×16) and high (128×128)
# Three solvers: Blaze2D, MPB-OMP, MPB-Multiproc
# Small benchmark: 2× threads jobs, 2 iterations
# ============================================================================

series7:
	@echo "========================================"
	@echo "Series 7: Multi-Threading Scaling (Full)"
	@echo "========================================"
	$(MPB_PYTHON) bench_series7_scaling.py --output results/series7_scaling
	python plot_series7_scaling.py --output results/series7_scaling/plots

series7-low:
	@echo "========================================"
	@echo "Series 7: Scaling (Low Resolution Only)"
	@echo "========================================"
	$(MPB_PYTHON) bench_series7_scaling.py --low-only --output results/series7_scaling
	python plot_series7_scaling.py --output results/series7_scaling/plots

series7-high:
	@echo "========================================"
	@echo "Series 7: Scaling (High Resolution Only)"
	@echo "========================================"
	$(MPB_PYTHON) bench_series7_scaling.py --high-only --output results/series7_scaling
	python plot_series7_scaling.py --output results/series7_scaling/plots

plot-series7:
	@echo "========================================"
	@echo "Plotting Series 7 Results"
	@echo "========================================"
	python plot_series7_scaling.py --output results/series7_scaling/plots

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
	@echo "MPB Multiprocessing (16 workers, 32 jobs × 2 iter)"
	@echo "========================================"
	$(MPB_PYTHON) bench_mpb_multiprocess.py --jobs 32 --workers 16 --iterations 2

bulk-blaze:
	@echo "========================================"
	@echo "Blaze2D Bulk Driver (16 threads, 32 jobs × 2 iter)"
	@echo "========================================"
	python bench_blaze2d.py --mode multi --threads 16 --jobs 32 --iterations 2 --config config_a_tm --output results

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
	@echo "  analyze      - Analyze existing results (generates speed_comparison.json)"
	@echo "  plot         - Regenerate ALL plots (main report + series1-4)"
	@echo ""
	@echo "Benchmark Series:"
	@echo "  series1       - Series 1: Epsilon sweep (2-13, square lattice)"
	@echo "  series1-quick - Series 1 quick validation run"
	@echo "  plot-series1  - Plot Series 1 results only"
	@echo "  series2       - Series 2: Bands sweep (4-20, square lattice)"
	@echo "  series2-quick - Series 2 quick validation run"
	@echo "  plot-series2  - Plot Series 2 results only"
	@echo "  series3       - Series 3: Resolution sweep (16-512, square lattice)"
	@echo "  series3-quick - Series 3 quick validation run"
	@echo "  plot-series3  - Plot Series 3 results only"
	@echo "  series4       - Series 4: Iteration count per k-point"
	@echo "  plot-series4  - Plot Series 4 results only"
	@echo "  series5       - Series 5: Memory usage comparison"
	@echo "  plot-series5  - Plot Series 5 results only"
	@echo "  series6       - Series 6: Accuracy comparison (f32 vs f64 vs MPB)"
	@echo "  series6-quick - Series 6 quick run (skip f64 build)"
	@echo "  plot-series6  - Plot Series 6 results only"
	@echo "  series7       - Series 7: Multi-threading scaling (low + high res)"
	@echo "  series7-low   - Series 7 low resolution only (16×16)"
	@echo "  series7-high  - Series 7 high resolution only (128×128)"
	@echo "  plot-series7  - Plot Series 7 results only"
	@echo ""
	@echo "Individual Solver Benchmarks:"
	@echo "  single-mpb    - MPB single-core only"
	@echo "  single-blaze  - Blaze2D single-core only"
	@echo "  multi-mpb     - MPB multi-core only"
	@echo "  multi-blaze   - Blaze2D multi-core only"
	@echo ""
	@echo "Alternative names (same as above):"
	@echo "  mpb-single, mpb-multi, blaze-single, blaze-multi"
	@echo ""
	@echo "Utility:"
	@echo "  clean        - Remove results directory"
	@echo "  summary      - Show results summary"
	@echo ""
	@echo "Benchmark Samples:"
	@echo "  Single-core: 5 runs × 2 iterations = 10 samples per config"
	@echo "  Multi-core: 32 jobs × 2 iterations = 64 samples per config (16 threads)"
	@echo ""
	@echo "Series 1 (Epsilon Sweep):"
	@echo "  Parameters: square lattice, r=0.2a, 64×64, 8 bands, single-core"
	@echo "  Epsilon values: 2.0, 2.5, 3.0, ..., 13.0 (23 values)"
	@echo "  Solvers: MPB, Blaze2D"
	@echo ""
	@echo "Series 2 (Bands Sweep):"
	@echo "  Parameters: square lattice, ε=8.9, r=0.2a, 32×32, single-core"
	@echo "  Band values: 4, 5, 6, ..., 20 (17 values)"
	@echo "  Solvers: MPB, Blaze2D"
	@echo ""
	@echo "Series 3 (Resolution Sweep):"
	@echo "  Parameters: square lattice, ε=8.9, r=0.2a, 10 bands, single-core"
	@echo "  Resolution values: 16, 24, 32, 48, 64, 96, 128, 192"
	@echo "  Solvers: MPB, Blaze2D"
	@echo ""
	@echo "Series 4 (Iteration Count):"
	@echo "  Parameters: square lattice, ε=8.9, r=0.2a, 32×32, 8 bands"
	@echo "  Measures: LOBPCG iterations per k-point for full k-path"
	@echo "  Solvers: MPB, Blaze2D"
	@echo ""
	@echo "Series 5 (Memory Usage):"
	@echo "  Parameters: square lattice, ε=8.9, r=0.2a"
	@echo "  Sweeps: resolution (16-128), bands (4-20), k-points (6-30)"
	@echo "  Measures: Peak RSS (Resident Set Size) in MB"
	@echo "  Solvers: MPB, Blaze2D"
	@echo ""
	@echo "Series 6 (Accuracy Comparison):"
	@echo "  Parameters: square lattice, ε=8.9, r=0.2a, 32×32, 8 bands"
	@echo "  Compares: MPB (f64 reference), Blaze-f32 (mixed), Blaze-f64 (full)"
	@echo "  Measures: Relative eigenvalue deviation with Hungarian band matching"
	@echo ""
	@echo "Series 7 (Multi-Threading Scaling):"
	@echo "  Thread counts: 1, 2, 4, 8, 12, 16"
	@echo "  Resolutions: low (16×16), high (128×128)"
	@echo "  Solvers: Blaze2D, MPB-OMP (native threads), MPB-Multiproc (workers)"
	@echo "  Measures: Throughput, speedup, parallel efficiency"
	@echo "  Small benchmark: 2× thread count jobs, 2 iterations"
	@echo ""
	@echo "Requirements:"
	@echo "  - MPB: 'mpb-reference' conda environment"
	@echo "  - Blaze2D: Rust toolchain"
	@echo "  - Plotting: matplotlib, numpy"
