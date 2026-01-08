# ============================================================================
# Profiling Makefile for blaze2d
# ============================================================================
#
# This Makefile runs profiling benchmarks for both TE and TM polarizations
# using 64x64 resolution, 12 bands, and 10 k-points per leg.
# Results are averaged over multiple runs for statistical stability.
#
# Usage:
#   make profile          # Run profiling and generate diff report
#   make profile-te       # Run only TE mode profiling
#   make profile-tm       # Run only TM mode profiling
#   make baseline         # Create new baseline profiles (run once before optimizing)
#   make clean            # Remove current profile outputs (keeps baselines and report)
#   make clean-all        # Remove everything including baselines and report
#
# A/B Testing (batched vs sequential FFT):
#   make ab-test          # Run A/B test comparing batched vs sequential FFT
#   make ab-test-te       # A/B test for TE mode only
#   make ab-test-tm       # A/B test for TM mode only
#
# The diff report is appended with each run, so you can track changes over time.
# Each entry has a blank "Description:" field for manual notes.
#
# ============================================================================

# Directories
ROOT_DIR := $(shell cd ../.. && pwd)
PROFILE_DIR := $(ROOT_DIR)/benchmarks/profiles
RESULTS_DIR := $(PROFILE_DIR)/results
BASELINE_DIR := $(PROFILE_DIR)/baseline
AB_TEST_DIR := $(PROFILE_DIR)/ab_test

# Tools
BLAZE_CLI := $(ROOT_DIR)/target/release/blaze2d-cli
BLAZE_CLI_SEQUENTIAL := $(ROOT_DIR)/target/release-sequential/release/blaze2d-cli
DIFF_TOOL := $(ROOT_DIR)/benchmarks/profile_diff.py
AGGREGATE_TOOL := $(ROOT_DIR)/benchmarks/profile_aggregate.py

# Configuration files (64x64 res, 12 bands, designed for profiling)
TE_CONFIG := $(ROOT_DIR)/examples/profile_square_te_res64.toml
TM_CONFIG := $(ROOT_DIR)/examples/profile_square_tm_res64.toml

# Number of iterations to average (higher = more stable, slower)
ITERATIONS := 2

# Output files
BASELINE_TE := $(BASELINE_DIR)/baseline_te.json
BASELINE_TM := $(BASELINE_DIR)/baseline_tm.json
CURRENT_TE := $(RESULTS_DIR)/current_te.json
CURRENT_TM := $(RESULTS_DIR)/current_tm.json
DIFF_REPORT := $(PROFILE_DIR)/profile_history.txt

# Threshold for highlighting changes (percentage)
THRESHOLD := 5.0

# ============================================================================
# Phony targets
# ============================================================================
.PHONY: all profile profile-te profile-tm baseline baseline-te baseline-tm \
        clean clean-all build build-sequential check-baseline help quick \
        ab-test ab-test-te ab-test-tm

# Default target
all: profile

help:
	@echo "Profiling Makefile for blaze2d"
	@echo ""
	@echo "Configuration:"
	@echo "  Resolution:    64x64"
	@echo "  Bands:         12"
	@echo "  K-points:      10 segments/leg (~40 k-points)"
	@echo "  Iterations:    $(ITERATIONS) runs (averaged)"
	@echo ""
	@echo "Targets:"
	@echo "  profile       Run profiling for both TE and TM modes, generate diff report"
	@echo "  profile-te    Run only TE mode profiling"
	@echo "  profile-tm    Run only TM mode profiling"
	@echo "  baseline      Create new baseline profiles (run once before optimizing)"
	@echo "  baseline-te   Create only TE baseline"
	@echo "  baseline-tm   Create only TM baseline"
	@echo "  quick         Single-run profiling (faster, higher variance)"
	@echo "  clean         Remove current profile outputs (keeps baselines and report)"
	@echo "  clean-all     Remove everything including baselines and report"
	@echo "  build         Build blaze2d with profiling enabled (batched FFT)"
	@echo ""
	@echo "A/B Testing (batched vs sequential FFT):"
	@echo "  ab-test       Compare batched vs sequential FFT for both modes"
	@echo "  ab-test-te    Compare batched vs sequential FFT for TE only"
	@echo "  ab-test-tm    Compare batched vs sequential FFT for TM only"
	@echo ""
	@echo "Output files:"
	@echo "  $(DIFF_REPORT)"
	@echo ""

# ============================================================================
# Build
# ============================================================================
build:
	@echo "Building blaze2d with profiling enabled (batched FFT)..."
	cd $(ROOT_DIR) && RUSTFLAGS="-C target-cpu=native" cargo build --release -p blaze2d-cli --features profiling
	@echo "Build complete."

build-sequential:
	@echo "Building blaze2d with profiling enabled (sequential FFT)..."
	cd $(ROOT_DIR) && RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=$(ROOT_DIR)/target/release-sequential cargo build --release -p blaze2d-cli --features profiling --no-default-features
	@echo "Build complete."

# Check that CLI exists
$(BLAZE_CLI):
	@echo "Error: $(BLAZE_CLI) not found."
	@echo "Run 'make build' first to compile with profiling enabled."
	@exit 1

$(BLAZE_CLI_SEQUENTIAL):
	@echo "Error: $(BLAZE_CLI_SEQUENTIAL) not found."
	@echo "Run 'make build-sequential' first to compile without batched FFT."
	@exit 1

# ============================================================================
# Directory setup
# ============================================================================
$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(BASELINE_DIR):
	mkdir -p $(BASELINE_DIR)

# ============================================================================
# Baseline creation (averaged over multiple runs)
# ============================================================================
baseline: baseline-te baseline-tm
	@echo ""
	@echo "Baselines created successfully (averaged over $(ITERATIONS) runs)."
	@echo "  TE: $(BASELINE_TE)"
	@echo "  TM: $(BASELINE_TM)"

baseline-te: $(BLAZE_CLI) $(BASELINE_DIR)
	@echo "Creating TE baseline profile ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TE_CONFIG) \
		--output $(BASELINE_TE) \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo "TE baseline saved to $(BASELINE_TE)"

baseline-tm: $(BLAZE_CLI) $(BASELINE_DIR)
	@echo "Creating TM baseline profile ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TM_CONFIG) \
		--output $(BASELINE_TM) \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo "TM baseline saved to $(BASELINE_TM)"

# ============================================================================
# Check baselines exist
# ============================================================================
check-baseline:
	@if [ ! -f $(BASELINE_TE) ] || [ ! -f $(BASELINE_TM) ]; then \
		echo "Error: Baseline profiles not found."; \
		echo "Run 'make baseline' first to create baseline profiles."; \
		exit 1; \
	fi

# ============================================================================
# Profiling runs (averaged over multiple runs)
# ============================================================================
profile: check-baseline profile-te profile-tm
	@echo ""
	@echo "============================================"
	@echo "Profiling complete!"
	@echo "============================================"
	@echo ""
	@echo "Diff report appended to: $(DIFF_REPORT)"
	@echo ""
	@echo "Don't forget to add a description to the report entries!"

profile-te: $(BLAZE_CLI) $(RESULTS_DIR) check-baseline $(PROFILE_DIR)
	@echo ""
	@echo "Running TE mode profiling ($(ITERATIONS) iterations)..."
	@echo "Config: $(TE_CONFIG)"
	python3 $(AGGREGATE_TOOL) \
		--config $(TE_CONFIG) \
		--output $(CURRENT_TE) \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo ""
	@echo "Comparing TE profile against baseline..."
	python3 $(DIFF_TOOL) $(BASELINE_TE) $(CURRENT_TE) \
		--threshold $(THRESHOLD) \
		--label "TE mode (64x64, 12 bands, $(ITERATIONS) iter avg)" \
		--append $(DIFF_REPORT) || true

profile-tm: $(BLAZE_CLI) $(RESULTS_DIR) check-baseline $(PROFILE_DIR)
	@echo ""
	@echo "Running TM mode profiling ($(ITERATIONS) iterations)..."
	@echo "Config: $(TM_CONFIG)"
	python3 $(AGGREGATE_TOOL) \
		--config $(TM_CONFIG) \
		--output $(CURRENT_TM) \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo ""
	@echo "Comparing TM profile against baseline..."
	python3 $(DIFF_TOOL) $(BASELINE_TM) $(CURRENT_TM) \
		--threshold $(THRESHOLD) \
		--label "TM mode (64x64, 12 bands, $(ITERATIONS) iter avg)" \
		--append $(DIFF_REPORT) || true

# ============================================================================
# Quick profiling (single run, for fast iteration)
# ============================================================================
quick: check-baseline $(BLAZE_CLI) $(RESULTS_DIR) $(PROFILE_DIR)
	@echo "Running quick profile (single run)..."
	$(BLAZE_CLI) --config $(TE_CONFIG) --output /dev/null --profile $(CURRENT_TE) 2>/dev/null
	$(BLAZE_CLI) --config $(TM_CONFIG) --output /dev/null --profile $(CURRENT_TM) 2>/dev/null
	@echo ""
	@echo "TE comparison:"
	python3 $(DIFF_TOOL) $(BASELINE_TE) $(CURRENT_TE) --threshold $(THRESHOLD) || true
	@echo ""
	@echo "TM comparison:"
	python3 $(DIFF_TOOL) $(BASELINE_TM) $(CURRENT_TM) --threshold $(THRESHOLD) || true

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -f $(CURRENT_TE) $(CURRENT_TM)
	@echo "Cleaned current profile outputs."

clean-all:
	rm -rf $(PROFILE_DIR)
	@echo "Cleaned all profile data including baselines and report."

# ============================================================================
# A/B Testing: Batched FFT vs Sequential FFT
# ============================================================================
$(AB_TEST_DIR):
	mkdir -p $(AB_TEST_DIR)

ab-test: ab-test-te ab-test-tm
	@echo ""
	@echo "============================================"
	@echo "A/B Test Complete!"
	@echo "============================================"
	@echo ""
	@echo "Results saved to $(AB_TEST_DIR)/"
	@echo "See profile_history.txt for comparison report"

ab-test-te: build build-sequential $(AB_TEST_DIR)
	@echo ""
	@echo "============================================"
	@echo "A/B Test: TE Mode (Batched vs Sequential FFT)"
	@echo "============================================"
	@echo ""
	@echo "Running BATCHED FFT ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TE_CONFIG) \
		--output $(AB_TEST_DIR)/te_batched.json \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo ""
	@echo "Running SEQUENTIAL FFT ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TE_CONFIG) \
		--output $(AB_TEST_DIR)/te_sequential.json \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI_SEQUENTIAL) \
		--verbose
	@echo ""
	@echo "TE A/B Comparison (Sequential as baseline, Batched as current):"
	python3 $(DIFF_TOOL) $(AB_TEST_DIR)/te_sequential.json $(AB_TEST_DIR)/te_batched.json \
		--threshold $(THRESHOLD) \
		--label "TE A/B: Sequential vs Batched FFT" \
		--append $(DIFF_REPORT) || true

ab-test-tm: build build-sequential $(AB_TEST_DIR)
	@echo ""
	@echo "============================================"
	@echo "A/B Test: TM Mode (Batched vs Sequential FFT)"
	@echo "============================================"
	@echo ""
	@echo "Running BATCHED FFT ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TM_CONFIG) \
		--output $(AB_TEST_DIR)/tm_batched.json \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI) \
		--verbose
	@echo ""
	@echo "Running SEQUENTIAL FFT ($(ITERATIONS) iterations)..."
	python3 $(AGGREGATE_TOOL) \
		--config $(TM_CONFIG) \
		--output $(AB_TEST_DIR)/tm_sequential.json \
		--iterations $(ITERATIONS) \
		--cli $(BLAZE_CLI_SEQUENTIAL) \
		--verbose
	@echo ""
	@echo "TM A/B Comparison (Sequential as baseline, Batched as current):"
	python3 $(DIFF_TOOL) $(AB_TEST_DIR)/tm_sequential.json $(AB_TEST_DIR)/tm_batched.json \
		--threshold $(THRESHOLD) \
		--label "TM A/B: Sequential vs Batched FFT" \
		--append $(DIFF_REPORT) || true