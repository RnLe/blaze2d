# Evaluation harness for MPB vs. blaze runs.
# Usage:
#   make                          # runs default config (compare_hex_lowres)
#   make CONFIG=config/foo.mk run # run arbitrary config file
#   make foo                      # run config/foo.mk via its short name
#   make RUST_LOG=debug foo       # enable detailed iteration logs
#   make RUST_LOG=error foo       # suppress all warnings
# Log levels: error < warn (default) < info < debug < trace
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
EVAL_ROOT := $(abspath $(dir $(THIS_MAKEFILE)))
CONFIG_DIR ?= $(EVAL_ROOT)/config
REFERENCE_DIR ?= $(EVAL_ROOT)/reference-data
DEFAULT_CONFIG := $(CONFIG_DIR)/compare_hex_lowres.mk
CONFIG ?= $(DEFAULT_CONFIG)

# Use RUST_LOG=debug for detailed iteration logs
RUST_LOG ?= info
export RUST_LOG

CONFIG_FILES := $(wildcard $(CONFIG_DIR)/*.mk)
ifeq ($(strip $(CONFIG_FILES)),)
  $(error No config files found under $(CONFIG_DIR))
endif

# All config targets, with special ones filtered out (they have explicit rules)
EVAL_TARGETS_ALL := $(patsubst $(CONFIG_DIR)/%.mk,%,$(CONFIG_FILES))
EVAL_TARGETS := $(filter-out compare_square_lowres,$(EVAL_TARGETS_ALL))

ifeq (,$(wildcard $(CONFIG)))
  $(error Config file '$(CONFIG)' not found. Override via CONFIG=path/to/config.mk)
endif

include $(CONFIG)

ifndef EVAL_NAME
  $(error EVAL_NAME must be defined inside $(CONFIG))
endif

.DEFAULT_GOAL := run

SKIP_REFERENCE_WIPE ?= 0

.PHONY: all run mpb blaze blaze-benchmark list-config clean prepare_reference_data help \
        export-epsilon export-epsilon-tensor export-epsilon-all export-iterations \
        export-epsilon-mpb export-epsilon-inverse-mpb export-epsilon-compare benchmark $(EVAL_TARGETS)

help:
	@echo "Evaluation harness for MPB vs. blaze runs."
	@echo ""
	@echo "Usage:"
	@echo "  make                               # run default config (compare_hex_lowres)"
	@echo "  make <config-name>                 # run a specific config by short name"
	@echo "  make CONFIG=config/foo.mk run      # run arbitrary config file"
	@echo "  make all                           # run all configs"
	@echo ""
	@echo "Data Export:"
	@echo "  make export-epsilon                # export epsilon(r) data to CSV"
	@echo "  make export-epsilon-tensor         # export inverse epsilon tensor to CSV"
	@echo "  make export-epsilon-all            # export both epsilon files"
	@echo "  make export-epsilon-inverse-mpb    # export inverse epsilon tensor from MPB"
	@echo "  make export-iterations             # export per-iteration convergence data"
	@echo ""
	@echo "Available configs: $(EVAL_TARGETS)"
	@echo ""
	@echo "Other targets:"
	@echo "  make list-config                   # show current config details"
	@echo "  make clean                         # remove generated outputs"
	@echo ""
	@echo "Environment:"
	@echo "  RUST_LOG=debug make <target>       # enable detailed iteration logs"
	@echo "  RUST_LOG=error make <target>       # suppress warnings"

all:
	@$(MAKE) --no-print-directory prepare_reference_data
	@echo "[evaluation] running all configs: $(EVAL_TARGETS)"
	@for target in $(EVAL_TARGETS); do \
		echo "[evaluation] >>> $$target"; \
		$(MAKE) --no-print-directory SKIP_REFERENCE_WIPE=1 $$target; \
	done

run:
	@$(MAKE) --no-print-directory prepare_reference_data
	@$(MAKE) --no-print-directory mpb
	@$(MAKE) --no-print-directory blaze

# Special target for compare_square_lowres: use benchmark-enabled blaze
compare_square_lowres:
	@echo "[evaluation] launching 'compare_square_lowres' with benchmark tracking"
	@$(MAKE) --no-print-directory SKIP_REFERENCE_WIPE=$(SKIP_REFERENCE_WIPE) prepare_reference_data CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk
	@$(MAKE) --no-print-directory mpb CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk
	@$(MAKE) --no-print-directory blaze-benchmark CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk
	@$(MAKE) --no-print-directory blaze-cuda CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk
	@$(MAKE) --no-print-directory export-epsilon-compare CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk
	@$(MAKE) --no-print-directory export-iterations CONFIG=$(CONFIG_DIR)/compare_square_lowres.mk

$(EVAL_TARGETS):
	@echo "[evaluation] launching '$@' via $(CONFIG_DIR)/$@.mk"
	@$(MAKE) --no-print-directory SKIP_REFERENCE_WIPE=$(SKIP_REFERENCE_WIPE) prepare_reference_data CONFIG=$(CONFIG_DIR)/$@.mk
	@$(MAKE) --no-print-directory mpb CONFIG=$(CONFIG_DIR)/$@.mk
	@$(MAKE) --no-print-directory blaze CONFIG=$(CONFIG_DIR)/$@.mk
	@$(MAKE) --no-print-directory export-epsilon-compare CONFIG=$(CONFIG_DIR)/$@.mk
	@$(MAKE) --no-print-directory export-iterations CONFIG=$(CONFIG_DIR)/$@.mk

mpb:
	@if [ -z "$(strip $(MPB_COMMAND))" ]; then \
		echo "MPB_COMMAND is empty in $(CONFIG)"; \
		exit 1; \
	fi
	@echo "[evaluation] running MPB for $(EVAL_NAME)"
	$(MPB_COMMAND)

blaze:
	@if [ -z "$(strip $(BLAZE_COMMAND))" ]; then \
		echo "BLAZE_COMMAND is empty in $(CONFIG)"; \
		exit 1; \
	fi
	@echo "[evaluation] running blaze for $(EVAL_NAME)"
	$(BLAZE_COMMAND)

# blaze with CUDA acceleration
blaze-cuda:
	@if [ -z "$(strip $(BLAZE_CUDA_COMMAND))" ]; then \
		echo "BLAZE_CUDA_COMMAND is empty in $(CONFIG)"; \
		exit 1; \
	fi
	@echo "[evaluation] running blaze (CUDA) for $(EVAL_NAME)"
	$(BLAZE_CUDA_COMMAND)

# blaze with benchmark logging (for compare_square_lowres)
# Uses script -q to preserve TTY for colored output while logging
blaze-benchmark:
	@if [ -z "$(strip $(BLAZE_TE_CMD))" ] || [ -z "$(strip $(BENCHMARK_YAML))" ]; then \
		echo "Error: This config does not support benchmarking."; \
		exit 1; \
	fi
	@echo "[evaluation] running blaze for $(EVAL_NAME) with benchmark tracking"
	@echo "[benchmark] running TE..."
	@script -q -c "$(BLAZE_TE_CMD)" $(BENCHMARK_LOG_TE)
	@$(EVAL_ROOT)/append_benchmark.sh $(BENCHMARK_YAML) TE $(BENCHMARK_LOG_TE)
	@echo "[benchmark] running TM..."
	@script -q -c "$(BLAZE_TM_CMD)" $(BENCHMARK_LOG_TM)
	@$(EVAL_ROOT)/append_benchmark.sh $(BENCHMARK_YAML) TM $(BENCHMARK_LOG_TM)
	@echo "[benchmark] results appended to $(BENCHMARK_YAML)"

# ============================================================================
# Benchmark Tracking Target
# ============================================================================
# Runs blaze with logging and appends metrics to a YAML file for tracking
# solver performance over time. Only works with configs that define
# BLAZE_TE_CMD, BLAZE_TM_CMD, BENCHMARK_YAML, BENCHMARK_LOG_TE, BENCHMARK_LOG_TM.
#
# Usage:
#   make benchmark CONFIG=config/compare_square_lowres.mk

benchmark:
	@if [ -z "$(strip $(BLAZE_TE_CMD))" ] || [ -z "$(strip $(BENCHMARK_YAML))" ]; then \
		echo "Error: This config does not support benchmarking."; \
		echo "Required: BLAZE_TE_CMD, BLAZE_TM_CMD, BENCHMARK_YAML, BENCHMARK_LOG_TE, BENCHMARK_LOG_TM"; \
		exit 1; \
	fi
	@$(MAKE) --no-print-directory prepare_reference_data
	@echo "[benchmark] running TE..."
	@$(BLAZE_TE_CMD) 2>&1 | tee $(BENCHMARK_LOG_TE)
	@$(EVAL_ROOT)/append_benchmark.sh $(BENCHMARK_YAML) TE $(BENCHMARK_LOG_TE)
	@echo "[benchmark] running TM..."
	@$(BLAZE_TM_CMD) 2>&1 | tee $(BENCHMARK_LOG_TM)
	@$(EVAL_ROOT)/append_benchmark.sh $(BENCHMARK_YAML) TM $(BENCHMARK_LOG_TM)
	@echo "[benchmark] done. Results in $(BENCHMARK_YAML)"

list-config:
	@echo "Loaded evaluation: $(EVAL_NAME)"
	@echo "Config file: $(CONFIG)"
	@echo "MPB command: $(MPB_COMMAND)"
	@echo "blaze command: $(BLAZE_COMMAND)"

# ============================================================================
# Epsilon Export Targets
# ============================================================================
# These targets export the dielectric function ε(r) and inverse epsilon tensor
# for visualization and debugging. They use the first geometry from the config.
#
# Usage:
#   make export-epsilon CONFIG=config/compare_square_lowres.mk
#   make export-epsilon-tensor CONFIG=config/compare_hex_lowres.mk
#
# The EPSILON_TOML variable can be overridden in the config file to specify
# which toml configuration to use for epsilon export.

# Default epsilon config - configs can override this
EPSILON_TOML ?= $(firstword $(shell echo "$(BLAZE_COMMAND)" | grep -oP '(?<=--config )\S+'))
EPSILON_OUTPUT_PREFIX ?= $(REFERENCE_DIR)/$(EVAL_NAME)

export-epsilon:
	@if [ -z "$(strip $(EPSILON_TOML))" ]; then \
		echo "Error: Could not determine EPSILON_TOML from config"; \
		echo "Set EPSILON_TOML=path/to/config.toml explicitly"; \
		exit 1; \
	fi
	@echo "[evaluation] exporting epsilon(r) for $(EVAL_NAME)"
	@echo "  config: $(EPSILON_TOML)"
	@echo "  output: $(EPSILON_OUTPUT_PREFIX)_epsilon.csv"
	cargo run --release -p blaze2d-cli -- \
		--config $(EPSILON_TOML) \
		$(SMOOTHING_ARGS) \
		--export-epsilon $(EPSILON_OUTPUT_PREFIX)_epsilon.csv \
		--skip-solve

export-epsilon-tensor:
	@if [ -z "$(strip $(EPSILON_TOML))" ]; then \
		echo "Error: Could not determine EPSILON_TOML from config"; \
		echo "Set EPSILON_TOML=path/to/config.toml explicitly"; \
		exit 1; \
	fi
	@echo "[evaluation] exporting inverse epsilon tensor for $(EVAL_NAME)"
	@echo "  config: $(EPSILON_TOML)"
	@echo "  output: $(EPSILON_OUTPUT_PREFIX)_epsilon_tensor.csv"
	cargo run --release -p blaze2d-cli -- \
		--config $(EPSILON_TOML) \
		$(SMOOTHING_ARGS) \
		--export-epsilon-tensor $(EPSILON_OUTPUT_PREFIX)_epsilon_tensor.csv \
		--skip-solve

export-epsilon-all:
	@$(MAKE) --no-print-directory export-epsilon
	@$(MAKE) --no-print-directory export-epsilon-tensor

# ============================================================================
# MPB Epsilon Export
# ============================================================================
# Export epsilon(r) from MPB for comparison with blaze.
# This uses the generate_square_tm_bands.py script with --export-epsilon.
#
# Usage:
#   make export-epsilon-mpb CONFIG=config/compare_square_lowres.mk

# Extract parameters from config (these should match the MPB_COMMAND settings)
MPB_RESOLUTION ?= 24
MPB_RADIUS ?= 0.3
MPB_EPS_BG ?= 13.0
MPB_EPS_HOLE ?= 1.0
MPB_LATTICE ?= square

export-epsilon-mpb:
	@echo "[evaluation] exporting epsilon(r) from MPB for $(EVAL_NAME)"
	@echo "  output: $(EPSILON_OUTPUT_PREFIX)_epsilon_mpb.csv"
	mamba run -n mpb-reference python ../generate_square_tm_bands.py \
		--output /dev/null \
		--resolution $(MPB_RESOLUTION) \
		--num-bands 1 \
		--k-density 1 \
		--radius $(MPB_RADIUS) \
		--eps-bg $(MPB_EPS_BG) \
		--eps-hole $(MPB_EPS_HOLE) \
		--polarization tm \
		--lattice $(MPB_LATTICE) \
		--export-epsilon $(EPSILON_OUTPUT_PREFIX)_epsilon_mpb.csv

export-epsilon-inverse-mpb:
	@echo "[evaluation] exporting inverse epsilon tensor from MPB for $(EVAL_NAME)"
	@echo "  output: $(EPSILON_OUTPUT_PREFIX)_epsilon_inv_mpb.csv"
	mamba run -n mpb-reference python ../generate_square_tm_bands.py \
		--output /dev/null \
		--resolution $(MPB_RESOLUTION) \
		--num-bands 1 \
		--k-density 1 \
		--radius $(MPB_RADIUS) \
		--eps-bg $(MPB_EPS_BG) \
		--eps-hole $(MPB_EPS_HOLE) \
		--polarization tm \
		--lattice $(MPB_LATTICE) \
		--export-epsilon-inverse $(EPSILON_OUTPUT_PREFIX)_epsilon_inv_mpb.csv

# Export RAW inverse epsilon tensor from MPB's HDF5 (no interpolation).
# This is what the MPB eigensolver actually uses internally.
export-epsilon-inverse-h5:
	@echo "[evaluation] exporting RAW inverse epsilon tensor from MPB HDF5 for $(EVAL_NAME)"
	@echo "  output: $(EPSILON_OUTPUT_PREFIX)_epsilon_inv_h5.csv"
	mamba run -n mpb-reference python ../generate_square_tm_bands.py \
		--output /dev/null \
		--resolution $(MPB_RESOLUTION) \
		--num-bands 1 \
		--k-density 1 \
		--radius $(MPB_RADIUS) \
		--eps-bg $(MPB_EPS_BG) \
		--eps-hole $(MPB_EPS_HOLE) \
		--polarization tm \
		--lattice $(MPB_LATTICE) \
		--export-epsilon-inverse-h5 $(EPSILON_OUTPUT_PREFIX)_epsilon_inv_h5.csv

export-epsilon-compare:
	@$(MAKE) --no-print-directory export-epsilon
	@$(MAKE) --no-print-directory export-epsilon-tensor
	@$(MAKE) --no-print-directory export-epsilon-mpb
	@$(MAKE) --no-print-directory export-epsilon-inverse-mpb
	@$(MAKE) --no-print-directory export-epsilon-inverse-h5

# ============================================================================
# Iteration Trace Export Targets
# ============================================================================
# Export per-iteration convergence data (eigenvalues, residuals) to CSV for
# plotting convergence curves. Requires running the full solver.
#
# Usage:
#   make export-iterations CONFIG=config/compare_square_lowres.mk
#
# Output: {prefix}_te_iterations.csv and {prefix}_tm_iterations.csv
#   Columns: k_index, iteration, band1_eigenvalue, band1_frequency, band1_residual, ...

ITERATION_OUTPUT_PREFIX ?= $(REFERENCE_DIR)/$(EVAL_NAME)

# Extract TE and TM config paths from BLAZE_COMMAND
TE_TOML ?= $(shell echo "$(BLAZE_COMMAND)" | grep -oP '(?<=--config )\S+te\S*\.toml' | head -1)
TM_TOML ?= $(shell echo "$(BLAZE_COMMAND)" | grep -oP '(?<=--config )\S+tm\S*\.toml' | head -1)

export-iterations:
	@echo "[evaluation] exporting iteration traces for $(EVAL_NAME)"
	@if [ -n "$(strip $(TE_TOML))" ]; then \
		echo "  TE config: $(TE_TOML)"; \
		echo "  TE output: $(ITERATION_OUTPUT_PREFIX)_te_iterations.csv"; \
		cargo run --release -p blaze2d-cli -- \
			--config $(TE_TOML) \
			$(SMOOTHING_ARGS) \
			--record-diagnostics \
			--diagnostics-output /dev/null \
			--iteration-csv-combined $(ITERATION_OUTPUT_PREFIX)_te_iterations.csv \
			--path square \
			--segments-per-leg 2 \
			--quiet; \
	else \
		echo "  No TE config found in BLAZE_COMMAND"; \
	fi
	@if [ -n "$(strip $(TM_TOML))" ]; then \
		echo "  TM config: $(TM_TOML)"; \
		echo "  TM output: $(ITERATION_OUTPUT_PREFIX)_tm_iterations.csv"; \
		cargo run --release -p blaze2d-cli -- \
			--config $(TM_TOML) \
			$(SMOOTHING_ARGS) \
			--record-diagnostics \
			--diagnostics-output /dev/null \
			--iteration-csv-combined $(ITERATION_OUTPUT_PREFIX)_tm_iterations.csv \
			--path square \
			--segments-per-leg 2 \
			--quiet; \
	else \
		echo "  No TM config found in BLAZE_COMMAND"; \
	fi

clean:
	@echo "Removing generated outputs under $(REFERENCE_DIR)"
	rm -f $(REFERENCE_DIR)/*.json $(REFERENCE_DIR)/*.csv
	rm -rf $(REFERENCE_DIR)/*_pipeline

prepare_reference_data:
	@if [ "$(SKIP_REFERENCE_WIPE)" = "1" ]; then \
		echo "[evaluation] reference-data already prepared; skipping wipe"; \
	else \
		echo "[evaluation] preparing reference-data under $(REFERENCE_DIR)"; \
		mkdir -p $(REFERENCE_DIR); \
		if [ -n "$(strip $(REFERENCE_TARGETS))" ]; then \
			echo "[evaluation] removing config-scoped artifacts:"; \
			for target in $(REFERENCE_TARGETS); do \
				printf '  - %s\n' "$$target"; \
				rm -rf "$$target"; \
			done; \
		else \
			echo "[evaluation] REFERENCE_TARGETS unset; wiping entire reference-data directory"; \
			rm -f $(REFERENCE_DIR)/*.json $(REFERENCE_DIR)/*.csv; \
			rm -rf $(REFERENCE_DIR)/*_pipeline; \
		fi; \
	fi
