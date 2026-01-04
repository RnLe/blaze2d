# Evaluation-specific configuration for compare_square_lowres.
# Mirrors compare_hex_lowres but uses a square lattice reference/problem.

EVAL_NAME := compare_square_lowres
SMOOTHING_ARGS := --mesh-size 4

SQUARE_DESC := eps13_r0p3_res24_k6_b8
SQUARE_TE_PREFIX := $(REFERENCE_DIR)/square_te_$(SQUARE_DESC)
SQUARE_TM_PREFIX := $(REFERENCE_DIR)/square_tm_$(SQUARE_DESC)

# MPB parameters for epsilon export
MPB_RESOLUTION := 24
MPB_RADIUS := 0.3
MPB_EPS_BG := 13.0
MPB_EPS_HOLE := 1.0
MPB_LATTICE := square

# Benchmark tracking
BENCHMARK_YAML := $(EVAL_ROOT)/benchmark_square.yaml
BENCHMARK_LOG_TE := /tmp/blaze_benchmark_te.log
BENCHMARK_LOG_TM := /tmp/blaze_benchmark_tm.log

REFERENCE_TARGETS := \
	$(SQUARE_TE_PREFIX)_mpb.json \
	$(SQUARE_TM_PREFIX)_mpb.json \
	$(SQUARE_TE_PREFIX)_blaze.csv \
	$(SQUARE_TM_PREFIX)_blaze.csv \
	$(SQUARE_TE_PREFIX)_blaze_cuda.csv \
	$(SQUARE_TM_PREFIX)_blaze_cuda.csv

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
        --output $(SQUARE_TE_PREFIX)_mpb.json \
        --resolution 24 \
        --num-bands 8 \
        --k-density 10 \
        --radius 0.3 \
        --eps-bg 13.0 \
        --eps-hole 1.0 \
        --polarization te \
        --lattice square \
        && mamba run -n mpb-reference python ../generate_square_tm_bands.py \
        --output $(SQUARE_TM_PREFIX)_mpb.json \
        --resolution 24 \
        --num-bands 8 \
        --k-density 10 \
        --radius 0.3 \
        --eps-bg 13.0 \
        --eps-hole 1.0 \
        --polarization tm \
        --lattice square

# Split commands for benchmark logging
BLAZE_TE_CMD := cargo run --release -p blaze2d-cli -- \
	--config ../../examples/square_eps13_r0p3_te_res24.toml \
	--output $(SQUARE_TE_PREFIX)_blaze.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10

BLAZE_TM_CMD := cargo run --release -p blaze2d-cli -- \
	--config ../../examples/square_eps13_r0p3_tm_res24.toml \
	--output $(SQUARE_TM_PREFIX)_blaze.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10

# CUDA-enabled commands (same config, but with --features cuda)
BLAZE_TE_CUDA_CMD := cargo run --release --features cuda -p blaze2d-cli -- \
	--config ../../examples/square_eps13_r0p3_te_res24.toml \
	--output $(SQUARE_TE_PREFIX)_blaze_cuda.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10

BLAZE_TM_CUDA_CMD := cargo run --release --features cuda -p blaze2d-cli -- \
	--config ../../examples/square_eps13_r0p3_tm_res24.toml \
	--output $(SQUARE_TM_PREFIX)_blaze_cuda.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10

# Combined command (for backward compatibility)
BLAZE_COMMAND := $(BLAZE_TE_CMD) && $(BLAZE_TM_CMD)

# CUDA combined command
BLAZE_CUDA_COMMAND := $(BLAZE_TE_CUDA_CMD) && $(BLAZE_TM_CUDA_CMD)
