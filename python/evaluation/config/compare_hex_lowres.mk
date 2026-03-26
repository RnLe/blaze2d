# Evaluation-specific configuration for compare_hex_lowres.
# Mirrors compare_square_lowres but uses a hexagonal lattice reference/problem.

EVAL_NAME := compare_hex_lowres
SMOOTHING_ARGS := --mesh-size 4 

HEX_DESC := eps13_r0p3_res24_k10_b8
HEX_TE_PREFIX := $(REFERENCE_DIR)/hex_te_$(HEX_DESC)
HEX_TM_PREFIX := $(REFERENCE_DIR)/hex_tm_$(HEX_DESC)

# MPB parameters for epsilon export
MPB_RESOLUTION := 24
MPB_RADIUS := 0.3
MPB_EPS_BG := 13.0
MPB_EPS_HOLE := 1.0
MPB_LATTICE := hexagonal

REFERENCE_TARGETS := \
	$(HEX_TE_PREFIX)_mpb.json \
	$(HEX_TM_PREFIX)_mpb.json \
	$(HEX_TE_PREFIX)_blaze.csv \
	$(HEX_TM_PREFIX)_blaze.csv

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(HEX_TE_PREFIX)_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 10 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice hexagonal \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(HEX_TM_PREFIX)_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 10 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization tm \
	--lattice hexagonal

# Split commands for potential future benchmark support
BLAZE_TE_CMD := cargo run --release -p blaze2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_te_res24.toml \
	--output $(HEX_TE_PREFIX)_blaze.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 10

BLAZE_TM_CMD := cargo run --release -p blaze2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_tm_res24.toml \
	--output $(HEX_TM_PREFIX)_blaze.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 10

# Combined command
BLAZE_COMMAND := $(BLAZE_TE_CMD) && $(BLAZE_TM_CMD)
