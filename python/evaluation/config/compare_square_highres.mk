# Evaluation-specific configuration for compare_square_highres.
# High resolution: eps_bg=6, r/a=0.3, res=64, k=10/leg, 8 bands

EVAL_NAME := compare_square_highres
SMOOTHING_ARGS := --mesh-size 4

SQUARE_DESC := eps6_r0p3_res64_k10_b8
SQUARE_TE_PREFIX := $(REFERENCE_DIR)/square_te_$(SQUARE_DESC)
SQUARE_TM_PREFIX := $(REFERENCE_DIR)/square_tm_$(SQUARE_DESC)

REFERENCE_TARGETS := \
	$(SQUARE_TE_PREFIX)_mpb.json \
	$(SQUARE_TM_PREFIX)_mpb.json \
	$(SQUARE_TE_PREFIX)_mpb2d.csv \
	$(SQUARE_TM_PREFIX)_mpb2d.csv

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(SQUARE_TE_PREFIX)_mpb.json \
	--resolution 64 \
	--num-bands 8 \
	--k-density 10 \
	--radius 0.3 \
	--eps-bg 6.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice square \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(SQUARE_TM_PREFIX)_mpb.json \
	--resolution 64 \
	--num-bands 8 \
	--k-density 10 \
	--radius 0.3 \
	--eps-bg 6.0 \
	--eps-hole 1.0 \
	--polarization tm \
	--lattice square


MPB2D_COMMAND := cargo run --release -p mpb2d-cli -- \
	--config ../../examples/square_eps6_r0p3_te_res64.toml \
	--output $(SQUARE_TE_PREFIX)_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10 \
	&& cargo run --release -p mpb2d-cli -- \
	--config ../../examples/square_eps6_r0p3_tm_res64.toml \
	--output $(SQUARE_TM_PREFIX)_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path square \
	--segments-per-leg 10
