# Evaluation-specific configuration for compare_square_lowres.
# Mirrors compare_hex_lowres but uses a square lattice reference/problem.

EVAL_NAME := compare_square_lowres
SMOOTHING_ARGS := --mesh-size 4

SQUARE_DESC := eps13_r0p3_res24_k6_b8
SQUARE_TE_PREFIX := $(REFERENCE_DIR)/square_te_$(SQUARE_DESC)
SQUARE_TM_PREFIX := $(REFERENCE_DIR)/square_tm_$(SQUARE_DESC)

REFERENCE_TARGETS := \
	$(SQUARE_TE_PREFIX)_mpb.json \
	$(SQUARE_TM_PREFIX)_mpb.json \
	$(SQUARE_TE_PREFIX)_mpb2d.csv \
	$(SQUARE_TM_PREFIX)_mpb2d.csv \
	$(SQUARE_TE_PREFIX)_pipeline \
	$(SQUARE_TM_PREFIX)_pipeline

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(SQUARE_TE_PREFIX)_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 6 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice square \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(SQUARE_TM_PREFIX)_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 6 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization tm \
	--lattice square


MPB2D_COMMAND := cargo run -p mpb2d-cli -- \
	--config ../../examples/square_eps13_r0p3_te_res24.toml \
	--output $(SQUARE_TE_PREFIX)_mpb2d.csv \
	--dump-pipeline $(SQUARE_TE_PREFIX)_pipeline \
	$(SMOOTHING_ARGS) \
	--preconditioner structured_diagonal \
	--path square \
	--segments-per-leg 4 \
	&& cargo run -p mpb2d-cli -- \
	--config ../../examples/square_eps13_r0p3_tm_res24.toml \
	--output $(SQUARE_TM_PREFIX)_mpb2d.csv \
	--dump-pipeline $(SQUARE_TM_PREFIX)_pipeline \
	$(SMOOTHING_ARGS) \
	--preconditioner structured_diagonal \
	--path square \
	--segments-per-leg 4
