# Evaluation-specific configuration for compare_hex_lowres.
# Loaded by python/evaluation/Makefile to orchestrate MPB + mpb2d runs.

EVAL_NAME := compare_hex_lowres
SMOOTHING_ARGS := --mesh-size 4

REFERENCE_TARGETS := \
	$(REFERENCE_DIR)/hex_te_eps13_r0p3_res24_k6_b8_mpb.json \
	$(REFERENCE_DIR)/hex_tm_eps13_r0p3_res24_k6_b8_mpb.json \
	$(REFERENCE_DIR)/hex_te_eps13_r0p3_res24_k6_b8_mpb2d.csv \
	$(REFERENCE_DIR)/hex_tm_eps13_r0p3_res24_k6_b8_mpb2d.csv

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/hex_te_eps13_r0p3_res24_k6_b8_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 6 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice hexagonal \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/hex_tm_eps13_r0p3_res24_k6_b8_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 6 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization tm \
	--lattice hexagonal


MPB2D_COMMAND := cargo run --release -p mpb2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_te_res24.toml \
	--output $(REFERENCE_DIR)/hex_te_eps13_r0p3_res24_k6_b8_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 4 \
	&& cargo run --release -p mpb2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_tm_res24.toml \
	--output $(REFERENCE_DIR)/hex_tm_eps13_r0p3_res24_k6_b8_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 4
