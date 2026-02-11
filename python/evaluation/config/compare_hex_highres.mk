# Evaluation-specific configuration for compare_hex_highres.
# Higher resolution version: 48x48 grid, 8 k-points per leg.

EVAL_NAME := compare_hex_highres
SMOOTHING_ARGS := --mesh-size 4

REFERENCE_TARGETS := \
	$(REFERENCE_DIR)/hex_te_eps13_r0p3_res48_k8_b8_mpb.json \
	$(REFERENCE_DIR)/hex_tm_eps13_r0p3_res48_k8_b8_mpb.json \
	$(REFERENCE_DIR)/hex_te_eps13_r0p3_res48_k8_b8_mpb2d.csv \
	$(REFERENCE_DIR)/hex_tm_eps13_r0p3_res48_k8_b8_mpb2d.csv

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/hex_te_eps13_r0p3_res48_k8_b8_mpb.json \
	--resolution 48 \
	--num-bands 8 \
	--k-density 8 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice hexagonal \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/hex_tm_eps13_r0p3_res48_k8_b8_mpb.json \
	--resolution 48 \
	--num-bands 8 \
	--k-density 8 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization tm \
	--lattice hexagonal


MPB2D_COMMAND := cargo run --release -p mpb2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_te_res48.toml \
	--output $(REFERENCE_DIR)/hex_te_eps13_r0p3_res48_k8_b8_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 8 \
	&& cargo run --release -p mpb2d-cli -- \
	--config ../../examples/hex_eps13_r0p3_tm_res48.toml \
	--output $(REFERENCE_DIR)/hex_tm_eps13_r0p3_res48_k8_b8_mpb2d.csv \
	$(SMOOTHING_ARGS) \
	--path hexagonal \
	--segments-per-leg 8
