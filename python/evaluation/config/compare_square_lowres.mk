# Evaluation-specific configuration for compare_square_lowres.
# Mirrors compare_hex_lowres but uses a square lattice reference/problem.

EVAL_NAME := compare_square_lowres
SMOOTHING_ARGS := --mesh-size 4

MPB_COMMAND := mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/square_te_eps13_r0p3_res24_k6_b8_mpb.json \
	--resolution 24 \
	--num-bands 8 \
	--k-density 6 \
	--radius 0.3 \
	--eps-bg 13.0 \
	--eps-hole 1.0 \
	--polarization te \
	--lattice square \
	&& mamba run -n mpb-reference python ../generate_square_tm_bands.py \
	--output $(REFERENCE_DIR)/square_tm_eps13_r0p3_res24_k6_b8_mpb.json \
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
	--output $(REFERENCE_DIR)/square_te_eps13_r0p3_res24_k6_b8_mpb2d.csv \
	--dump-pipeline $(REFERENCE_DIR)/square_te_eps13_r0p3_res24_k6_b8_pipeline \
	$(SMOOTHING_ARGS) \
	--preconditioner structured_diagonal \
	--path square \
	--segments-per-leg 4 \
	&& cargo run -p mpb2d-cli -- \
	--config ../../examples/square_eps13_r0p3_tm_res24.toml \
	--output $(REFERENCE_DIR)/square_tm_eps13_r0p3_res24_k6_b8_mpb2d.csv \
	--dump-pipeline $(REFERENCE_DIR)/square_tm_eps13_r0p3_res24_k6_b8_pipeline \
	$(SMOOTHING_ARGS) \
	--preconditioner structured_diagonal \
	--path square \
	--segments-per-leg 4
