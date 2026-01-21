.PHONY: precompute-data prepare-precompute \
	precompute-square-res24-mesh1 precompute-square-res24-mesh4 \
	precompute-square-res48-mesh1 precompute-square-res48-mesh4 \
	precompute-square-res128-mesh1 precompute-square-res128-mesh4 \
	precompute-tri-res24-mesh1 precompute-tri-res24-mesh4 \
	precompute-tri-res48-mesh1 precompute-tri-res48-mesh4 \
	precompute-tri-res128-mesh1 precompute-tri-res128-mesh4

precompute-data: \
	precompute-square-res24-mesh1 precompute-square-res24-mesh4 \
	precompute-square-res48-mesh1 precompute-square-res48-mesh4 \
	precompute-square-res128-mesh1 precompute-square-res128-mesh4 \
	precompute-tri-res24-mesh1 precompute-tri-res24-mesh4 \
	precompute-tri-res48-mesh1 precompute-tri-res48-mesh4 \
	precompute-tri-res128-mesh1 precompute-tri-res128-mesh4

prepare-precompute:
	@mkdir -p $(PRECOMPUTE_DIR)

precompute-square-res24-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_square_res24_bg11p0_hole1p0_mesh1.json --tag square_res24_mesh1

precompute-square-res24-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_square_res24_bg11p0_hole1p0_mesh4.json --tag square_res24_mesh4

precompute-square-res48-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_square_res48_bg8p5_hole2p0_mesh1.json --tag square_res48_mesh1

precompute-square-res48-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_square_res48_bg8p5_hole2p0_mesh4.json --tag square_res48_mesh4

precompute-square-res128-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 128 --radius 0.20 --eps-bg 16.0 --eps-inside 4.0 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_square_res128_bg16p0_hole4p0_mesh1.json --tag square_res128_mesh1

precompute-square-res128-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice square --resolution 128 --radius 0.20 --eps-bg 16.0 --eps-inside 4.0 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_square_res128_bg16p0_hole4p0_mesh4.json --tag square_res128_mesh4

precompute-tri-res24-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_triangular_res24_bg10p5_hole3p5_mesh1.json --tag triangular_res24_mesh1

precompute-tri-res24-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_triangular_res24_bg10p5_hole3p5_mesh4.json --tag triangular_res24_mesh4

precompute-tri-res48-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_triangular_res48_bg6p0_hole1p5_mesh1.json --tag triangular_res48_mesh1

precompute-tri-res48-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_triangular_res48_bg6p0_hole1p5_mesh4.json --tag triangular_res48_mesh4

precompute-tri-res128-mesh1: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --mesh-size 1 --output $(PRECOMPUTE_DIR)/precompute_triangular_res128_bg14p0_hole5p5_mesh1.json --tag triangular_res128_mesh1

precompute-tri-res128-mesh4: prepare-precompute
	$(CARGO) run -p $(VALIDATION_BIN) -- precompute-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --mesh-size 4 --output $(PRECOMPUTE_DIR)/precompute_triangular_res128_bg14p0_hole5p5_mesh4.json --tag triangular_res128_mesh4
