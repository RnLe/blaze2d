.PHONY: smoothing-data prepare-smoothing \
	smoothing-square-res24 smoothing-square-res48 smoothing-square-res128 \
	smoothing-square-res24-mesh1 smoothing-square-res24-mesh4 smoothing-square-res24-mesh12 \
	smoothing-square-res48-mesh1 smoothing-square-res48-mesh4 smoothing-square-res48-mesh12 \
	smoothing-square-res128-mesh1 smoothing-square-res128-mesh4 smoothing-square-res128-mesh12 \
	smoothing-tri-res24 smoothing-tri-res48 smoothing-tri-res128 \
	smoothing-tri-res24-mesh1 smoothing-tri-res24-mesh4 smoothing-tri-res24-mesh12 \
	smoothing-tri-res48-mesh1 smoothing-tri-res48-mesh4 smoothing-tri-res48-mesh12 \
	smoothing-tri-res128-mesh1 smoothing-tri-res128-mesh4 smoothing-tri-res128-mesh12

smoothing-data: smoothing-square-res24 smoothing-square-res48 smoothing-square-res128 \
	smoothing-tri-res24 smoothing-tri-res48 smoothing-tri-res128

prepare-smoothing:
	@mkdir -p $(SMOOTHING_DIR)

smoothing-square-res24: smoothing-square-res24-mesh1 smoothing-square-res24-mesh4 smoothing-square-res24-mesh12

smoothing-square-res24-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_square_res24_bg11p0_hole1p0_mesh1.json

smoothing-square-res24-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_square_res24_bg11p0_hole1p0_mesh4.json

smoothing-square-res24-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_square_res24_bg11p0_hole1p0_mesh12.json

smoothing-square-res48: smoothing-square-res48-mesh1 smoothing-square-res48-mesh4 smoothing-square-res48-mesh12

smoothing-square-res48-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_square_res48_bg8p5_hole2p0_mesh1.json

smoothing-square-res48-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_square_res48_bg8p5_hole2p0_mesh4.json

smoothing-square-res48-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_square_res48_bg8p5_hole2p0_mesh12.json

smoothing-square-res128: smoothing-square-res128-mesh1 smoothing-square-res128-mesh4 smoothing-square-res128-mesh12

smoothing-square-res128-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 128 --radius 0.2 --eps-bg 16.0 --eps-inside 4.0 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_square_res128_bg16p0_hole4p0_mesh1.json

smoothing-square-res128-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 128 --radius 0.2 --eps-bg 16.0 --eps-inside 4.0 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_square_res128_bg16p0_hole4p0_mesh4.json

smoothing-square-res128-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice square --resolution 128 --radius 0.2 --eps-bg 16.0 --eps-inside 4.0 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_square_res128_bg16p0_hole4p0_mesh12.json

smoothing-tri-res24: smoothing-tri-res24-mesh1 smoothing-tri-res24-mesh4 smoothing-tri-res24-mesh12

smoothing-tri-res24-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_triangular_res24_bg10p5_hole3p5_mesh1.json

smoothing-tri-res24-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_triangular_res24_bg10p5_hole3p5_mesh4.json

smoothing-tri-res24-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_triangular_res24_bg10p5_hole3p5_mesh12.json

smoothing-tri-res48: smoothing-tri-res48-mesh1 smoothing-tri-res48-mesh4 smoothing-tri-res48-mesh12

smoothing-tri-res48-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_triangular_res48_bg6p0_hole1p5_mesh1.json

smoothing-tri-res48-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_triangular_res48_bg6p0_hole1p5_mesh4.json

smoothing-tri-res48-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_triangular_res48_bg6p0_hole1p5_mesh12.json

smoothing-tri-res128: smoothing-tri-res128-mesh1 smoothing-tri-res128-mesh4 smoothing-tri-res128-mesh12

smoothing-tri-res128-mesh1: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --mesh-size 1 --output $(SMOOTHING_DIR)/smoothing_triangular_res128_bg14p0_hole5p5_mesh1.json

smoothing-tri-res128-mesh4: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --mesh-size 4 --output $(SMOOTHING_DIR)/smoothing_triangular_res128_bg14p0_hole5p5_mesh4.json

smoothing-tri-res128-mesh12: prepare-smoothing
	$(CARGO) run -p $(VALIDATION_BIN) -- smoothing-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --mesh-size 12 --output $(SMOOTHING_DIR)/smoothing_triangular_res128_bg14p0_hole5p5_mesh12.json
