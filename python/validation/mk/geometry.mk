.PHONY: geometry-data prepare-geometry \
	geometry-square-res24 geometry-square-res48 geometry-square-res128 \
	geometry-tri-res24 geometry-tri-res48 geometry-tri-res128

geometry-data: geometry-square-res24 geometry-square-res48 geometry-square-res128 \
	geometry-tri-res24 geometry-tri-res48 geometry-tri-res128

prepare-geometry:
	@mkdir -p $(GEOMETRY_DIR)

geometry-square-res24: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice square --resolution 24 --radius 0.28 --eps-bg 11.0 --eps-inside 1.0 --output $(GEOMETRY_DIR)/geometry_square_res24_bg11p0_hole1p0.json

geometry-square-res48: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice square --resolution 48 --radius 0.25 --eps-bg 8.5 --eps-inside 2.0 --output $(GEOMETRY_DIR)/geometry_square_res48_bg8p5_hole2p0.json

geometry-square-res128: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice square --resolution 128 --radius 0.2 --eps-bg 16.0 --eps-inside 4.0 --output $(GEOMETRY_DIR)/geometry_square_res128_bg16p0_hole4p0.json

geometry-tri-res24: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice triangular --resolution 24 --radius 0.22 --eps-bg 10.5 --eps-inside 3.5 --output $(GEOMETRY_DIR)/geometry_triangular_res24_bg10p5_hole3p5.json

geometry-tri-res48: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice triangular --resolution 48 --radius 0.18 --eps-bg 6.0 --eps-inside 1.5 --output $(GEOMETRY_DIR)/geometry_triangular_res48_bg6p0_hole1p5.json

geometry-tri-res128: prepare-geometry
	$(CARGO) run -p $(VALIDATION_BIN) -- geometry-data --lattice triangular --resolution 128 --radius 0.15 --eps-bg 14.0 --eps-inside 5.5 --output $(GEOMETRY_DIR)/geometry_triangular_res128_bg14p0_hole5p5.json
