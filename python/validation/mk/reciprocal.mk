.PHONY: reciprocal-data prepare-reciprocal \
	reciprocal-square-res32-mesh4 \
	reciprocal-tri-res32-mesh4

reciprocal-data: reciprocal-square-res32-mesh4 reciprocal-tri-res32-mesh4

prepare-reciprocal:
	@mkdir -p $(RECIPROCAL_DIR)

reciprocal-square-res32-mesh4: prepare-reciprocal
	$(CARGO) run -p $(VALIDATION_BIN) -- reciprocal-data --lattice square --resolution 32 --mesh-size 4 --points-per-leg 3 --output $(RECIPROCAL_DIR)/reciprocal_square_res32_mesh4.json

reciprocal-tri-res32-mesh4: prepare-reciprocal
	$(CARGO) run -p $(VALIDATION_BIN) -- reciprocal-data --lattice triangular --resolution 32 --mesh-size 4 --points-per-leg 3 --output $(RECIPROCAL_DIR)/reciprocal_triangular_res32_mesh4.json
