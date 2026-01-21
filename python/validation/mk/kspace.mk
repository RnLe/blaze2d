.PHONY: kspace-data prepare-kspace kspace-square-path kspace-triangular-path kspace-custom-mesh kspace-custom-path

kspace-data: kspace-square-path kspace-triangular-path kspace-custom-mesh kspace-custom-path

prepare-kspace:
	@mkdir -p $(KSPACE_DIR)

kspace-square-path: prepare-kspace
	$(CARGO) run -p $(VALIDATION_BIN) -- k-space-data --lattice square --mode path --path-kind square --segments-per-leg 10 --output $(KSPACE_DIR)/kspace_square_path.json

kspace-triangular-path: prepare-kspace
	$(CARGO) run -p $(VALIDATION_BIN) -- k-space-data --lattice triangular --mode path --path-kind triangular --segments-per-leg 10 --output $(KSPACE_DIR)/kspace_triangular_path.json

kspace-custom-mesh: prepare-kspace
	$(CARGO) run -p $(VALIDATION_BIN) -- k-space-data --lattice triangular --mode mesh --mesh-nx 15 --mesh-ny 11 --mesh-extent 0.65 --output $(KSPACE_DIR)/kspace_custom_mesh.json

kspace-custom-path: prepare-kspace
	$(CARGO) run -p $(VALIDATION_BIN) -- k-space-data --lattice square --mode path --path-kind custom --custom-path '0.0,0.0;0.45,0.05;0.45,0.35;0.05,0.45;-0.25,0.25;0.0,0.0' --output $(KSPACE_DIR)/kspace_custom_path.json
