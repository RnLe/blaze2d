.PHONY: lattice-data prepare-lattice lattice-data-square lattice-data-triangular

lattice-data: lattice-data-square lattice-data-triangular

prepare-lattice:
	@mkdir -p $(LATTICE_DIR)

lattice-data-square: prepare-lattice
	$(CARGO) run -p $(VALIDATION_BIN) -- lattice-data --lattice square --a 1.0 --output $(LATTICE_DIR)/lattice_square.json

lattice-data-triangular: prepare-lattice
	$(CARGO) run -p $(VALIDATION_BIN) -- lattice-data --lattice triangular --a 1.0 --output $(LATTICE_DIR)/lattice_triangular.json
