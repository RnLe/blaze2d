# MPB-GPU-2D Makefile
# Build targets for WASM and other artifacts

# Configuration
WASM_CRATE := mpb2d-backend-wasm
WASM_OUT := wasm-dist
WASM_FEATURES := streaming

# wasm-pack build profile (release for production)
WASM_PROFILE := --release

.PHONY: wasm wasm-dev wasm-clean help

# =============================================================================
# WASM Targets
# =============================================================================

## Build WASM bindings for web (production)
## Output: ./wasm-dist/ - copy this folder to your Next.js public/ directory
wasm:
	@echo "🔨 Building WASM bindings (release)..."
	wasm-pack build crates/backend-wasm \
		--target web \
		--out-dir ../../$(WASM_OUT) \
		$(WASM_PROFILE) \
		-- --features $(WASM_FEATURES)
	@# Clean up unnecessary files
	@rm -f $(WASM_OUT)/.gitignore $(WASM_OUT)/package.json $(WASM_OUT)/README.md
	@echo ""
	@echo "WASM build complete!"
	@echo ""
	@echo "Output: ./$(WASM_OUT)/"
	@echo ""
	@echo "Usage in Next.js:"
	@echo "  1. Copy $(WASM_OUT)/ contents to public/wasm/"
	@echo "  2. Import in your component:"
	@echo ""
	@echo "     import init, { WasmBulkDriver } from '/wasm/mpb2d_backend_wasm.js';"
	@echo ""
	@echo "     async function run() {"
	@echo "       await init();"
	@echo "       const driver = new WasmBulkDriver(tomlConfig);"
	@echo "       driver.runWithCallback((result) => { ... });"
	@echo "     }"
	@echo ""

## Build WASM bindings (debug, faster compilation)
wasm-dev:
	@echo "Building WASM bindings (debug)..."
	wasm-pack build crates/backend-wasm \
		--target web \
		--out-dir ../../$(WASM_OUT) \
		--dev \
		-- --features $(WASM_FEATURES)
	@rm -f $(WASM_OUT)/.gitignore $(WASM_OUT)/package.json $(WASM_OUT)/README.md
	@echo "WASM debug build complete: ./$(WASM_OUT)/"

## Clean WASM build artifacts
wasm-clean:
	@echo "Cleaning WASM artifacts..."
	rm -rf $(WASM_OUT)
	rm -rf crates/backend-wasm/pkg
	@echo "Clean complete"

# =============================================================================
# General Targets
# =============================================================================

## Show this help
help:
	@echo "MPB-GPU-2D Build System"
	@echo ""
	@echo "WASM Targets:"
	@echo "  make wasm        Build production WASM (./$(WASM_OUT)/)"
	@echo "  make wasm-dev    Build debug WASM (faster)"
	@echo "  make wasm-clean  Remove WASM build artifacts"
	@echo ""
	@echo "Output can be copied to Next.js public/ folder:"
	@echo "  cp -r $(WASM_OUT)/* my-nextjs-app/public/wasm/"
	@echo ""

# Default target
.DEFAULT_GOAL := help
