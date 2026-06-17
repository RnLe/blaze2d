# Makefile for cleaning up the repo and set it to a fresh-clone state
# Run in root of the repo with sudo:

# sudo make clean

# Run 'pnpm i' to install dependencies again
# Run 'pnpm build' to build the project again

# Folders to blow away
CLEAN_DIRS := \
  node_modules \
  .next \
  out \
  dist \
  public/wasm

# Files to blow away
CLEAN_FILES := \
  *.tsbuildinfo

.PHONY: clean
clean:
	@echo "🧹 Cleaning all build artifacts and installs…"
	@for d in $(CLEAN_DIRS); do \
	  sudo rm -rf $$d; \
	done
	@for f in $(CLEAN_FILES); do \
	  sudo rm -f $$f; \
	done
	@echo "✅  Done. Repo is now back to a fresh-clone state."

# Serve the website's out/ folder on 3001
.PHONY: serve
serve:
	pnpm run serve:out

# Completely clean the repo, install dependencies, build, and serve
.PHONY: dev
dev: clean
	pnpm i && pnpm run build && pnpm run serve:out

# WASM Build Pipeline
# ===================

# Build WASM module into public/wasm-blaze
.PHONY: build-wasm
build-wasm:
	@echo "🦀 Building WASM module..."
	pnpm run build:wasm
	@echo "✅ WASM module built and copied successfully!"

# Build WASM and install dependencies
.PHONY: build-wasm-dev
build-wasm-dev: build-wasm
	@echo "📦 Installing dependencies..."
	pnpm install
	@echo "✅ WASM build and dependencies ready for development!"

# Clean WASM artifacts from website
.PHONY: clean-wasm
clean-wasm:
	@echo "🧹 Cleaning WASM artifacts from website..."
	pnpm run clean:wasm
	@echo "✅ WASM artifacts cleaned!"

# Full development build including WASM
.PHONY: dev-with-wasm
dev-with-wasm: clean build-wasm-dev
	pnpm run build && pnpm run serve:out

# Python Build Pipeline
# ======================

# Build Python bindings using maturin
.PHONY: build-python
build-python:
	@echo "🐍 Building Python bindings..."
	cd ../rust-python && maturin develop --release
	@echo "✅ Python bindings built successfully!"

# Build Python bindings for development (debug mode)
.PHONY: build-python-dev
build-python-dev:
	@echo "🐍 Building Python bindings (debug mode)..."
	cd ../rust-python && maturin develop
	@echo "✅ Python bindings built successfully in debug mode!"

# Install Python dependencies and build bindings
.PHONY: python
python: build-python
	@echo "🐍 Python bindings ready!"

# Clean Python build artifacts
.PHONY: clean-python
clean-python:
	@echo "🧹 Cleaning Python build artifacts..."
	cd ../rust-python && cargo clean
	cd ../rust-python && rm -rf target/
	cd ../rust-python && find . -name "*.so" -delete
	cd ../rust-python && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	cd ../rust-python && find . -name "*.pyc" -delete
	@echo "✅ Python build artifacts cleaned!"
