.PHONY: test native python wasm website

test:
	cargo test --locked -p blaze2d-core -p blaze2d-interface -p blaze2d-runner
	python3 scripts/check-public-prose.py

native:
	cargo build --locked --release -p blaze2d-cli

python:
	python3 -m pip install ./crates/python

wasm:
	cd web && pnpm build:wasm

website:
	cd web && pnpm build
