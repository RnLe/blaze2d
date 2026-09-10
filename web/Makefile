.PHONY: build dev test serve wasm clean
build:
	pnpm build
dev:
	pnpm dev
test:
	pnpm test:browser
serve:
	pnpm serve:out
wasm:
	pnpm build:wasm
clean:
	rm -rf .next out dist public/wasm-blaze
	rm -f *.tsbuildinfo
