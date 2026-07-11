/**
 * Shared WASM loader for worker contexts.
 *
 * Both the studio solve worker and util worker (and the examples worker) load
 * the same committed artifact from `${basePath}/wasm-blaze/`. The dynamic
 * import is deliberately `webpackIgnore`d: the glue is a plain ESM file served
 * as a static asset, not a bundled module.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WasmModule = any;

let cached: WasmModule | null = null;
let loading: Promise<WasmModule> | null = null;

export async function loadWasm(basePath: string): Promise<WasmModule> {
  if (cached) return cached;
  if (loading) return loading;

  loading = (async () => {
    const jsUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm.js`;
    const binUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm_bg.wasm`;
    const mod = await import(/* webpackIgnore: true */ jsUrl);
    await mod.default(binUrl);
    if (mod.initPanicHook) mod.initPanicHook();
    cached = mod;
    return mod;
  })();

  return loading;
}

/** Recursively convert a wasm-bindgen JS value into a structured-clonable plain object. */
export function toPlain(value: unknown): unknown {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(toPlain);
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value as Record<string, unknown>)) {
    out[key] = toPlain((value as Record<string, unknown>)[key]);
  }
  return out;
}
