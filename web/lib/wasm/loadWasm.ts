/** Load the solver built from this checkout. Public assets are generated before Next builds. */
export type WasmModule = typeof import('../../public/wasm-blaze/blaze2d_backend_wasm');
let loading: Promise<WasmModule> | undefined;

export function loadWasm(basePath: string): Promise<WasmModule> {
  if (!loading) {
    loading = (async () => {
      const url = `${basePath}/wasm-blaze/blaze2d_backend_wasm.js`;
      const wasm: WasmModule = await import(/* webpackIgnore: true */ url);
      await wasm.default({ module_or_path: `${basePath}/wasm-blaze/blaze2d_backend_wasm_bg.wasm` });
      wasm.initPanicHook();
      const info = wasm.buildInfo();
      if (info.config_schema !== 'blaze2d/1' || info.result_schema !== 'blaze2d/result/1'
        || (process.env.NEXT_PUBLIC_BLAZE_REVISION && info.source_revision !== process.env.NEXT_PUBLIC_BLAZE_REVISION)
        || (process.env.NEXT_PUBLIC_BLAZE_VERSION && info.version !== process.env.NEXT_PUBLIC_BLAZE_VERSION)) {
        throw new Error('The browser solver and interface use different versions or schemas. Reload the page.');
      }
      return wasm;
    })().catch(error => { loading = undefined; throw error; });
  }
  return loading;
}
