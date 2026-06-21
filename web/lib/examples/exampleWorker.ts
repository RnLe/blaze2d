/**
 * Web Worker for running example computations via the WASM bulk driver.
 *
 * Runs off the main thread so the UI stays responsive and so a long
 * computation can be aborted by terminating the worker (the single-threaded
 * WASM call cannot otherwise be interrupted mid-flight).
 *
 * Two modes are supported:
 *  - `stream`   → `runWithKPointStreaming`, emitting one message per k-point
 *                 (preferred for live band-diagram rendering).
 *  - `filtered` → `runStreamingFiltered`, emitting one job-level result per
 *                 job (used for the "selective k-points & bands" example).
 */

export interface WorkerInitMessage {
  type: 'init';
  basePath: string;
}

export interface WorkerRunMessage {
  type: 'run';
  configToml: string;
  mode: 'stream' | 'filtered';
  kIndices?: number[] | null;
  bandIndices?: number[] | null;
}

export type ExampleWorkerInMessage = WorkerInitMessage | WorkerRunMessage;

export interface WorkerReadyMessage {
  type: 'ready';
}

/** A raw k-point streaming dict (matches the WASM `k_result_to_js` shape). */
export interface WorkerKPointMessage {
  type: 'kpoint';
  data: Record<string, unknown>;
}

/** A raw job-level result dict (matches the WASM `result_to_js` shape). */
export interface WorkerResultMessage {
  type: 'result';
  data: Record<string, unknown>;
}

export interface WorkerDoneMessage {
  type: 'done';
  stats: Record<string, unknown>;
  totalTime: number;
}

export interface WorkerErrorMessage {
  type: 'error';
  message: string;
}

export type ExampleWorkerOutMessage =
  | WorkerReadyMessage
  | WorkerKPointMessage
  | WorkerResultMessage
  | WorkerDoneMessage
  | WorkerErrorMessage;

// ---------------------------------------------------------------------------
// Worker implementation
// ---------------------------------------------------------------------------

const isWorker = typeof self !== 'undefined' && typeof Window === 'undefined';

if (isWorker) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let wasmModule: any = null;
  let isInitialized = false;
  let basePath = '';

  async function initWasm(): Promise<void> {
    if (isInitialized) return;
    try {
      const wasmJsUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm.js`;
      const wasmBinaryUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm_bg.wasm`;
      const mod = await import(/* webpackIgnore: true */ wasmJsUrl);
      await mod.default(wasmBinaryUrl);
      if (mod.initPanicHook) mod.initPanicHook();
      wasmModule = mod;
      isInitialized = true;
      self.postMessage({ type: 'ready' } as WorkerReadyMessage);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({
        type: 'error',
        message: `Failed to init WASM: ${message}`,
      } as WorkerErrorMessage);
    }
  }

  /** Recursively convert a wasm-bindgen JS value into a structured-clonable plain object. */
  function toPlain(value: unknown): unknown {
    if (value === null || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(toPlain);
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>)) {
      out[key] = toPlain((value as Record<string, unknown>)[key]);
    }
    return out;
  }

  function run(msg: WorkerRunMessage): void {
    if (!wasmModule) {
      self.postMessage({ type: 'error', message: 'WASM not initialized' } as WorkerErrorMessage);
      return;
    }
    const startTime = performance.now();
    try {
      const driver = new wasmModule.WasmBulkDriver(msg.configToml);
      let stats: unknown;

      if (msg.mode === 'filtered') {
        stats = driver.runStreamingFiltered(
          msg.kIndices ?? null,
          msg.bandIndices ?? null,
          (result: unknown) => {
            self.postMessage({
              type: 'result',
              data: toPlain(result) as Record<string, unknown>,
            } as WorkerResultMessage);
          },
        );
      } else {
        stats = driver.runWithKPointStreaming((kResult: unknown) => {
          self.postMessage({
            type: 'kpoint',
            data: toPlain(kResult) as Record<string, unknown>,
          } as WorkerKPointMessage);
        });
      }

      self.postMessage({
        type: 'done',
        stats: toPlain(stats) as Record<string, unknown>,
        totalTime: performance.now() - startTime,
      } as WorkerDoneMessage);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({ type: 'error', message } as WorkerErrorMessage);
    }
  }

  self.onmessage = async (event: MessageEvent<ExampleWorkerInMessage>) => {
    const msg = event.data;
    switch (msg.type) {
      case 'init':
        basePath = msg.basePath || '';
        await initWasm();
        break;
      case 'run':
        if (!isInitialized) await initWasm();
        run(msg);
        break;
    }
  };
}
