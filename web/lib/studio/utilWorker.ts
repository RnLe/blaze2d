/**
 * Studio util worker: authoritative config validation via WASM.
 *
 * Kept separate from the solve worker so validation stays responsive even
 * while a (single-threaded) solve is running in the solve worker. Calls the
 * real Rust parser through `validateConfig`, so the diagnostics the editor
 * shows are exactly the ones the driver would raise: zero drift.
 */

import { loadWasm, toPlain, type WasmModule } from '../wasm/loadWasm';

export interface UtilInitMessage {
  type: 'init';
  basePath: string;
}

export interface UtilValidateMessage {
  type: 'validate';
  /** Monotonic id so stale responses can be dropped. */
  id: number;
  configToml: string;
}

export type UtilWorkerInMessage = UtilInitMessage | UtilValidateMessage;

export interface UtilReadyMessage {
  type: 'ready';
}

export interface Diagnostic {
  path: string;
  message: string;
  span: [number, number] | null;
}

export interface ValidationSummary {
  jobs: number;
  nx: number;
  ny: number;
  n_bands: number;
  k_points: number;
  precision: string;
  solver_type: string;
  polarization: string;
}

export interface UtilValidatedMessage {
  type: 'validated';
  id: number;
  ok: boolean;
  errors: Diagnostic[];
  summary?: ValidationSummary;
}

export type UtilWorkerOutMessage = UtilReadyMessage | UtilValidatedMessage;

const isWorker = typeof self !== 'undefined' && typeof Window === 'undefined';

if (isWorker) {
  let wasm: WasmModule | null = null;
  let basePath = '';

  async function ensureWasm(): Promise<void> {
    if (wasm) return;
    wasm = await loadWasm(basePath);
    self.postMessage({ type: 'ready' } as UtilReadyMessage);
  }

  function validate(msg: UtilValidateMessage): void {
    if (!wasm) return;
    try {
      const report = toPlain(wasm.validateConfig(msg.configToml)) as {
        ok: boolean;
        errors: Diagnostic[];
        summary?: ValidationSummary;
      };
      self.postMessage({
        type: 'validated',
        id: msg.id,
        ok: report.ok,
        errors: report.errors ?? [],
        summary: report.summary,
      } as UtilValidatedMessage);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({
        type: 'validated',
        id: msg.id,
        ok: false,
        errors: [{ path: '', message, span: null }],
      } as UtilValidatedMessage);
    }
  }

  self.onmessage = async (event: MessageEvent<UtilWorkerInMessage>) => {
    const msg = event.data;
    switch (msg.type) {
      case 'init':
        basePath = msg.basePath || '';
        await ensureWasm();
        break;
      case 'validate':
        if (!wasm) await ensureWasm();
        validate(msg);
        break;
    }
  };
}
