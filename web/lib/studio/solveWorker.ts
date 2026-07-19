/**
 * Studio solve worker.
 *
 * Runs the WASM bulk driver off the main thread so a long (possibly multi-job)
 * solve can be aborted by terminating the worker. Emits one message per
 * k-point (with job context) so the run panel can render band diagrams
 * progressively and label results per sweep job.
 */

import { loadWasm, toPlain, type WasmModule } from '../wasm/loadWasm';

export interface SolveInitMessage {
  type: 'init';
  basePath: string;
}

export interface SolveRunMessage {
  type: 'run';
  configToml: string;
}

export type SolveWorkerInMessage = SolveInitMessage | SolveRunMessage;

export interface SolveReadyMessage {
  type: 'ready';
}

export interface SolveMetaMessage {
  type: 'meta';
  totalJobs: number;
  solverType: string;
  precision: string;
}

export interface SolveKPointMessage {
  type: 'kpoint';
  data: Record<string, unknown>;
}

export interface SolveDoneMessage {
  type: 'done';
  stats: Record<string, unknown>;
  totalTime: number;
}

export interface SolveErrorMessage {
  type: 'error';
  message: string;
}

export type SolveWorkerOutMessage =
  | SolveReadyMessage
  | SolveMetaMessage
  | SolveKPointMessage
  | SolveDoneMessage
  | SolveErrorMessage;

const isWorker = typeof self !== 'undefined' && typeof Window === 'undefined';

if (isWorker) {
  let wasm: WasmModule | null = null;
  let basePath = '';

  async function ensureWasm(): Promise<void> {
    if (wasm) return;
    wasm = await loadWasm(basePath);
    self.postMessage({ type: 'ready' } as SolveReadyMessage);
  }

  function run(msg: SolveRunMessage): void {
    if (!wasm) {
      self.postMessage({ type: 'error', message: 'WASM not initialized' } as SolveErrorMessage);
      return;
    }
    const start = performance.now();
    try {
      const driver = new wasm.WasmBulkDriver(msg.configToml);
      self.postMessage({
        type: 'meta',
        totalJobs: driver.jobCount,
        solverType: driver.solverType,
        precision: driver.precision,
      } as SolveMetaMessage);

      const stats = driver.runWithKPointStreaming((k: unknown) => {
        self.postMessage({
          type: 'kpoint',
          data: toPlain(k) as Record<string, unknown>,
        } as SolveKPointMessage);
      });

      self.postMessage({
        type: 'done',
        stats: toPlain(stats) as Record<string, unknown>,
        totalTime: performance.now() - start,
      } as SolveDoneMessage);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({ type: 'error', message } as SolveErrorMessage);
    }
  }

  self.onmessage = async (event: MessageEvent<SolveWorkerInMessage>) => {
    const msg = event.data;
    switch (msg.type) {
      case 'init':
        basePath = msg.basePath || '';
        await ensureWasm();
        break;
      case 'run':
        if (!wasm) await ensureWasm();
        run(msg);
        break;
    }
  };
}
