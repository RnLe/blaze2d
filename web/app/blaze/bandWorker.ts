/**
 * Web Worker for Band Structure Computation
 * 
 * This worker runs WASM band structure calculations off the main thread,
 * enabling true live streaming of k-point results back to the UI.
 */

// Message types from main thread to worker
export interface WorkerComputeMessage {
  type: 'compute';
  configToml: string;
  polarization: 'TM' | 'TE';
}

export interface WorkerInitMessage {
  type: 'init';
  basePath: string;
}

export type WorkerInMessage = WorkerComputeMessage | WorkerInitMessage;

// Message types from worker to main thread
export interface WorkerKPointMessage {
  type: 'kpoint';
  polarization: 'TM' | 'TE';
  kIndex: number;
  totalKPoints: number;
  distance: number;
  omegas: number[];
  numBands: number;
  progress: number;
}

export interface WorkerDoneMessage {
  type: 'done';
  polarization: 'TM' | 'TE';
  totalTime: number;
}

export interface WorkerErrorMessage {
  type: 'error';
  message: string;
}

export interface WorkerReadyMessage {
  type: 'ready';
}

export type WorkerOutMessage = WorkerKPointMessage | WorkerDoneMessage | WorkerErrorMessage | WorkerReadyMessage;

// ============================================================================
// Worker Implementation
// ============================================================================

// Only run worker code if we're in a worker context
const isWorker = typeof self !== 'undefined' && typeof Window === 'undefined';

if (isWorker) {
  let wasmModule: any = null;
  let isInitialized = false;
  let basePath = ''; // Will be set from main thread

  // Initialize WASM module
  async function initWasm() {
    if (isInitialized) return;
    
    try {
      // In a worker, we need to use importScripts or dynamic import
      // The WASM files are served from /wasm-blaze/ (with basePath prefix for GitHub Pages)
      const wasmJsUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm.js`;
      const wasmBinaryUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm_bg.wasm`;
      
      // Dynamic import works in modern workers
      const mod = await import(/* webpackIgnore: true */ wasmJsUrl);
      await mod.default(wasmBinaryUrl);
      
      // Initialize panic hook for better error messages
      if (mod.initPanicHook) {
        mod.initPanicHook();
      }
      
      wasmModule = mod;
      isInitialized = true;
      
      self.postMessage({ type: 'ready' } as WorkerReadyMessage);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({ type: 'error', message: `Failed to init WASM: ${message}` } as WorkerErrorMessage);
    }
  }

  // Run band structure computation
  function compute(configToml: string, polarization: 'TM' | 'TE') {
    if (!wasmModule) {
      self.postMessage({ type: 'error', message: 'WASM not initialized' } as WorkerErrorMessage);
      return;
    }
    
    const startTime = performance.now();
    
    try {
      const driver = new wasmModule.WasmBulkDriver(configToml);
      
      // Run with k-point streaming
      driver.runWithKPointStreaming((kResult: any) => {
        // Post each k-point result back to main thread
        self.postMessage({
          type: 'kpoint',
          polarization,
          kIndex: kResult.k_index,
          totalKPoints: kResult.total_k_points,
          distance: kResult.distance,
          omegas: [...kResult.omegas],
          numBands: kResult.num_bands,
          progress: kResult.progress,
        } as WorkerKPointMessage);
      });
      
      const endTime = performance.now();
      self.postMessage({
        type: 'done',
        polarization,
        totalTime: endTime - startTime,
      } as WorkerDoneMessage);
      
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      self.postMessage({ type: 'error', message } as WorkerErrorMessage);
    }
  }

  // Handle messages from main thread
  self.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
    const msg = event.data;
    
    switch (msg.type) {
      case 'init':
        basePath = msg.basePath || '';
        await initWasm();
        break;
        
      case 'compute':
        if (!isInitialized) {
          await initWasm();
        }
        compute(msg.configToml, msg.polarization);
        break;
    }
  };

  // Wait for init message from main thread (which provides basePath)
  // Don't auto-initialize - we need the basePath for GitHub Pages
}
