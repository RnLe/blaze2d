/**
 * MPB2D Web Worker for Background Band Structure Computation
 * 
 * This worker runs the WASM computation in a background thread, allowing
 * the main thread to remain responsive and update the UI in real-time
 * as k-points are computed.
 * 
 * SETUP
 * -----
 * 1. Copy this file to your project (e.g., public/workers/bandWorker.ts)
 * 2. For Next.js, you may need to configure webpack for workers
 * 3. Use with useBandStructureWorker hook from useBandStructure.tsx
 * 
 * HOW IT WORKS
 * ------------
 * Main Thread                    Worker Thread
 *     |                               |
 *     |-- postMessage(config) ------->|
 *     |                               | await init()
 *     |                               | driver = new WasmBulkDriver(config)
 *     |                               | driver.runWithKPointStreaming(...)
 *     |<-- postMessage(kResult) ------|  (for each k-point)
 *     |<-- postMessage(kResult) ------|
 *     |<-- postMessage(kResult) ------|
 *     |<-- postMessage(complete) -----|
 *     |                               |
 */

// Worker message types
export interface WorkerMessage {
    type: 'start' | 'abort';
    config?: string;
}

export interface WorkerResponse {
    type: 'init' | 'progress' | 'complete' | 'error';
    data?: KPointResultMessage;
    stats?: StatsMessage;
    error?: string;
}

export interface KPointResultMessage {
    stream_type: 'k_point';
    job_index: number;
    k_index: number;
    total_k_points: number;
    k_point: [number, number];
    distance: number;
    omegas: number[];
    bands: number[];
    iterations: number;
    is_gamma: boolean;
    num_bands: number;
    progress: number;
}

export interface StatsMessage {
    total_jobs: number;
    completed: number;
    failed: number;
    total_time_ms: number;
}

// =============================================================================
// Worker Implementation
// =============================================================================

// This runs in the worker context
const workerCode = `
let wasmModule = null;

async function initWasm() {
    if (wasmModule) return wasmModule;
    
    // Import the WASM module
    // Adjust path based on your deployment
    const mod = await import('/wasm/mpb2d_backend_wasm.js');
    await mod.default();
    wasmModule = mod;
    return mod;
}

self.onmessage = async (e) => {
    const msg = e.data;
    
    if (msg.type === 'start') {
        try {
            // Initialize WASM if needed
            const wasm = await initWasm();
            self.postMessage({ type: 'init' });
            
            // Create driver
            const driver = new wasm.WasmBulkDriver(msg.config);
            
            // Run with k-point streaming
            const stats = driver.runWithKPointStreaming((kResult) => {
                // Send each k-point result back to main thread
                self.postMessage({
                    type: 'progress',
                    data: {
                        stream_type: kResult.stream_type,
                        job_index: kResult.job_index,
                        k_index: kResult.k_index,
                        total_k_points: kResult.total_k_points,
                        k_point: [...kResult.k_point],
                        distance: kResult.distance,
                        omegas: [...kResult.omegas],
                        bands: [...kResult.bands],
                        iterations: kResult.iterations,
                        is_gamma: kResult.is_gamma,
                        num_bands: kResult.num_bands,
                        progress: kResult.progress,
                    }
                });
            });
            
            // Send completion message
            self.postMessage({
                type: 'complete',
                stats: {
                    total_jobs: stats.total_jobs,
                    completed: stats.completed,
                    failed: stats.failed,
                    total_time_ms: stats.total_time_ms,
                }
            });
            
        } catch (err) {
            self.postMessage({
                type: 'error',
                error: err.message || String(err)
            });
        }
    }
};
`;

// =============================================================================
// Worker Factory
// =============================================================================

/**
 * Create a Web Worker for band structure computation.
 * 
 * This uses a Blob URL to avoid needing a separate worker file,
 * making it easier to integrate with bundlers like webpack/vite.
 * 
 * @returns Worker instance
 */
export function createBandWorker(): Worker {
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url, { type: 'module' });
    
    // Clean up blob URL when worker terminates
    worker.addEventListener('error', () => URL.revokeObjectURL(url));
    
    return worker;
}

// =============================================================================
// Alternative: Standalone Worker File
// =============================================================================

/**
 * If the inline Blob approach doesn't work with your bundler,
 * save the workerCode content to a separate file (e.g., public/workers/band.worker.js)
 * and use:
 * 
 * const worker = new Worker('/workers/band.worker.js', { type: 'module' });
 */
