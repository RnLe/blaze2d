/**
 * MPB2D React Hooks for Band Structure Calculations
 * 
 * This module provides React hooks for computing photonic crystal band
 * structures using the MPB2D WASM module in React/Next.js applications.
 * 
 * AVAILABLE HOOKS (in order of recommendation):
 * 
 * 1. useBandStructureWorker() - BEST FOR UI
 *    Uses a Web Worker for background computation. The UI stays responsive
 *    and React can re-render the band diagram progressively as k-points
 *    are computed.
 * 
 * 2. useBandStructureStreaming() - Synchronous k-point streaming
 *    Simpler but blocks the main thread. UI won't update until complete.
 *    Good for testing or when workers aren't available.
 * 
 * 3. useBandStructure() - Job-level streaming
 *    For parameter sweeps. Gets complete band structures for each job.
 * 
 * INSTALLATION
 * ------------
 * 1. Build WASM module: `make wasm` in the blaze2d repo
 * 2. Copy wasm-dist/* to your Next.js public/wasm/ folder
 * 3. Copy this file to your hooks/ directory
 * 4. Copy blaze2d-wasm.d.ts to your types/ directory
 * 
 * QUICK START
 * -----------
 * IMPORTANT: All configurations MUST include a [bulk] section!
 * 
 * ```tsx
 * 'use client';
 * 
 * import { useBandStructureWorker } from '@/hooks/useBandStructure';
 * 
 * const CONFIG = `
 * [bulk]
 * 
 * [solver]
 * type = "maxwell"
 * polarization = "TM"
 * 
 * [geometry]
 * eps_bg = 1.0
 * 
 * [geometry.lattice]
 * type = "square"
 * a = 1.0
 * 
 * [[geometry.atoms]]
 * pos = [0.5, 0.5]
 * radius = 0.2
 * eps_inside = 12.0
 * 
 * [grid]
 * nx = 32
 * ny = 32
 * lx = 1.0
 * ly = 1.0
 * 
 * [path]
 * preset = "square"
 * segments_per_leg = 12
 * 
 * [eigensolver]
 * n_bands = 8
 * max_iter = 200
 * tol = 1e-8
 * `;
 * 
 * export function MyComponent() {
 *     // Use Web Worker for responsive UI
 *     const { data, progress, isLoading, compute, abort } = useBandStructureWorker();
 *     
 *     const handleCompute = () => compute(CONFIG);
 *     
 *     if (isLoading) {
 *         return (
 *             <div>
 *                 Computing... {(progress * 100).toFixed(0)}%
 *                 <button onClick={abort}>Cancel</button>
 *             </div>
 *         );
 *     }
 *     
 *     if (data.distances.length === 0) {
 *         return <button onClick={handleCompute}>Compute</button>;
 *     }
 *     
 *     return <BandPlot distances={data.distances} bands={data.bands} />;
 * }
 * ```
 */

'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import type { MaxwellResult, DriverStats, BandResult, KPointResult } from './blaze2d-wasm';

// =============================================================================
// Types
// =============================================================================

/** Progressive band data structure (for k-point streaming) */
export interface BandData {
    /** Cumulative distances along k-path */
    distances: number[];
    /** Band frequencies: bands[k_index][band_index] */
    bands: number[][];
}

export interface UseBandStructureStreamingResult {
    /** Progressive band data (grows as k-points complete) */
    data: BandData;
    
    /** Progress fraction (0.0 to 1.0) */
    progress: number;
    
    /** Whether computation is in progress */
    isLoading: boolean;
    
    /** Error message if computation failed */
    error: string | null;
    
    /** Statistics from the last run */
    stats: DriverStats | null;
    
    /** Current k-point index being computed */
    currentKIndex: number;
    
    /** Total number of k-points */
    totalKPoints: number;
    
    /** Start a new computation with the given TOML config */
    compute: (configToml: string) => Promise<void>;
    
    /** Reset state and clear results */
    reset: () => void;
    
    /** Whether WASM module is initialized */
    isInitialized: boolean;
}

export interface UseBandStructureResult {
    /** The computed band structure result (null if not computed yet) */
    result: MaxwellResult | null;
    
    /** Whether computation is in progress */
    isLoading: boolean;
    
    /** Progress percentage (0-100) for multi-job configs */
    progress: number;
    
    /** Error message if computation failed */
    error: string | null;
    
    /** Statistics from the last run */
    stats: DriverStats | null;
    
    /** Start a new computation with the given TOML config */
    compute: (configToml: string) => Promise<void>;
    
    /** Reset state and clear results */
    reset: () => void;
    
    /** Whether WASM module is initialized */
    isInitialized: boolean;
}

export interface UseBandStructureOptions {
    /** Auto-initialize WASM on mount (default: true) */
    autoInit?: boolean;
    
    /** Callback for each streaming result */
    onResult?: (result: BandResult) => void;
    
    /** Callback when computation completes */
    onComplete?: (stats: DriverStats) => void;
    
    /** Callback on error */
    onError?: (error: Error) => void;
}

// =============================================================================
// WASM Module Loading
// =============================================================================

let wasmModule: typeof import('./blaze2d-wasm') | null = null;
let wasmInitPromise: Promise<void> | null = null;

async function loadWasm(): Promise<typeof import('./blaze2d-wasm')> {
    if (wasmModule) return wasmModule;
    
    if (!wasmInitPromise) {
        wasmInitPromise = (async () => {
            // Dynamic import for Next.js compatibility
            // Adjust the path based on your public folder structure
            const mod = await import('/wasm/blaze2d_backend_wasm.js' as any);
            await mod.default();  // Initialize WASM
            wasmModule = mod;
        })();
    }
    
    await wasmInitPromise;
    return wasmModule!;
}

// =============================================================================
// Hook Implementation: K-Point Streaming (RECOMMENDED)
// =============================================================================

/**
 * React hook for band structure computation with K-POINT STREAMING.
 * 
 * This is the RECOMMENDED hook for real-time visualization of band structures.
 * The data is updated progressively as each k-point is computed, enabling
 * smooth rendering of the band diagram.
 * 
 * @example
 * ```tsx
 * const { data, progress, isLoading, compute } = useBandStructureStreaming();
 * 
 * // Trigger computation
 * compute(configToml);
 * 
 * // data.distances and data.bands grow as k-points complete
 * // progress goes from 0.0 to 1.0
 * ```
 */
export function useBandStructureStreaming(
    options: UseBandStructureOptions = {}
): UseBandStructureStreamingResult {
    const { autoInit = true, onError } = options;
    
    const [data, setData] = useState<BandData>({ distances: [], bands: [] });
    const [progress, setProgress] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [stats, setStats] = useState<DriverStats | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);
    const [currentKIndex, setCurrentKIndex] = useState(0);
    const [totalKPoints, setTotalKPoints] = useState(0);
    
    const abortRef = useRef(false);
    
    // Auto-initialize WASM on mount
    useEffect(() => {
        if (autoInit) {
            loadWasm()
                .then(() => setIsInitialized(true))
                .catch((e) => {
                    console.error('Failed to load WASM:', e);
                    setError('Failed to load WASM module');
                });
        }
    }, [autoInit]);
    
    const compute = useCallback(async (configToml: string) => {
        try {
            abortRef.current = false;
            setIsLoading(true);
            setError(null);
            setProgress(0);
            setData({ distances: [], bands: [] });
            setCurrentKIndex(0);
            setTotalKPoints(0);
            
            const wasm = await loadWasm();
            setIsInitialized(true);
            
            const driver = new wasm.WasmBulkDriver(configToml);
            
            // Use K-POINT STREAMING for real-time updates
            const runStats = driver.runWithKPointStreaming((kResult: KPointResult) => {
                if (abortRef.current) return;
                
                // Update progress
                setProgress(kResult.progress);
                setCurrentKIndex(kResult.k_index);
                setTotalKPoints(kResult.total_k_points);
                
                // Append new k-point data (immutable update for React)
                setData(prev => ({
                    distances: [...prev.distances, kResult.distance],
                    bands: [...prev.bands, [...kResult.omegas]]
                }));
            });
            
            setStats(runStats);
            
        } catch (e) {
            const err = e instanceof Error ? e : new Error(String(e));
            setError(err.message);
            onError?.(err);
        } finally {
            setIsLoading(false);
        }
    }, [onError]);
    
    const reset = useCallback(() => {
        abortRef.current = true;
        setData({ distances: [], bands: [] });
        setIsLoading(false);
        setProgress(0);
        setError(null);
        setStats(null);
        setCurrentKIndex(0);
        setTotalKPoints(0);
    }, []);
    
    return {
        data,
        progress,
        isLoading,
        error,
        stats,
        currentKIndex,
        totalKPoints,
        compute,
        reset,
        isInitialized,
    };
}

// =============================================================================
// Hook Implementation: Web Worker K-Point Streaming (RECOMMENDED FOR UI)
// =============================================================================

/**
 * React hook for band structure computation using a WEB WORKER.
 * 
 * This is the BEST hook for real-time visualization because it runs the
 * WASM computation in a background thread, allowing React to update the
 * DOM smoothly as each k-point completes.
 * 
 * The main thread remains responsive and can re-render the band diagram
 * progressively without freezing.
 * 
 * @example
 * ```tsx
 * const { data, progress, isLoading, compute, abort } = useBandStructureWorker();
 * 
 * // Trigger computation (runs in background thread!)
 * compute(configToml);
 * 
 * // data.distances and data.bands grow as k-points complete
 * // UI remains responsive throughout
 * ```
 */
export function useBandStructureWorker(): UseBandStructureStreamingResult & { abort: () => void } {
    const [data, setData] = useState<BandData>({ distances: [], bands: [] });
    const [progress, setProgress] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [stats, setStats] = useState<DriverStats | null>(null);
    const [currentKIndex, setCurrentKIndex] = useState(0);
    const [totalKPoints, setTotalKPoints] = useState(0);
    const [isInitialized, setIsInitialized] = useState(true); // Worker handles init
    
    const workerRef = useRef<Worker | null>(null);
    
    // Cleanup worker on unmount
    useEffect(() => {
        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
        };
    }, []);
    
    const compute = useCallback(async (configToml: string) => {
        // Terminate any existing worker
        if (workerRef.current) {
            workerRef.current.terminate();
        }
        
        setIsLoading(true);
        setError(null);
        setProgress(0);
        setData({ distances: [], bands: [] });
        setCurrentKIndex(0);
        setTotalKPoints(0);
        
        // Create worker with inline code (works with most bundlers)
        const workerCode = `
let wasmModule = null;

async function initWasm() {
    if (wasmModule) return wasmModule;
    const mod = await import('/wasm/blaze2d_backend_wasm.js');
    await mod.default();
    wasmModule = mod;
    return mod;
}

self.onmessage = async (e) => {
    const msg = e.data;
    
    if (msg.type === 'start') {
        try {
            const wasm = await initWasm();
            self.postMessage({ type: 'init' });
            
            const driver = new wasm.WasmBulkDriver(msg.config);
            
            const stats = driver.runWithKPointStreaming((kResult) => {
                self.postMessage({
                    type: 'progress',
                    data: {
                        k_index: kResult.k_index,
                        total_k_points: kResult.total_k_points,
                        distance: kResult.distance,
                        omegas: [...kResult.omegas],
                        progress: kResult.progress,
                    }
                });
            });
            
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
            self.postMessage({ type: 'error', error: err.message || String(err) });
        }
    }
};
`;
        
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url, { type: 'module' });
        workerRef.current = worker;
        
        worker.onmessage = (e) => {
            const msg = e.data;
            
            switch (msg.type) {
                case 'init':
                    // WASM initialized in worker
                    break;
                    
                case 'progress':
                    // K-point completed - update React state
                    setProgress(msg.data.progress);
                    setCurrentKIndex(msg.data.k_index);
                    setTotalKPoints(msg.data.total_k_points);
                    setData(prev => ({
                        distances: [...prev.distances, msg.data.distance],
                        bands: [...prev.bands, [...msg.data.omegas]]
                    }));
                    break;
                    
                case 'complete':
                    setStats(msg.stats);
                    setIsLoading(false);
                    URL.revokeObjectURL(url);
                    break;
                    
                case 'error':
                    setError(msg.error);
                    setIsLoading(false);
                    URL.revokeObjectURL(url);
                    break;
            }
        };
        
        worker.onerror = (e) => {
            setError(e.message || 'Worker error');
            setIsLoading(false);
            URL.revokeObjectURL(url);
        };
        
        // Start computation
        worker.postMessage({ type: 'start', config: configToml });
    }, []);
    
    const abort = useCallback(() => {
        if (workerRef.current) {
            workerRef.current.terminate();
            workerRef.current = null;
            setIsLoading(false);
        }
    }, []);
    
    const reset = useCallback(() => {
        abort();
        setData({ distances: [], bands: [] });
        setProgress(0);
        setError(null);
        setStats(null);
        setCurrentKIndex(0);
        setTotalKPoints(0);
    }, [abort]);
    
    return {
        data,
        progress,
        isLoading,
        error,
        stats,
        currentKIndex,
        totalKPoints,
        compute,
        reset,
        abort,
        isInitialized,
    };
}

// =============================================================================
// Hook Implementation: Job-Level Streaming (for parameter sweeps)
// =============================================================================

/**
 * React hook for band structure computation with JOB-LEVEL streaming.
 * 
 * Use this hook when running parameter sweeps where you want the complete
 * band structure for each job. For single calculations with real-time updates,
 * prefer `useBandStructureStreaming()` instead.
 * 
 * @example
 * ```tsx
 * const { result, isLoading, compute } = useBandStructure();
 * compute(configToml);
 * // result is set once the entire band structure is complete
 * ```
 */
export function useBandStructure(
    options: UseBandStructureOptions = {}
): UseBandStructureResult {
    const { autoInit = true, onResult, onComplete, onError } = options;
    
    const [result, setResult] = useState<MaxwellResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [stats, setStats] = useState<DriverStats | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);
    
    const abortRef = useRef(false);
    
    // Auto-initialize WASM on mount
    useEffect(() => {
        if (autoInit) {
            loadWasm()
                .then(() => setIsInitialized(true))
                .catch((e) => {
                    console.error('Failed to load WASM:', e);
                    setError('Failed to load WASM module');
                });
        }
    }, [autoInit]);
    
    const compute = useCallback(async (configToml: string) => {
        try {
            abortRef.current = false;
            setIsLoading(true);
            setError(null);
            setProgress(0);
            setResult(null);
            
            const wasm = await loadWasm();
            setIsInitialized(true);
            
            const driver = new wasm.WasmBulkDriver(configToml);
            const totalJobs = driver.jobCount;
            let completed = 0;
            
            // Run with streaming callback
            const runStats = driver.runWithCallback((res: BandResult) => {
                if (abortRef.current) return;
                
                completed++;
                setProgress((completed / totalJobs) * 100);
                
                // For Maxwell results, update the main result
                if (res.result_type === 'maxwell') {
                    setResult(res as MaxwellResult);
                }
                
                // Call user callback if provided
                onResult?.(res);
            });
            
            setStats(runStats);
            onComplete?.(runStats);
            
        } catch (e) {
            const err = e instanceof Error ? e : new Error(String(e));
            setError(err.message);
            onError?.(err);
        } finally {
            setIsLoading(false);
        }
    }, [onResult, onComplete, onError]);
    
    const reset = useCallback(() => {
        abortRef.current = true;
        setResult(null);
        setIsLoading(false);
        setProgress(0);
        setError(null);
        setStats(null);
    }, []);
    
    return {
        result,
        isLoading,
        progress,
        error,
        stats,
        compute,
        reset,
        isInitialized,
    };
}

// =============================================================================
// Utility Hook: Multiple Jobs with Progress
// =============================================================================

export interface UseMultiJobResult {
    /** All collected results */
    results: BandResult[];
    
    /** Whether computation is in progress */
    isLoading: boolean;
    
    /** Progress percentage (0-100) */
    progress: number;
    
    /** Error message if computation failed */
    error: string | null;
    
    /** Statistics from the last run */
    stats: DriverStats | null;
    
    /** Start computation */
    compute: (configToml: string) => Promise<void>;
    
    /** Reset state */
    reset: () => void;
}

/**
 * Hook for computing multiple jobs (parameter sweeps) with streaming updates.
 */
export function useMultiJobBandStructure(): UseMultiJobResult {
    const [results, setResults] = useState<BandResult[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [stats, setStats] = useState<DriverStats | null>(null);
    
    const compute = useCallback(async (configToml: string) => {
        try {
            setIsLoading(true);
            setError(null);
            setProgress(0);
            setResults([]);
            
            const wasm = await loadWasm();
            const driver = new wasm.WasmBulkDriver(configToml);
            const totalJobs = driver.jobCount;
            
            const runStats = driver.runWithCallback((res: BandResult) => {
                setResults(prev => [...prev, res]);
                setProgress((prev) => Math.min(100, prev + (100 / totalJobs)));
            });
            
            setStats(runStats);
            setProgress(100);
            
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setIsLoading(false);
        }
    }, []);
    
    const reset = useCallback(() => {
        setResults([]);
        setIsLoading(false);
        setProgress(0);
        setError(null);
        setStats(null);
    }, []);
    
    return { results, isLoading, progress, error, stats, compute, reset };
}

// =============================================================================
// Example Components
// =============================================================================

/**
 * Example: Band structure viewer with WEB WORKER K-POINT STREAMING
 * 
 * This component demonstrates real-time progressive rendering of a band
 * structure using a Web Worker. The UI remains responsive as each k-point
 * is computed in the background.
 * 
 * IMPORTANT: The [bulk] section is REQUIRED for all configurations.
 * Without it, the WasmBulkDriver will throw an error.
 * 
 * ```tsx
 * import { BandStructureViewer } from './useBandStructure';
 * 
 * const config = `
 *     # REQUIRED: [bulk] section marks this as a valid configuration
 *     [bulk]
 *     
 *     [solver]
 *     type = "maxwell"
 *     
 *     polarization = "TM"
 *     
 *     [geometry]
 *     eps_bg = 1.0
 *     
 *     [geometry.lattice]
 *     type = "square"
 *     a = 1.0
 *     
 *     [[geometry.atoms]]
 *     pos = [0.5, 0.5]
 *     radius = 0.2
 *     eps_inside = 12.0
 *     
 *     [grid]
 *     nx = 32
 *     ny = 32
 *     lx = 1.0
 *     ly = 1.0
 *     
 *     [path]
 *     preset = "square"
 *     segments_per_leg = 12
 *     
 *     [eigensolver]
 *     n_bands = 8
 *     max_iter = 200
 *     tol = 1e-8
 * `;
 * 
 * <BandStructureViewer config={config} />
 * ```
 */
export function BandStructureViewer({ config }: { config: string }) {
    // Use WEB WORKER for responsive UI during computation
    const { data, progress, isLoading, error, compute, abort, currentKIndex, totalKPoints } = 
        useBandStructureWorker();
    
    useEffect(() => {
        if (config) {
            compute(config);
        }
        // Cleanup: abort on unmount
        return () => abort();
    }, [config, compute, abort]);
    
    if (error) {
        return (
            <div style={{ padding: 20, color: 'red' }}>
                <strong>Error:</strong> {error}
            </div>
        );
    }
    
    // Show progress during computation with streaming data
    if (isLoading) {
        const numBands = data.bands[0]?.length || 0;
        return (
            <div style={{ padding: 20, fontFamily: 'monospace' }}>
                <h3>Computing Band Structure...</h3>
                <p>
                    Progress: {(progress * 100).toFixed(0)}% 
                    ({currentKIndex + 1}/{totalKPoints} k-points)
                </p>
                <button onClick={abort} style={{ marginBottom: 10 }}>
                    Cancel
                </button>
                {data.distances.length > 0 && (
                    <>
                        <h4>Band Frequencies (so far):</h4>
                        <p>{data.distances.length} k-points computed, {numBands} bands</p>
                        <ul>
                            {data.bands[data.bands.length - 1]?.slice(0, 4).map((freq, i) => (
                                <li key={i}>Band {i + 1}: ω = {freq.toFixed(4)}</li>
                            ))}
                            {numBands > 4 && <li>... and {numBands - 4} more bands</li>}
                        </ul>
                    </>
                )}
            </div>
        );
    }
    
    if (data.distances.length === 0) {
        return <div style={{ padding: 20 }}>No results yet</div>;
    }
    
    // Display completed results
    const numBands = data.bands[0]?.length || 0;
    return (
        <div style={{ padding: 20, fontFamily: 'monospace' }}>
            <h3>Band Structure Results</h3>
            <p>K-points: {data.distances.length}</p>
            <p>Bands: {numBands}</p>
            
            <h4>Band Frequencies at Γ (k=0):</h4>
            <ul>
                {data.bands[0]?.map((freq, i) => (
                    <li key={i}>Band {i + 1}: ω = {freq.toFixed(4)}</li>
                ))}
            </ul>
            
            <h4>Band Frequencies at last k-point:</h4>
            <ul>
                {data.bands[data.bands.length - 1]?.map((freq, i) => (
                    <li key={i}>Band {i + 1}: ω = {freq.toFixed(4)}</li>
                ))}
            </ul>
        </div>
    );
}

/**
 * Example: Plotly-based band structure plot with streaming updates
 * 
 * This component requires react-plotly.js to be installed:
 * npm install react-plotly.js plotly.js
 * 
 * ```tsx
 * import { StreamingBandPlot } from './useBandStructure';
 * 
 * <StreamingBandPlot distances={data.distances} bands={data.bands} />
 * ```
 */
export function StreamingBandPlot({ 
    distances, 
    bands,
    title = 'Photonic Band Structure'
}: { 
    distances: number[], 
    bands: number[][],
    title?: string
}) {
    // This is a placeholder - replace with your plotting library
    // For a real implementation with Plotly, see the tutorial TOML file
    
    if (distances.length === 0) {
        return <div>No data to plot yet...</div>;
    }
    
    const numBands = bands[0]?.length || 0;
    const numKPoints = distances.length;
    
    return (
        <div style={{ 
            padding: 20, 
            border: '1px solid #ccc', 
            borderRadius: 8,
            backgroundColor: '#f9f9f9'
        }}>
            <h4>{title}</h4>
            <p style={{ fontFamily: 'monospace', fontSize: 12 }}>
                {numKPoints} k-points × {numBands} bands
            </p>
            <p style={{ color: '#666', fontSize: 12 }}>
                Replace this placeholder with react-plotly.js or your preferred plotting library.
                See square_lattice_tutorial.toml for Plotly integration examples.
            </p>
        </div>
    );
}
