/**
 * Blaze2D WASM TypeScript Definitions
 * 
 * This file provides TypeScript interfaces for working with the Blaze2D WASM module.
 * Copy this to your project's types/ directory.
 */

// =============================================================================
// Result Types
// =============================================================================

/** Atom parameters in a unit cell */
export interface AtomParams {
    index: number;
    pos: [number, number];
    radius: number;
    eps_inside: number;
}

/** Job parameters used for calculation */
export interface JobParams {
    eps_bg: number;
    resolution: number;
    polarization: 'TE' | 'TM';
    lattice_type?: string;
    atoms: AtomParams[];
}

/** Maxwell solver result (photonic crystal band structure) */
export interface MaxwellResult {
    result_type: 'maxwell';
    job_index: number;
    params: JobParams;
    
    /** K-points as [kx, ky] in reciprocal lattice units */
    k_path: [number, number][];
    
    /** Cumulative distance along k-path */
    distances: number[];
    
    /** Band frequencies: bands[k_index][band_index] = ωa/2πc */
    bands: number[][];
    
    /** Number of k-points in path */
    num_k_points: number;
    
    /** Number of computed bands */
    num_bands: number;
}

/** EA (Envelope Approximation) solver result */
export interface EAResult {
    result_type: 'ea';
    job_index: number;
    params: JobParams;
    
    /** Eigenvalues */
    eigenvalues: number[];
    
    /** Eigenvectors as [band][grid_index][re, im] */
    eigenvectors: [number, number][][];
    
    /** Grid dimensions [nx, ny] */
    grid_dims: [number, number];
    
    /** Number of LOBPCG iterations */
    n_iterations: number;
    
    /** Whether solver converged */
    converged: boolean;
    
    num_eigenvalues: number;
    num_bands: number;
}

/** Union type for all result types */
export type BandResult = MaxwellResult | EAResult;

// =============================================================================
// K-Point Streaming Types
// =============================================================================

/** K-point streaming result (emitted after each k-point solve) */
export interface KPointResult {
    /** Identifies this as k-point streaming data */
    stream_type: 'k_point';
    
    /** Which job this belongs to */
    job_index: number;
    
    /** Index of this k-point (0-based) */
    k_index: number;
    
    /** Total number of k-points in the path */
    total_k_points: number;
    
    /** K-point coordinates [kx, ky] in fractional units */
    k_point: [number, number];
    
    /** Cumulative path distance to this k-point */
    distance: number;
    
    /** Frequencies for all bands at this k-point (normalized: ωa/2πc) */
    omegas: number[];
    
    /** Same as omegas (alias for convenience) */
    bands: number[];
    
    /** Number of LOBPCG iterations for this k-point */
    iterations: number;
    
    /** Whether this is a Γ-point (k ≈ 0) */
    is_gamma: boolean;
    
    /** Number of bands */
    num_bands: number;
    
    /** Completion fraction (0.0 to 1.0) */
    progress: number;
    
    /** Job parameters */
    params: JobParams;
}

/** Statistics from a driver run */
export interface DriverStats {
    total_jobs: number;
    completed: number;
    failed: number;
    total_time_ms: number;
    jobs_per_second?: number;
}

/** Dry run result */
export interface DryRunResult {
    total_jobs: number;
    thread_mode: string;
    solver_type: 'maxwell' | 'ea';
}

/** Collected results with statistics */
export interface CollectResult {
    results: BandResult[];
    stats: DriverStats;
}

// =============================================================================
// WASM Module Interface
// =============================================================================

/** WASM Bulk Driver class */
export declare class WasmBulkDriver {
    constructor(configToml: string);
    
    /** Number of jobs to execute */
    readonly jobCount: number;
    
    /** Solver type: "maxwell" or "ea" */
    readonly solverType: 'maxwell' | 'ea';
    
    /** True if using EA solver */
    readonly isEA: boolean;
    
    /** True if using Maxwell solver */
    readonly isMaxwell: boolean;
    
    /** Grid dimensions [nx, ny] */
    readonly gridSize: [number, number];
    
    /** Number of bands being computed */
    readonly numBands: number;
    
    /** Run with streaming callback (STREAM mode) - callback per job */
    runWithCallback(callback: (result: BandResult) => void): DriverStats;
    
    /** 
     * Run with K-POINT STREAMING - callback after EACH k-point solve.
     * This is the preferred mode for real-time visualization.
     * The callback receives incremental results as each k-point completes.
     */
    runWithKPointStreaming(callback: (result: KPointResult) => void): DriverStats;
    
    /** Run with selective filtering (SELECTIVE mode) */
    runStreamingFiltered(
        kIndices: number[] | null,
        bandIndices: number[] | null,
        callback: (result: BandResult) => void
    ): DriverStats;
    
    /** Run and collect all results (COLLECT mode) */
    runCollect(): CollectResult;
    
    /** Run and collect with filtering */
    runCollectFiltered(
        kIndices: number[] | null,
        bandIndices: number[] | null
    ): CollectResult;
    
    /** Get job info without executing */
    dryRun(): DryRunResult;
    
    /** Get first N job configurations */
    getJobConfigs(n: number): JobParams[];
}

/** Selective filter configuration */
export declare class WasmSelectiveFilter {
    constructor();
    
    setKIndices(indices: number[]): void;
    setBandIndices(indices: number[]): void;
    clear(): void;
    
    readonly isActive: boolean;
    readonly kCount: number | null;
    readonly bandCount: number | null;
    readonly kIndices: number[];
    readonly bandIndices: number[];
}

// =============================================================================
// Module Functions
// =============================================================================

/** Initialize the WASM module (call once before using) */
export declare function init(): Promise<void>;

/** Get library version */
export declare function getVersion(): string;

/** Check if streaming mode is supported */
export declare function isStreamingSupported(): boolean;

/** Check if selective filtering is supported */
export declare function isSelectiveSupported(): boolean;

/** Get list of supported solver types */
export declare function getSupportedSolvers(): string[];
