/* tslint:disable */
/* eslint-disable */
/**
 * Check if streaming mode is supported.
 */
export function isStreamingSupported(): boolean;
/**
 * Get the library version.
 */
export function getVersion(): string;
/**
 * Get supported solver types.
 */
export function getSupportedSolvers(): Array<any>;
/**
 * Initialize the WASM module with proper panic handling.
 * Call this once at the start of your application.
 * 
 * This sets up the panic hook so that Rust panics are printed
 * to the browser console with full stack traces instead of
 * just showing "RuntimeError: unreachable".
 */
export function initPanicHook(): void;
/**
 * Check if selective filtering is supported.
 */
export function isSelectiveSupported(): boolean;
export class WasmBackend {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
/**
 * WebAssembly wrapper for the bulk driver with streaming support.
 *
 * This provides a unified interface for running band structure calculations
 * in the browser, with support for both Maxwell and EA solvers.
 */
export class WasmBulkDriver {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Run and return all results as an array (COLLECT mode).
   *
   * @returns Object with `results` (array) and `stats`
   */
  runCollect(): any;
  /**
   * Get the first N expanded job configurations.
   */
  getJobConfigs(n: number): Array<any>;
  /**
   * Run the computation with a callback for each result (STREAM mode).
   *
   * @param callback - Function(result) called for each completed job
   * @returns Statistics object
   */
  runWithCallback(callback: Function): any;
  /**
   * Run and collect with optional filtering.
   */
  runCollectFiltered(k_indices?: Uint32Array | null, band_indices?: Uint32Array | null): any;
  /**
   * Run with streaming output and server-side filtering (SELECTIVE mode).
   *
   * @param k_indices - Array of k-point indices to include (0-based), or null for all
   * @param band_indices - Array of band indices to include (0-based), or null for all
   * @param callback - Function(result) called for each filtered result
   * @returns Statistics object
   */
  runStreamingFiltered(k_indices: Uint32Array | null | undefined, band_indices: Uint32Array | null | undefined, callback: Function): any;
  /**
   * Run with K-POINT STREAMING: callback is called after EACH k-point is solved.
   *
   * This is the preferred mode for real-time visualization of band structure
   * computation. The callback receives incremental results as each k-point
   * completes, enabling smooth progressive rendering of the band diagram.
   *
   * @param callback - Function(kPointResult) called after each k-point solve
   * @returns Statistics object with completion info
   *
   * The callback receives an object with:
   * - `stream_type`: "k_point" (to distinguish from job-level streaming)
   * - `k_index`: Index of this k-point (0-based)
   * - `total_k_points`: Total number of k-points
   * - `k_point`: [kx, ky] in fractional coordinates
   * - `distance`: Cumulative path distance to this k-point
   * - `omegas`: Array of frequencies for all bands at this k-point
   * - `bands`: Same as omegas (alias for convenience)
   * - `iterations`: Number of LOBPCG iterations for this k-point
   * - `is_gamma`: Whether this is a Γ-point
   * - `progress`: Completion fraction (0.0 to 1.0)
   * - `params`: Job parameters (eps_bg, resolution, polarization, etc.)
   */
  runWithKPointStreaming(callback: Function): any;
  /**
   * Create a new bulk driver from TOML configuration.
   *
   * @param config_str - Configuration as TOML string
   * @throws Error if configuration is invalid
   */
  constructor(config_str: string);
  /**
   * Dry run: get job count and parameter info without executing.
   */
  dryRun(): any;
  /**
   * Check if this is a Maxwell solver.
   */
  readonly isMaxwell: boolean;
  /**
   * Get the solver type ("maxwell" or "ea").
   */
  readonly solverType: string;
  /**
   * Check if this is an EA solver.
   */
  readonly isEA: boolean;
  /**
   * Get the grid dimensions as [nx, ny].
   */
  readonly gridSize: Array<any>;
  /**
   * Get the number of jobs that will be executed.
   */
  readonly jobCount: number;
  /**
   * Get the number of bands being computed.
   */
  readonly numBands: number;
}
/**
 * WASM wrapper for selective filtering configuration.
 */
export class WasmSelectiveFilter {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Set k-point indices to include.
   */
  setKIndices(indices: Uint32Array): void;
  /**
   * Set band indices to include.
   */
  setBandIndices(indices: Uint32Array): void;
  /**
   * Create a new selective filter (pass-through by default).
   */
  constructor();
  /**
   * Clear all filter settings.
   */
  clear(): void;
  /**
   * Get the number of bands in the filter.
   */
  readonly bandCount: number | undefined;
  /**
   * Get band indices as array.
   */
  readonly bandIndices: Uint32Array;
  /**
   * Get the number of k-points in the filter.
   */
  readonly kCount: number | undefined;
  /**
   * Check if the filter is active.
   */
  readonly isActive: boolean;
  /**
   * Get k-indices as array.
   */
  readonly kIndices: Uint32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmbulkdriver_free: (a: number, b: number) => void;
  readonly __wbg_wasmselectivefilter_free: (a: number, b: number) => void;
  readonly getSupportedSolvers: () => number;
  readonly getVersion: (a: number) => void;
  readonly isSelectiveSupported: () => number;
  readonly wasmbulkdriver_dryRun: (a: number, b: number) => void;
  readonly wasmbulkdriver_getJobConfigs: (a: number, b: number, c: number) => void;
  readonly wasmbulkdriver_gridSize: (a: number) => number;
  readonly wasmbulkdriver_isEA: (a: number) => number;
  readonly wasmbulkdriver_isMaxwell: (a: number) => number;
  readonly wasmbulkdriver_jobCount: (a: number) => number;
  readonly wasmbulkdriver_new: (a: number, b: number, c: number) => void;
  readonly wasmbulkdriver_numBands: (a: number) => number;
  readonly wasmbulkdriver_runCollect: (a: number, b: number) => void;
  readonly wasmbulkdriver_runCollectFiltered: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly wasmbulkdriver_runStreamingFiltered: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly wasmbulkdriver_runWithCallback: (a: number, b: number, c: number) => void;
  readonly wasmbulkdriver_runWithKPointStreaming: (a: number, b: number, c: number) => void;
  readonly wasmbulkdriver_solverType: (a: number, b: number) => void;
  readonly wasmselectivefilter_bandCount: (a: number) => number;
  readonly wasmselectivefilter_bandIndices: (a: number, b: number) => void;
  readonly wasmselectivefilter_clear: (a: number) => void;
  readonly wasmselectivefilter_isActive: (a: number) => number;
  readonly wasmselectivefilter_kCount: (a: number) => number;
  readonly wasmselectivefilter_kIndices: (a: number, b: number) => void;
  readonly wasmselectivefilter_new: () => number;
  readonly wasmselectivefilter_setBandIndices: (a: number, b: number, c: number) => void;
  readonly wasmselectivefilter_setKIndices: (a: number, b: number, c: number) => void;
  readonly initPanicHook: () => void;
  readonly isStreamingSupported: () => number;
  readonly __wbg_wasmbackend_free: (a: number, b: number) => void;
  readonly __wbindgen_export: (a: number) => void;
  readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export3: (a: number, b: number) => number;
  readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
