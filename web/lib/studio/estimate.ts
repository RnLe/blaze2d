/**
 * Rough resource estimates for a configuration, for the studio footer and the
 * pre-run gate. Time is MODELED (labeled as such in the UI), not measured.
 */

import type { ValidationSummary } from './utilWorker';

export interface RunEstimate {
  jobs: number;
  kPointsTotal: number;
  /** Peak working-set bytes for one solve (very rough). */
  bytesPerSolve: number;
  /** Modeled wall-clock seconds for the whole sweep, single browser core. */
  modeledSeconds: number;
}

/**
 * Peak memory of one solve: the LOBPCG block state plus operator scratch.
 * Block state is roughly nx*ny complex elements per vector, across
 * ~3*(n_bands+slack) vectors (X, P, W), in the storage precision, plus a
 * handful of scratch grids.
 */
export function estimateBytes(nx: number, ny: number, nBands: number, precision: string): number {
  const elems = nx * ny;
  const complexBytes = precision === 'f32' ? 8 : 16; // Complex<f32|f64>
  const blockVecs = 3 * (nBands + 2);
  const scratchGrids = 8; // FFT scratch, grad_x/grad_y, epsilon, etc.
  return elems * complexBytes * (blockVecs + scratchGrids);
}

/**
 * Modeled per-k-point solve time on a single browser core. Anchored loosely to
 * the site's documented native TE numbers (~0.3 s at 32x32) scaled by grid
 * area and band count, with a browser-slowdown factor. Deliberately coarse.
 */
function modeledSecondsPerKPoint(nx: number, ny: number, nBands: number, precision: string): number {
  const area = nx * ny;
  const base = 3.0e-6; // s per (grid element * band), tuned to be order-right
  const precFactor = precision === 'f32' ? 0.7 : 1.0;
  return base * area * nBands * precFactor;
}

export function estimateRun(
  summary: ValidationSummary | null,
): RunEstimate | null {
  if (!summary) return null;
  const { jobs, nx, ny, n_bands, k_points, precision } = summary;
  const perK = modeledSecondsPerKPoint(nx, ny, n_bands, precision);
  return {
    jobs,
    kPointsTotal: jobs * k_points,
    bytesPerSolve: estimateBytes(nx, ny, n_bands, precision),
    modeledSeconds: jobs * k_points * perK,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) return '< 1 s';
  if (seconds < 90) return `~${Math.round(seconds)} s`;
  if (seconds < 3600) return `~${Math.round(seconds / 60)} min`;
  return `~${(seconds / 3600).toFixed(1)} h`;
}
