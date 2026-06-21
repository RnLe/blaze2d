/**
 * BandResult mapping and band sorting.
 *
 * Mirrors the `BandResult` container and `_sort_bands` logic from the Python
 * high-level API (`crates/python/blaze/solve.py`) so that in-browser results
 * expose the exact same attribute shape that Python users see.
 */

import { kPathDistances } from './configGen';

/** Strict parity with Python's `BandResult` public attributes. */
export interface BandResult {
  /** Frequencies, shape (n_kpoints, n_bands), units c/a. */
  freqs: number[][];
  /** Cumulative k-path distance, shape (n_kpoints,). */
  distances: number[];
  /** Fractional reciprocal coords, shape (n_kpoints, 2). */
  k_points: number[][];
  lattice_type: string;
  polarization: string;
  epsilon_background: number;
  epsilon_atoms: number;
  radius_atom: number;
  resolution: number;
  n_bands: number;
  n_kpoints: number;
  k_labels: string[];
  k_label_distances: number[];
}

/**
 * Re-index bands so each band varies smoothly along the k-path.
 *
 * Port of `_sort_bands`. The Python version uses the Hungarian algorithm
 * (`scipy.optimize.linear_sum_assignment`); here we use a greedy
 * nearest-neighbour assignment which is identical for the non-degenerate case
 * and a close approximation near crossings. The greedy match guarantees a
 * valid permutation (each new band used exactly once).
 */
export function sortBands(freqs: number[][]): number[][] {
  if (freqs.length === 0) return [];
  const out = freqs.map((row) => [...row]);
  const nB = out[0].length;

  for (let i = 1; i < out.length; i++) {
    const prev = out[i - 1];
    const cur = out[i];

    // Build candidate (prevBand, curBand, cost) and greedily assign smallest.
    const candidates: { p: number; c: number; cost: number }[] = [];
    for (let p = 0; p < nB; p++) {
      for (let c = 0; c < nB; c++) {
        const d = prev[p] - cur[c];
        candidates.push({ p, c, cost: d * d });
      }
    }
    candidates.sort((a, b) => a.cost - b.cost);

    const colForRow = new Array<number>(nB).fill(-1);
    const usedCol = new Array<boolean>(nB).fill(false);
    const usedRow = new Array<boolean>(nB).fill(false);
    let assigned = 0;
    for (const cand of candidates) {
      if (assigned === nB) break;
      if (usedRow[cand.p] || usedCol[cand.c]) continue;
      colForRow[cand.p] = cand.c;
      usedRow[cand.p] = true;
      usedCol[cand.c] = true;
      assigned++;
    }

    // colForRow[p] = which current column maps to previous band p.
    const reordered = new Array<number>(nB);
    for (let p = 0; p < nB; p++) {
      reordered[p] = cur[colForRow[p]];
    }
    out[i] = reordered;
  }
  return out;
}

/**
 * Accumulator for streamed k-point results that produces a `BandResult`.
 *
 * Feed each k-point result (from the WASM k-point streaming callback) via
 * `addKPoint`, then call `finish` to obtain the mapped, band-sorted result.
 */
export class BandResultBuilder {
  private kPoints: number[][] = [];
  private distances: number[] = [];
  private bands: number[][] = [];

  constructor(
    private readonly meta: {
      lattice_type: string;
      polarization: string;
      epsilon_background: number;
      epsilon_atoms: number;
      radius_atom: number;
      resolution: number;
      k_labels: string[];
      k_label_distances: number[];
    },
  ) {}

  addKPoint(k: { k_point: [number, number]; distance: number; bands: number[] }): void {
    this.kPoints.push([k.k_point[0], k.k_point[1]]);
    this.distances.push(k.distance);
    this.bands.push([...k.bands]);
  }

  get count(): number {
    return this.kPoints.length;
  }

  finish(): BandResult {
    const sorted = sortBands(this.bands);
    // Prefer the driver-provided distances; fall back to recomputed ones.
    const distances =
      this.distances.length === this.kPoints.length
        ? this.distances
        : kPathDistances(this.kPoints);
    const nBands = sorted.length > 0 ? sorted[0].length : 0;
    return {
      freqs: sorted,
      distances,
      k_points: this.kPoints,
      lattice_type: this.meta.lattice_type,
      polarization: this.meta.polarization,
      epsilon_background: this.meta.epsilon_background,
      epsilon_atoms: this.meta.epsilon_atoms,
      radius_atom: this.meta.radius_atom,
      resolution: this.meta.resolution,
      n_bands: nBands,
      n_kpoints: this.kPoints.length,
      k_labels: this.meta.k_labels,
      k_label_distances: this.meta.k_label_distances,
    };
  }
}
