import type { Config, ResolvedConfig } from '@/lib/contract/generated';
import type { ExecutionState } from '@/lib/compute/controller';
import { object } from '@/lib/contract/records';

export const CRYSTALS = ['square', 'triangular', 'honeycomb', 'rectangular'] as const;
export type Crystal = typeof CRYSTALS[number];
export type Polarization = 'TM' | 'TE';
export type DemoSettings = { crystal: Crystal; radius: number; background: number; inclusion: number; ratio: number; bands: number };
export const DEFAULT_SETTINGS: DemoSettings = { crystal: 'square', radius: 0.2, background: 7, inclusion: 1, ratio: 0.5, bands: 8 };
export const EXTRA_BANDS = 4;
export const PREVIEW_PERIODS = 7;
export type DemoPath = Pick<ResolvedConfig, 'distances' | 'k_labels' | 'k_label_indices'>;
export type DemoSeries = { polarization: Polarization; distances: ArrayLike<number>; frequencies: number[]; bands: number };

export function demoConfig(settings: DemoSettings): Config {
  const lattice = settings.crystal === 'honeycomb' ? 'triangular' : settings.crystal;
  // Honeycomb has a two-site basis on the solver's 60-degree triangular lattice.
  const { centers } = crystalLattice(settings);
  return {
    schema: 'blaze2d/1', task: 'bands', polarization: 'TM',
    geometry: {
      lattice: lattice === 'rectangular' ? { type: lattice, a: 1, b: 1 / settings.ratio } : { type: lattice, a: 1 },
      background_epsilon: settings.background,
      objects: centers.map((center, index) => ({ name: `circle_${index + 1}`, kind: 'circle', center, radius: settings.radius, epsilon: settings.inclusion })),
    },
    grid: { resolution: 32 },
    bands: { count: settings.bands + EXTRA_BANDS, path: { preset: lattice, intervals_per_segment: 15 } },
    eigensolver: { precision: 'f64' }, results: { eigenvectors: false },
    sweeps: [{ name: 'polarization', target: 'polarization', values: ['TM', 'TE'] }],
  };
}

/** Drop the guard bands separately in every row, for both streamed and final data. */
export function visibleFrequencies(values: ArrayLike<number>, computed: number, shown: number): number[] {
  const visible = Math.min(computed, shown), rows = Math.floor(values.length / computed), result: number[] = [];
  for (let row = 0; row < rows; row++) for (let band = 0; band < visible; band++) result.push(values[row * computed + band]);
  return result;
}

export function demoSeries(state: ExecutionState, polarization: Polarization, shown: number): DemoSeries | undefined {
  const result = state.run?.results.find(result => object(object(result.metadata).sweep_parameters).polarization === polarization);
  if (result?.arrays.frequencies && result.arrays.distances) {
    const { data, shape } = result.arrays.frequencies;
    return { polarization, distances: result.arrays.distances.data, frequencies: visibleFrequencies(data, shape[1], shown), bands: Math.min(shape[1], shown) };
  }
  if (state.job?.resolved.config.polarization === polarization && state.live.length) {
    const computed = state.live[0].frequencies.length;
    return { polarization, distances: state.live.map(point => point.distance),
      frequencies: visibleFrequencies(state.live.flatMap(point => point.frequencies), computed, shown), bands: Math.min(computed, shown) };
  }
}

type Vector = [number, number];
export function crystalLattice(settings: Pick<DemoSettings, 'crystal' | 'ratio'>): { vectors: [Vector, Vector]; centers: Vector[] } {
  const triangular = settings.crystal === 'triangular' || settings.crystal === 'honeycomb';
  return {
    vectors: [[1, 0], triangular ? [0.5, Math.sqrt(3) / 2] : [0, settings.crystal === 'rectangular' ? 1 / settings.ratio : 1]],
    centers: settings.crystal === 'honeycomb' ? [[0, 0], [1 / 3, 1 / 3]] : [[0, 0]],
  };
}

/** Tile exact circles over the square view, including objects crossing its edges. */
export function analyticGeometry(settings: Pick<DemoSettings, 'crystal' | 'ratio' | 'radius'>) {
  const { vectors: [a, b], centers } = crystalLattice(settings);
  const span = PREVIEW_PERIODS * Math.min(Math.hypot(...a), Math.hypot(...b));
  const extent = span / 2 + settings.radius, determinant = a[0] * b[1] - a[1] * b[0];
  const corners = [-extent, extent].flatMap(x => [-extent, extent].map(y =>
    [(b[1] * x - b[0] * y) / determinant, (a[0] * y - a[1] * x) / determinant]));
  const minU = Math.floor(Math.min(...corners.map(point => point[0]))) - 1;
  const maxU = Math.ceil(Math.max(...corners.map(point => point[0])));
  const minV = Math.floor(Math.min(...corners.map(point => point[1]))) - 1;
  const maxV = Math.ceil(Math.max(...corners.map(point => point[1])));
  const circles: { key: string; x: number; y: number }[] = [];
  for (let u = minU; u <= maxU; u++) for (let v = minV; v <= maxV; v++) {
    centers.forEach(([cu, cv], index) => {
      const x = (u + cu) * a[0] + (v + cv) * b[0], y = (u + cu) * a[1] + (v + cv) * b[1];
      if (Math.abs(x) <= extent && Math.abs(y) <= extent) circles.push({ key: `${u}:${v}:${index}`, x, y });
    });
  }
  return { span, circles };
}
