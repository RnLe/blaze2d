/**
 * The canonical Workbench Studio configuration model.
 *
 * `StudioConfig` mirrors the Rust schema v2 `Config`
 * (`crates/bulk-driver-core/src/config.rs`) field for field. It is the single
 * source of truth for the studio: the structured editor reads and writes it,
 * `tomlSerialize` turns it into TOML text, and `tomlParse` reads TOML back into
 * it. Authoritative validation is delegated to the WASM `validateConfig`
 * (the real Rust parser); the clamp constants here only drive instant slider
 * bounds and never replace the Rust rules.
 */

export type SolverType = 'maxwell' | 'operator_data';
export type Precision = 'f64' | 'f32';
export type Polarization = 'TM' | 'TE';
export type LatticeKind = 'square' | 'rectangular' | 'triangular' | 'oblique' | 'custom';
export type PathPreset = 'auto' | 'square' | 'rectangular' | 'triangular' | 'hexagonal';
export type SmoothingKind = 'analytic' | 'subgrid' | 'none';
export type OutputMode = 'full' | 'selective';

export interface StudioAtom {
  pos: [number, number];
  radius: number;
  eps_inside: number;
}

export interface StudioLattice {
  kind: LatticeKind;
  a: number;
  /** Required for rectangular and oblique. */
  b: number | null;
  /** Required for oblique; degrees, in (0, 180). */
  alpha_deg: number | null;
  /** custom lattices only. */
  a1: [number, number] | null;
  a2: [number, number] | null;
}

export interface StudioSweep {
  parameter: string;
  /** 'range' uses min/max/step; 'values' uses the discrete list. */
  mode: 'range' | 'values';
  min: number;
  max: number;
  step: number;
  values: (number | string)[];
}

export interface StudioConfig {
  run: {
    threads: number | null;
    verbose: boolean;
    skip_final_gamma: boolean;
    disable_band_tracking: boolean;
  };
  solver: {
    type: SolverType;
    precision: Precision;
    polarization: Polarization;
  };
  geometry: {
    eps_bg: number;
    lattice: StudioLattice;
    atoms: StudioAtom[];
  };
  grid: {
    nx: number;
    /** null means "same as nx". */
    ny: number | null;
    lx: number;
    ly: number;
    centered: boolean;
  };
  path: {
    /** 'preset' emits [path].preset; 'points' emits [path].points. */
    mode: 'preset' | 'points';
    preset: PathPreset;
    pointsPerSegment: number;
    points: [number, number][];
  };
  sweeps: StudioSweep[];
  eigensolver: {
    n_bands: number;
    max_iter: number;
    tol: number;
    /** 0 = auto. */
    block_size: number;
    record_diagnostics: boolean;
  };
  dielectric: {
    smoothing: SmoothingKind;
    mesh_size: number;
    interface_tolerance: number;
  };
  output: {
    mode: OutputMode;
    directory: string;
    filename: string;
    prefix: string;
    selective: {
      k_indices: number[];
      k_labels: string[];
      bands: number[];
    };
  };
  operatorData: {
    k0: [number, number];
    n_retained: number;
    n_remote: number;
    compute_mass_tensor: boolean;
    compute_born_huang: boolean;
    compute_slow_coefficient: boolean;
    compute_r_derivatives: boolean;
    atom_index: number;
    fd_step: number;
  };
}

// ---------------------------------------------------------------------------
// Clamp constants (mirror the Rust physical-validation ranges; UI only)
// ---------------------------------------------------------------------------

export const LIMITS = {
  epsMin: 1.0,
  epsMax: 30.0,
  radiusMin: 0.01,
  radiusMax: 0.49,
  resolutionMin: 4,
  resolutionMax: 512,
  nBandsMin: 1,
  nBandsMax: 60,
  posMin: 0.0,
  posMax: 0.999,
  alphaMin: 1,
  alphaMax: 179,
} as const;

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

// ---------------------------------------------------------------------------
// Default config: a classic square-lattice dielectric-rods-in-air crystal.
// ---------------------------------------------------------------------------

export function defaultConfig(): StudioConfig {
  return {
    run: {
      threads: null,
      verbose: false,
      skip_final_gamma: false,
      disable_band_tracking: false,
    },
    solver: {
      type: 'maxwell',
      precision: 'f64',
      polarization: 'TM',
    },
    geometry: {
      eps_bg: 1.0,
      lattice: {
        kind: 'square',
        a: 1.0,
        b: null,
        alpha_deg: null,
        a1: null,
        a2: null,
      },
      atoms: [{ pos: [0.5, 0.5], radius: 0.2, eps_inside: 8.9 }],
    },
    grid: {
      nx: 32,
      ny: null,
      lx: 1.0,
      ly: 1.0,
      centered: false,
    },
    path: {
      mode: 'preset',
      preset: 'auto',
      pointsPerSegment: 12,
      points: [
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.0],
      ],
    },
    sweeps: [],
    eigensolver: {
      n_bands: 8,
      max_iter: 200,
      tol: 1e-6,
      block_size: 0,
      record_diagnostics: false,
    },
    dielectric: {
      smoothing: 'analytic',
      mesh_size: 3,
      interface_tolerance: 1e-6,
    },
    output: {
      mode: 'full',
      directory: './bulk_output',
      filename: './bulk_results.csv',
      prefix: 'job',
      selective: {
        k_indices: [],
        k_labels: [],
        bands: [],
      },
    },
    operatorData: {
      k0: [0.0, 0.0],
      n_retained: 4,
      n_remote: 8,
      compute_mass_tensor: true,
      compute_born_huang: false,
      compute_slow_coefficient: false,
      compute_r_derivatives: true,
      atom_index: 0,
      fd_step: 0.001,
    },
  };
}

/** Structural deep clone (config is plain data). */
export function cloneConfig(c: StudioConfig): StudioConfig {
  return structuredClone(c);
}

/** Effective ny (falls back to nx). */
export function effectiveNy(c: StudioConfig): number {
  return c.grid.ny ?? c.grid.nx;
}

/** Valid sweep parameter names for the current geometry. */
export function sweepParameterOptions(c: StudioConfig): string[] {
  const globals = ['eps_bg', 'resolution', 'polarization', 'lattice_type'];
  const atomProps = ['radius', 'pos_x', 'pos_y', 'eps_inside'];
  const atomParams = c.geometry.atoms.flatMap((_, i) =>
    atomProps.map((p) => `atom${i}.${p}`),
  );
  return [...globals, ...atomParams];
}
