/**
 * TOML config generation for the WASM bulk driver.
 *
 * This is a faithful TypeScript port of the config-generation logic in the
 * Python high-level API (`crates/python/blaze/solve.py`). The goal is strict
 * alignment: the same parameters produce the same TOML and therefore the same
 * computation, whether run via the Python bindings or the in-browser WASM
 * driver.
 *
 * Unlike the Python `solve()` (which routes single runs through
 * `OperatorDataExtractor`), the browser always runs through `WasmBulkDriver`
 * fed a generated TOML string — even for a single (non-sweep) calculation.
 */

// ---------------------------------------------------------------------------
// Lattice presets (mirror solve.py)
// ---------------------------------------------------------------------------

export const LATTICE_VECTORS: Record<string, [[number, number], [number, number]]> = {
  square: [[1.0, 0.0], [0.0, 1.0]],
  hexagonal: [[1.0, 0.0], [0.5, Math.sqrt(3) / 2]],
  rectangular: [[1.0, 0.0], [0.0, 1.5]],
};

export const K_PATH_CORNERS: Record<string, number[][]> = {
  square: [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0]],
  hexagonal: [[0, 0], [0.5, 0], [1 / 3, 1 / 3], [0, 0]],
  rectangular: [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0]],
};

export const K_PATH_LABELS: Record<string, string[]> = {
  square: ['Γ', 'X', 'M', 'Γ'],
  hexagonal: ['Γ', 'M', 'K', 'Γ'],
  rectangular: ['Γ', 'X', 'S', 'Y', 'Γ'],
};

const LATTICE_ALIASES: Record<string, string> = {
  square: 'square',
  hex: 'hexagonal',
  hexagonal: 'hexagonal',
  triangular: 'hexagonal',
  rectangular: 'rectangular',
  rect: 'rectangular',
};

// ---------------------------------------------------------------------------
// Parameter / sweep types
// ---------------------------------------------------------------------------

/** A 3-element [start, stop, step] sweep specification. */
export type SweepSpec = [number, number, number];

export type ScalarOrSweep = number | SweepSpec;

export interface AtomSpec {
  pos: [number, number];
  radius: number;
  eps_inside: number;
}

export interface SolveParams {
  lattice_type?: string;
  resolution?: number | SweepSpec;
  epsilon_background?: number | SweepSpec;
  epsilon_atoms?: number | SweepSpec;
  radius_atom?: number | SweepSpec;
  polarization?: string | string[];
  n_bands?: number;
  points_per_segment?: number;
  /** Custom k-path in fractional reciprocal coords; overrides preset. */
  k_path?: number[][] | null;
  threads?: number;
  /**
   * Optional explicit multi-atom basis. When provided, overrides the default
   * single atom at the origin. Sweeps over per-atom parameters are not
   * supported in this mode (use it for fixed multi-atom geometries).
   */
  atoms?: AtomSpec[];
}

interface SweepDef {
  parameter: string;
  values: (number | string)[];
}

export interface PreparedSolve {
  lattice: string;
  toml: string;
  hasSweep: boolean;
  sweepDefs: SweepDef[];
  /** k-points actually used (custom path) or null when a preset is used. */
  kPath: number[][] | null;
  kLabels: string[];
  kLabelDistances: number[];
  pointsPerSegment: number;
  base: {
    eps_bg: number;
    eps_atom: number;
    radius: number;
    polarization: string;
    resolution: number;
  };
  nBands: number;
  totalJobs: number;
}

// ---------------------------------------------------------------------------
// Helpers (mirror solve.py)
// ---------------------------------------------------------------------------

export function resolveLattice(name: string): string {
  const key = name.trim().toLowerCase();
  const resolved = LATTICE_ALIASES[key];
  if (!resolved) {
    const allowed = Object.keys(LATTICE_ALIASES).sort().join(', ');
    throw new Error(`Unknown lattice_type '${name}'. Choose from: ${allowed}`);
  }
  return resolved;
}

function isSweep(val: unknown): val is SweepSpec {
  return Array.isArray(val) && val.length === 3 && val.every((v) => typeof v === 'number');
}

function isPolList(val: unknown): val is string[] {
  return Array.isArray(val) && val.every((v) => typeof v === 'string');
}

function round10(x: number): number {
  return Math.round(x * 1e10) / 1e10;
}

/** Convert [start, stop, step] → inclusive list of values. */
export function buildRange(spec: SweepSpec, name: string): number[] {
  const [start, stop, step] = spec;
  if (step === 0) throw new Error(`${name}: step cannot be zero`);
  if (stop - start !== 0 && Math.sign(stop - start) !== Math.sign(step)) {
    throw new Error(
      `${name}: step sign (${step}) does not match direction start=${start} → stop=${stop}`,
    );
  }

  const values: number[] = [];
  let current = start;
  if (step > 0) {
    while (current <= stop + 1e-12) {
      values.push(round10(current));
      current += step;
    }
  } else {
    while (current >= stop - 1e-12) {
      values.push(round10(current));
      current += step;
    }
  }
  if (Math.abs(values[values.length - 1] - stop) > 1e-12) {
    values.push(round10(stop));
  }
  return values;
}

function validateRadius(values: number[]): void {
  for (const v of values) {
    if (v <= 0 || v >= 0.5) {
      throw new Error(`radius_atom=${v} out of range. Must be in (0, 0.5) (units of a).`);
    }
  }
}

function validateEpsilon(values: number[], name: string): void {
  for (const v of values) {
    if (v < 1.0) throw new Error(`${name}=${v} invalid. Permittivity must be ≥ 1.`);
  }
}

function validateResolution(values: number[]): void {
  for (const v of values) {
    if (v < 4) throw new Error(`resolution=${v} too small. Must be ≥ 4.`);
    if (v > 512) throw new Error(`resolution=${v} too large. Must be ≤ 512.`);
  }
}

/** Linearly interpolate between high-symmetry corners (mirror _make_kpath). */
export function makeKPath(corners: number[][], nPerSeg: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < corners.length - 1; i++) {
    const s = corners[i];
    const e = corners[i + 1];
    for (let j = 0; j < nPerSeg; j++) {
      const t = j / nPerSeg;
      pts.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
    }
  }
  let last = [...corners[corners.length - 1]];
  const first = corners[0];
  const closed = Math.abs(first[0] - last[0]) < 1e-12 && Math.abs(first[1] - last[1]) < 1e-12;
  if (closed) {
    last = last.map((c) => c + 1e-10);
  }
  pts.push(last);
  return pts;
}

/** Cumulative Euclidean distance along a k-path. */
export function kPathDistances(kpts: number[][]): number[] {
  const out = [0.0];
  for (let i = 1; i < kpts.length; i++) {
    const dx = kpts[i][0] - kpts[i - 1][0];
    const dy = kpts[i][1] - kpts[i - 1][1];
    out.push(out[i - 1] + Math.hypot(dx, dy));
  }
  return out;
}

// ---------------------------------------------------------------------------
// TOML generation (mirror _generate_sweep_toml)
// ---------------------------------------------------------------------------

interface TomlOptions {
  lattice: string;
  baseEpsBg: number;
  baseEpsAtom: number;
  baseRadius: number;
  basePol: string;
  sweeps: SweepDef[];
  nBands: number;
  resolution: number;
  pointsPerSegment: number;
  kPath: number[][] | null;
  threads: number;
  atoms?: AtomSpec[];
}

function fmtNum(v: number): string {
  // Keep integers as-is, but emit floats with a decimal so TOML parses f64.
  return Number.isInteger(v) ? `${v}.0` : `${v}`;
}

export function generateSweepToml(opts: TomlOptions): string {
  // Schema v2: base values for non-swept parameters live in the main
  // sections (eps_bg in [geometry], resolution in [grid], polarization in
  // [solver]); there is no [defaults] section.
  const lines: string[] = [
    'schema = 2',
    '',
    '[run]',
    `threads = ${opts.threads}`,
    'verbose = false',
    '',
    '[solver]',
    'type = "maxwell"',
    `polarization = "${opts.basePol}"`,
    '',
  ];

  lines.push('[geometry]');
  lines.push(`eps_bg = ${opts.baseEpsBg}`);
  lines.push('');
  lines.push('[geometry.lattice]');
  lines.push(`type = "${opts.lattice}"`);
  lines.push('a = 1.0');
  if (opts.lattice === 'rectangular') {
    // v2 requires b explicitly for rectangular lattices.
    lines.push('b = 1.5');
  }
  lines.push('');

  const atoms = opts.atoms ?? [
    { pos: [0.0, 0.0] as [number, number], radius: opts.baseRadius, eps_inside: opts.baseEpsAtom },
  ];
  for (const atom of atoms) {
    lines.push('[[geometry.atoms]]');
    lines.push(`pos = [${fmtNum(atom.pos[0])}, ${fmtNum(atom.pos[1])}]`);
    lines.push(`radius = ${atom.radius}`);
    lines.push(`eps_inside = ${atom.eps_inside}`);
    lines.push('');
  }

  lines.push('[grid]');
  lines.push(`nx = ${opts.resolution}`);
  lines.push('');

  lines.push('[path]');
  if (opts.kPath !== null) {
    const ptsStr = opts.kPath.map((p) => `[${p[0]}, ${p[1]}]`).join(', ');
    lines.push(`points = [${ptsStr}]`);
  } else {
    lines.push('preset = "auto"');
    lines.push(`points_per_segment = ${opts.pointsPerSegment}`);
  }
  lines.push('');

  for (const sw of opts.sweeps) {
    lines.push('[[sweeps]]');
    lines.push(`parameter = "${sw.parameter}"`);
    const vals = sw.values;
    const valsStr =
      typeof vals[0] === 'string'
        ? vals.map((v) => `"${v}"`).join(', ')
        : vals.map((v) => `${v}`).join(', ');
    lines.push(`values = [${valsStr}]`);
    lines.push('');
  }

  lines.push('[eigensolver]');
  lines.push(`n_bands = ${opts.nBands}`);
  lines.push('max_iter = 200');
  lines.push('tol = 1e-4');
  lines.push('');

  lines.push('[output]');
  lines.push('mode = "full"');

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Top-level preparation (mirror solve() parameter parsing)
// ---------------------------------------------------------------------------

export function prepareSolve(params: SolveParams): PreparedSolve {
  const lattice = resolveLattice(params.lattice_type ?? 'square');

  const sweepDefs: SweepDef[] = [];
  let hasSweep = false;

  // resolution
  let baseResolution: number;
  if (isSweep(params.resolution)) {
    const vals = buildRange(params.resolution, 'resolution').map((v) => Math.round(v));
    validateResolution(vals);
    sweepDefs.push({ parameter: 'resolution', values: vals });
    baseResolution = vals[0];
    hasSweep = true;
  } else {
    baseResolution = Math.round((params.resolution as number) ?? 32);
    validateResolution([baseResolution]);
  }

  // epsilon_background
  let baseEpsBg: number;
  if (isSweep(params.epsilon_background)) {
    const vals = buildRange(params.epsilon_background, 'epsilon_background');
    validateEpsilon(vals, 'epsilon_background');
    sweepDefs.push({ parameter: 'eps_bg', values: vals });
    baseEpsBg = vals[0];
    hasSweep = true;
  } else {
    baseEpsBg = (params.epsilon_background as number) ?? 1.0;
    validateEpsilon([baseEpsBg], 'epsilon_background');
  }

  // epsilon_atoms
  let baseEpsAtom: number;
  if (isSweep(params.epsilon_atoms)) {
    const vals = buildRange(params.epsilon_atoms, 'epsilon_atoms');
    validateEpsilon(vals, 'epsilon_atoms');
    sweepDefs.push({ parameter: 'atom0.eps_inside', values: vals });
    baseEpsAtom = vals[0];
    hasSweep = true;
  } else {
    baseEpsAtom = (params.epsilon_atoms as number) ?? 8.9;
    validateEpsilon([baseEpsAtom], 'epsilon_atoms');
  }

  // radius_atom
  let baseRadius: number;
  if (isSweep(params.radius_atom)) {
    const vals = buildRange(params.radius_atom, 'radius_atom');
    validateRadius(vals);
    sweepDefs.push({ parameter: 'atom0.radius', values: vals });
    baseRadius = vals[0];
    hasSweep = true;
  } else {
    baseRadius = (params.radius_atom as number) ?? 0.2;
    validateRadius([baseRadius]);
  }

  // polarization
  let basePol: string;
  if (isPolList(params.polarization)) {
    for (const p of params.polarization) {
      if (!['TM', 'TE'].includes(p.toUpperCase())) {
        throw new Error(`Unknown polarization '${p}'. Use 'TM' or 'TE'.`);
      }
    }
    sweepDefs.push({
      parameter: 'polarization',
      values: params.polarization.map((p) => p.toUpperCase()),
    });
    basePol = params.polarization[0].toUpperCase();
    hasSweep = true;
  } else {
    basePol = String(params.polarization ?? 'TM').toUpperCase();
    if (!['TM', 'TE'].includes(basePol)) {
      throw new Error(`Unknown polarization '${params.polarization}'. Use 'TM' or 'TE'.`);
    }
  }

  const nBands = params.n_bands ?? 8;
  const pps = params.points_per_segment ?? 15;
  const threads = params.threads ?? 0;

  // k-path + labels
  let kPath: number[][] | null;
  let kLabels: string[];
  let kLabelDistances: number[];
  if (params.k_path != null) {
    kPath = params.k_path;
    kLabels = [];
    kLabelDistances = [];
  } else {
    const corners = K_PATH_CORNERS[lattice];
    const kpts = makeKPath(corners, pps);
    kLabels = K_PATH_LABELS[lattice];
    const dists = kPathDistances(kpts);
    kLabelDistances = [];
    for (let i = 0; i < corners.length - 1; i++) {
      kLabelDistances.push(dists[i * pps]);
    }
    kLabelDistances.push(dists[dists.length - 1]);
    // For a preset, the driver generates the path internally; pass null path.
    kPath = null;
  }

  const toml = generateSweepToml({
    lattice,
    baseEpsBg,
    baseEpsAtom,
    baseRadius,
    basePol,
    sweeps: sweepDefs,
    nBands,
    resolution: baseResolution,
    pointsPerSegment: pps,
    kPath: params.k_path ?? null,
    threads,
    atoms: params.atoms,
  });

  let totalJobs = 1;
  for (const sw of sweepDefs) totalJobs *= sw.values.length;

  return {
    lattice,
    toml,
    hasSweep,
    sweepDefs,
    kPath,
    kLabels,
    kLabelDistances,
    pointsPerSegment: pps,
    base: {
      eps_bg: baseEpsBg,
      eps_atom: baseEpsAtom,
      radius: baseRadius,
      polarization: basePol,
      resolution: baseResolution,
    },
    nBands,
    totalJobs,
  };
}
