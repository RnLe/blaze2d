/**
 * Parse schema v2 TOML text into a `StudioConfig`.
 *
 * This does structural mapping only (shape, defaults). Physical validity is
 * delegated to the WASM `validateConfig`, so this parser is lenient: unknown
 * or malformed pieces fall back to defaults rather than throwing. Returns null
 * only when the text is not valid TOML syntax at all.
 */

import { parse as parseTomlText } from 'smol-toml';
import {
  defaultConfig,
  type StudioConfig,
  type LatticeKind,
  type Polarization,
  type Precision,
  type SolverType,
  type PathPreset,
  type SmoothingKind,
  type StudioAtom,
  type StudioSweep,
} from './configModel';

type Obj = Record<string, unknown>;

function asObj(v: unknown): Obj | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : null;
}
function num(v: unknown, fallback: number): number {
  return typeof v === 'number' ? v : fallback;
}
function bool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback;
}
function str<T extends string>(v: unknown, fallback: T): T {
  return typeof v === 'string' ? (v as T) : fallback;
}
function vec2(v: unknown, fallback: [number, number]): [number, number] {
  if (Array.isArray(v) && v.length >= 2 && typeof v[0] === 'number' && typeof v[1] === 'number') {
    return [v[0], v[1]];
  }
  return fallback;
}

export function parseToml(text: string): StudioConfig | null {
  let doc: Obj;
  try {
    doc = parseTomlText(text) as Obj;
  } catch {
    return null;
  }

  const c = defaultConfig();

  // [run]
  const run = asObj(doc.run);
  if (run) {
    c.run.threads = typeof run.threads === 'number' ? run.threads : null;
    c.run.verbose = bool(run.verbose, false);
    c.run.skip_final_gamma = bool(run.skip_final_gamma, false);
    c.run.disable_band_tracking = bool(run.disable_band_tracking, false);
  }

  // [solver]
  const solver = asObj(doc.solver);
  if (solver) {
    c.solver.type = str<SolverType>(solver.type, 'maxwell');
    if (solver.type === 'ea') c.solver.type = 'operator_data';
    c.solver.precision = str<Precision>(solver.precision, 'f64');
    c.solver.polarization = str<Polarization>(solver.polarization, 'TM');
  }

  // [geometry]
  const geo = asObj(doc.geometry);
  if (geo) {
    c.geometry.eps_bg = num(geo.eps_bg, 1.0);
    const lat = asObj(geo.lattice);
    if (lat) {
      const kind = str<LatticeKind>(lat.type, 'square');
      c.geometry.lattice.kind = kind;
      c.geometry.lattice.a = num(lat.a, 1.0);
      c.geometry.lattice.b = typeof lat.b === 'number' ? lat.b : null;
      c.geometry.lattice.alpha_deg = typeof lat.alpha_deg === 'number' ? lat.alpha_deg : null;
      c.geometry.lattice.a1 = lat.a1 !== undefined ? vec2(lat.a1, [1, 0]) : null;
      c.geometry.lattice.a2 = lat.a2 !== undefined ? vec2(lat.a2, [0, 1]) : null;
    }
    const atoms = geo.atoms;
    if (Array.isArray(atoms) && atoms.length > 0) {
      c.geometry.atoms = atoms.map((a): StudioAtom => {
        const o = asObj(a) ?? {};
        return {
          pos: vec2(o.pos, [0.5, 0.5]),
          radius: num(o.radius, 0.2),
          eps_inside: num(o.eps_inside, 1.0),
        };
      });
    }
  }

  // [grid]
  const grid = asObj(doc.grid);
  if (grid) {
    c.grid.nx = num(grid.nx, 32);
    c.grid.ny = typeof grid.ny === 'number' ? grid.ny : null;
    c.grid.lx = num(grid.lx, 1.0);
    c.grid.ly = num(grid.ly, 1.0);
    c.grid.centered = bool(grid.centered, false);
  }

  // [path]
  const path = asObj(doc.path);
  if (path) {
    if (Array.isArray(path.points) && path.points.length > 0) {
      c.path.mode = 'points';
      c.path.points = (path.points as unknown[]).map((p) => vec2(p, [0, 0]));
    } else {
      c.path.mode = 'preset';
      c.path.preset = str<PathPreset>(path.preset, 'auto');
      c.path.pointsPerSegment = num(path.points_per_segment, 12);
    }
  }

  // [[sweeps]]
  if (Array.isArray(doc.sweeps)) {
    c.sweeps = (doc.sweeps as unknown[]).map((s): StudioSweep => {
      const o = asObj(s) ?? {};
      const hasRange = o.min !== undefined && o.max !== undefined && o.step !== undefined;
      return {
        parameter: str(o.parameter, 'eps_bg'),
        mode: hasRange ? 'range' : 'values',
        min: num(o.min, 0),
        max: num(o.max, 1),
        step: num(o.step, 0.1),
        values: Array.isArray(o.values) ? (o.values as (number | string)[]) : [],
      };
    });
  }

  // [eigensolver]
  const eig = asObj(doc.eigensolver);
  if (eig) {
    c.eigensolver.n_bands = num(eig.n_bands, 8);
    c.eigensolver.max_iter = num(eig.max_iter, 200);
    c.eigensolver.tol = num(eig.tol, 1e-6);
    c.eigensolver.block_size = num(eig.block_size, 0);
    c.eigensolver.record_diagnostics = bool(eig.record_diagnostics, false);
  }

  // [dielectric]
  const diel = asObj(doc.dielectric);
  if (diel) {
    c.dielectric.smoothing = str<SmoothingKind>(diel.smoothing, 'analytic');
    c.dielectric.mesh_size = num(diel.mesh_size, 3);
    c.dielectric.interface_tolerance = num(diel.interface_tolerance, 1e-6);
  }

  // [output]
  const output = asObj(doc.output);
  if (output) {
    c.output.mode = str(output.mode, 'full');
    if (typeof output.directory === 'string') c.output.directory = output.directory;
    if (typeof output.filename === 'string') c.output.filename = output.filename;
    if (typeof output.prefix === 'string') c.output.prefix = output.prefix;
    const sel = asObj(output.selective);
    if (sel) {
      if (Array.isArray(sel.k_indices)) c.output.selective.k_indices = sel.k_indices as number[];
      if (Array.isArray(sel.k_labels)) c.output.selective.k_labels = sel.k_labels as string[];
      if (Array.isArray(sel.bands)) c.output.selective.bands = sel.bands as number[];
    }
  }

  // [operator_data]
  const od = asObj(doc.operator_data) ?? asObj(doc.ea);
  if (od) {
    c.operatorData.k0 = vec2(od.k0, [0, 0]);
    c.operatorData.n_retained = num(od.n_retained, 4);
    c.operatorData.n_remote = num(od.n_remote, 8);
    c.operatorData.compute_mass_tensor = bool(od.compute_mass_tensor, true);
    c.operatorData.compute_born_huang = bool(od.compute_born_huang, false);
    c.operatorData.compute_slow_coefficient = bool(od.compute_slow_coefficient, false);
    c.operatorData.compute_r_derivatives = bool(od.compute_r_derivatives, true);
    c.operatorData.atom_index = num(od.atom_index, 0);
    c.operatorData.fd_step = num(od.fd_step, 0.001);
  }

  return c;
}
