/**
 * Serialize a `StudioConfig` into deterministic, commented schema v2 TOML.
 *
 * Hand-rolled (rather than a generic TOML encoder) so we control key order,
 * float formatting, and section comments, matching the style of the hand
 * written example configs and `configGen.ts`. `schema = 2` is always first.
 */

import type { StudioConfig } from './configModel';

/** Format a number so TOML parses it as the intended type. */
function num(v: number): string {
  if (Number.isInteger(v)) return `${v}`;
  return `${v}`;
}

/** Format a float, always with a decimal point so it round-trips as f64. */
function float(v: number): string {
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

function vec(v: [number, number]): string {
  return `[${float(v[0])}, ${float(v[1])}]`;
}

function tomlString(s: string): string {
  return JSON.stringify(s); // safe TOML basic string for our value set
}

export function serializeConfig(c: StudioConfig): string {
  const L: string[] = [];
  const push = (s = '') => L.push(s);

  push('schema = 2');
  push();

  // [run] — only emit when it deviates from defaults, to keep files lean.
  const run = c.run;
  const runNeeded =
    run.threads !== null ||
    run.verbose ||
    run.skip_final_gamma ||
    run.disable_band_tracking;
  if (runNeeded) {
    push('[run]');
    if (run.threads !== null) push(`threads = ${num(run.threads)}`);
    if (run.verbose) push('verbose = true');
    if (run.skip_final_gamma) push('skip_final_gamma = true');
    if (run.disable_band_tracking) push('disable_band_tracking = true');
    push();
  }

  // [solver]
  push('[solver]');
  push(`type = ${tomlString(c.solver.type)}`);
  push(`precision = ${tomlString(c.solver.precision)}`);
  push(`polarization = ${tomlString(c.solver.polarization)}`);
  push();

  // [geometry]
  push('[geometry]');
  push(`eps_bg = ${float(c.geometry.eps_bg)}`);
  push();

  // [geometry.lattice]
  const lat = c.geometry.lattice;
  push('[geometry.lattice]');
  push(`type = ${tomlString(lat.kind)}`);
  if (lat.kind === 'custom') {
    push(`a1 = ${vec(lat.a1 ?? [1, 0])}`);
    push(`a2 = ${vec(lat.a2 ?? [0, 1])}`);
  } else {
    push(`a = ${float(lat.a)}`);
    if (lat.kind === 'rectangular' || lat.kind === 'oblique') {
      push(`b = ${float(lat.b ?? lat.a * 1.5)}`);
    }
    if (lat.kind === 'oblique') {
      push(`alpha_deg = ${float(lat.alpha_deg ?? 75)}`);
    }
  }
  push();

  // [[geometry.atoms]]
  for (const atom of c.geometry.atoms) {
    push('[[geometry.atoms]]');
    push(`pos = ${vec(atom.pos)}`);
    push(`radius = ${float(atom.radius)}`);
    push(`eps_inside = ${float(atom.eps_inside)}`);
    push();
  }

  // [grid]
  push('[grid]');
  push(`nx = ${num(c.grid.nx)}`);
  if (c.grid.ny !== null && c.grid.ny !== c.grid.nx) push(`ny = ${num(c.grid.ny)}`);
  if (c.grid.lx !== 1.0) push(`lx = ${float(c.grid.lx)}`);
  if (c.grid.ly !== 1.0) push(`ly = ${float(c.grid.ly)}`);
  if (c.grid.centered) push('centered = true');
  push();

  // [path] — only for the Maxwell solver.
  if (c.solver.type === 'maxwell') {
    push('[path]');
    if (c.path.mode === 'points') {
      const pts = c.path.points.map((p) => vec(p)).join(', ');
      push(`points = [${pts}]`);
    } else {
      push(`preset = ${tomlString(c.path.preset)}`);
      push(`points_per_segment = ${num(c.path.pointsPerSegment)}`);
    }
    push();
  }

  // [[sweeps]]
  for (const sweep of c.sweeps) {
    push('[[sweeps]]');
    push(`parameter = ${tomlString(sweep.parameter)}`);
    if (sweep.mode === 'range') {
      push(`min = ${float(sweep.min)}`);
      push(`max = ${float(sweep.max)}`);
      push(`step = ${float(sweep.step)}`);
    } else {
      const vals = sweep.values
        .map((v) => (typeof v === 'string' ? tomlString(v) : float(v)))
        .join(', ');
      push(`values = [${vals}]`);
    }
    push();
  }

  // [eigensolver] — emit when it differs from Rust defaults.
  const eig = c.eigensolver;
  const eigNeeded =
    eig.n_bands !== 8 ||
    eig.max_iter !== 200 ||
    eig.tol !== 1e-6 ||
    eig.block_size !== 0 ||
    eig.record_diagnostics;
  if (eigNeeded) {
    push('[eigensolver]');
    push(`n_bands = ${num(eig.n_bands)}`);
    if (eig.max_iter !== 200) push(`max_iter = ${num(eig.max_iter)}`);
    if (eig.tol !== 1e-6) push(`tol = ${eig.tol}`);
    if (eig.block_size !== 0) push(`block_size = ${num(eig.block_size)}`);
    if (eig.record_diagnostics) push('record_diagnostics = true');
    push();
  }

  // [dielectric] — emit when it differs from defaults.
  const diel = c.dielectric;
  if (diel.smoothing !== 'analytic' || diel.mesh_size !== 3 || diel.interface_tolerance !== 1e-6) {
    push('[dielectric]');
    push(`smoothing = ${tomlString(diel.smoothing)}`);
    if (diel.smoothing === 'subgrid') push(`mesh_size = ${num(diel.mesh_size)}`);
    if (diel.interface_tolerance !== 1e-6) {
      push(`interface_tolerance = ${diel.interface_tolerance}`);
    }
    push();
  }

  // [output] — emit only for selective mode (full is the default).
  if (c.output.mode === 'selective') {
    push('[output]');
    push('mode = "selective"');
    push();
    push('[output.selective]');
    if (c.output.selective.k_indices.length > 0) {
      push(`k_indices = [${c.output.selective.k_indices.join(', ')}]`);
    }
    if (c.output.selective.k_labels.length > 0) {
      push(`k_labels = [${c.output.selective.k_labels.map(tomlString).join(', ')}]`);
    }
    push(`bands = [${c.output.selective.bands.join(', ')}]`);
    push();
  }

  // [operator_data] — only when the operator-data solver is selected.
  if (c.solver.type === 'operator_data') {
    const od = c.operatorData;
    push('[operator_data]');
    push(`k0 = ${vec(od.k0)}`);
    push(`n_retained = ${num(od.n_retained)}`);
    push(`n_remote = ${num(od.n_remote)}`);
    push(`compute_mass_tensor = ${od.compute_mass_tensor}`);
    push(`compute_born_huang = ${od.compute_born_huang}`);
    push(`compute_slow_coefficient = ${od.compute_slow_coefficient}`);
    push(`compute_r_derivatives = ${od.compute_r_derivatives}`);
    push(`atom_index = ${num(od.atom_index)}`);
    push(`fd_step = ${od.fd_step}`);
    push();
  }

  // Trim the trailing blank line.
  while (L.length > 0 && L[L.length - 1] === '') L.pop();
  return L.join('\n') + '\n';
}
