import {
  prepareSolve,
  type SolveParams,
} from './configGen';
import type { RunMeta } from './useExampleRunner';
import type { BandResult } from './bandResult';

export type DisplayKind = 'BandResult' | 'list' | 'streamDict';
export type RunMode = 'stream' | 'filtered';

export interface PreparedRun {
  toml: string;
  meta: RunMeta;
  totalJobs: number;
  mode: RunMode;
  kIndices?: number[] | null;
  bandIndices?: number[] | null;
  displayKind: DisplayKind;
  /** Human label describing the returned variable, e.g. "BandResult" or "list[BandResult]". */
  resultLabel: string;
  /** Name shown for the root variable in the inspector. */
  resultVar: string;
}

/** Context handed to an example's `output` formatter to render live stdout. */
export interface OutputContext {
  /** Finalized per-job results (one entry per sweep job). */
  results: BandResult[];
  /** Live snapshot of the active job (progressive). */
  live: BandResult | null;
  /** Raw streamed dicts (k-point dicts, or filtered job dicts). */
  rawStream: Record<string, unknown>[];
  status: string;
}

export interface Example {
  slug: string;
  title: string;
  description: string;
  /** A short context primer shown below the title in the runner.
   *  Holds the core lesson, caveats, or things to watch out for.
   *  Distinct from `description`, which is the card subtitle. */
  primer?: string;
  category: string;
  /** Displayed Python source (syntax-highlighted in the UI). */
  code: string;
  /** File name shown in the explorer / code-window for the Python script. */
  pyFile: string;
  /** When set, a TOML config file is shown alongside the script. */
  tomlFile?: string;
  /** Whether to render the TOML config box (content = prepared TOML). */
  showToml?: boolean;
  /** When true, the runner shows a single-core browser note. */
  singleCore?: boolean;
  /** Build the WASM run configuration for this example. */
  prepare: () => PreparedRun;
  /** Render the live stdout the `print(...)` statements would produce. */
  output?: (ctx: OutputContext) => string | null;
}

export const CATEGORIES = [
  'Getting Started',
  'Lattices & Geometry',
  'Parameter Sweeps',
  'Performance',
  'Streaming & Inspection',
] as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fromSolve(
  params: SolveParams,
  opts: {
    displayKind: DisplayKind;
    resultLabel: string;
    resultVar: string;
    mode?: RunMode;
    kIndices?: number[] | null;
    bandIndices?: number[] | null;
  },
): PreparedRun {
  const prepared = prepareSolve(params);
  const meta: RunMeta = {
    lattice_type: prepared.lattice,
    k_labels: prepared.kLabels,
    k_label_distances: prepared.kLabelDistances,
  };
  return {
    toml: prepared.toml,
    meta,
    totalJobs: prepared.totalJobs,
    mode: opts.mode ?? 'stream',
    kIndices: opts.kIndices ?? null,
    bandIndices: opts.bandIndices ?? null,
    displayKind: opts.displayKind,
    resultLabel: opts.resultLabel,
    resultVar: opts.resultVar,
  };
}

/** Max frequency of band `b` across the k-path. */
function bandMax(r: BandResult, b: number): number {
  let m = -Infinity;
  for (const row of r.freqs) {
    if (row[b] != null && row[b] > m) m = row[b];
  }
  return m;
}

function labelsLine(r: BandResult): string {
  return `[${r.k_labels.map((l) => `'${l}'`).join(', ')}]`;
}

/** Join a capped list of lines (head + tail) so live output stays compact. */
function cappedLines(lines: string[], max = 16): string {
  if (lines.length <= max) return lines.join('\n');
  const head = lines.slice(0, 3);
  const tail = lines.slice(-(max - 4));
  return [...head, `... ${lines.length - (max - 1)} more ...`, ...tail].join('\n');
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const EXAMPLES: Example[] = [
  // ---- Getting Started -----------------------------------------------------
  {
    slug: 'first-band-diagram',
    title: 'Your first band diagram',
    description:
      'Compute the band structure of a simple square-lattice photonic crystal, one crystal, one diagram.',
    primer:
      'Frequencies are returned in dimensionless units of $c/a$, where $a$ is the lattice constant. The k-path is picked automatically from the lattice preset ($\Gamma, X, M, \Gamma$ for a square), and the returned `freqs` array *always* has shape $(n_k, n_\\text{bands})$, sorted ascending in frequency at every k-point.',
    category: 'Getting Started',
    pyFile: 'band_diagram.py',
    code: `import blaze

# A square lattice of high-index rods in air.
result = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,   # air
    epsilon_atoms=8.9,        # dielectric rods
    radius_atom=0.2,          # in units of the lattice constant a
    polarization="TM",
    n_bands=8,
)

print(result.freqs.shape)   # (n_kpoints, 8)
print(result.k_labels)      # ['Γ', 'X', 'M', 'Γ']`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
          n_bands: 8,
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      if (!r) return null;
      return `(${r.n_kpoints}, ${r.n_bands})\n${labelsLine(r)}`;
    },
  },
  {
    slug: 'crystal-from-toml',
    title: 'Drive a crystal from a TOML file',
    description:
      'Describe the whole simulation once in a reproducible TOML config and run it through BulkDriver.',
    primer:
      'TOML configs are the *canonical* way to share a simulation: every parameter (geometry, grid, k-path, solver) lives in one file you can version-control. `BulkDriver.run_streaming()` yields one dict per job, so even a single-job run reads naturally as a `for` loop, ready to grow into a sweep **without rewriting the loop body**.',
    category: 'Getting Started',
    pyFile: 'run.py',
    tomlFile: 'crystal.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# Everything about the crystal lives in a version-controllable TOML file.
driver = BulkDriver("crystal.toml")

# run_streaming() yields one result dict per job (here: a single job).
for result in driver.run_streaming():
    print(result["num_bands"], "bands ×", result["num_k_points"], "k-points")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.25,
          polarization: 'TM',
          n_bands: 8,
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      if (!r) return null;
      return `${r.n_bands} bands × ${r.n_kpoints} k-points`;
    },
  },
  {
    slug: 'tm-vs-te',
    title: 'TM vs TE polarization',
    description:
      'Solve both polarizations in one call and compare their band diagrams side by side.',
    primer:
      'In 2D, Maxwell decouples into two independent scalar problems. **TM** has the electric field out of plane ($E$ along $z$), **TE** has the magnetic field out of plane. The two polarizations see the same geometry but produce very different band structures, especially for high-contrast rod or hole lattices. Passing a list of polarizations expands the call into a sweep, one `BandResult` per polarization.',
    category: 'Getting Started',
    pyFile: 'polarizations.py',
    code: `import blaze

# Pass a list of polarizations to compute both in one sweep.
results = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=0.2,
    polarization=["TM", "TE"],
    n_bands=8,
)

for r in results:
    print(r.polarization, r.freqs[:, 0].max())`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: ['TM', 'TE'],
          n_bands: 8,
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const valid = results.filter(Boolean);
      return valid.length === 0
        ? null
        : valid.map((r) => `${r.polarization} ${bandMax(r, 0).toFixed(4)}`).join('\n');
    },
  },

  // ---- Lattices & Geometry -------------------------------------------------
  {
    slug: 'hexagonal-lattice',
    title: 'Hexagonal lattice',
    description:
      'Switch to a triangular (hexagonal) lattice, the Γ to M to K to Γ path is selected automatically.',
    primer:
      'Changing `lattice_type` swaps in a different Brillouin zone and a new high-symmetry path. The classic TE photonic bandgap in a dielectric slab with air holes lives in this geometry, between the first two bands around the $K$ point. The lattice vectors are normalised so the lattice constant $a$ is still $1$, but the reciprocal lattice (and therefore the k-coordinates) are *different* from the square case.',
    category: 'Lattices & Geometry',
    pyFile: 'hexagonal.py',
    code: `import blaze

# "hex" and "triangular" are accepted aliases for "hexagonal".
result = blaze.solve(
    lattice_type="hexagonal",
    epsilon_background=13.0,
    epsilon_atoms=1.0,        # air holes in a dielectric slab
    radius_atom=0.3,
    polarization="TE",
    n_bands=8,
)

print(result.k_labels)   # ['Γ', 'M', 'K', 'Γ']`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'hexagonal',
          epsilon_background: 13.0,
          epsilon_atoms: 1.0,
          radius_atom: 0.3,
          polarization: 'TE',
          n_bands: 8,
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      return r ? labelsLine(r) : null;
    },
  },
  {
    slug: 'honeycomb-toml',
    title: 'Honeycomb (two-atom) basis',
    description:
      'A hexagonal lattice with two air holes per unit cell, described as a multi-atom TOML config.',
    primer:
      'A honeycomb crystal is a hexagonal lattice decorated with two basis atoms, sitting at the $(1/3, 1/3)$ and $(2/3, 2/3)$ fractional positions. This is the photonic analogue of graphene and famously hosts a **Dirac point** at $K$. Multi-atom bases are *not* expressible through `solve()`; they require a `[[geometry.atoms]]` block per inclusion in a TOML config.',
    category: 'Lattices & Geometry',
    pyFile: 'honeycomb.py',
    tomlFile: 'honeycomb.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# A honeycomb is a hexagonal lattice with TWO inclusions per cell.
# Multi-atom bases go beyond solve(); describe them in TOML.
driver = BulkDriver("honeycomb.toml")

for result in driver.run_streaming():
    print(result["num_bands"], "bands on", result["num_k_points"], "k-points")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'hexagonal',
          epsilon_background: 13.0,
          polarization: 'TE',
          n_bands: 8,
          atoms: [
            { pos: [1 / 3, 1 / 3], radius: 0.18, eps_inside: 1.0 },
            { pos: [2 / 3, 2 / 3], radius: 0.18, eps_inside: 1.0 },
          ],
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      if (!r) return null;
      return `${r.n_bands} bands on ${r.n_kpoints} k-points`;
    },
  },
  {
    slug: 'custom-k-path',
    title: 'Choose a specific k-path',
    description:
      'Override the lattice preset and trace your own path through the Brillouin zone.',
    primer:
      'k-path coordinates are given in the fractional reciprocal basis $(b_1, b_2)$, not Cartesian, so they stay *invariant* under lattice scaling. Custom paths are useful for zooming in on a band crossing or for stitching together unusual cuts through the Brillouin zone. The points you pass are the path corners; intermediate samples are filled in by **linear interpolation**.',
    category: 'Lattices & Geometry',
    pyFile: 'k_path.py',
    code: `import blaze

# A custom k-path in fractional reciprocal coordinates.
# Here: Γ → X → M → Γ, sampled densely.
k_path = [
    [0.0, 0.0], [0.25, 0.0], [0.5, 0.0],
    [0.5, 0.25], [0.5, 0.5],
    [0.25, 0.25], [0.0, 0.0],
]

result = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=0.2,
    polarization="TM",
    k_path=k_path,
)

print(result.k_points.shape)   # (7, 2)`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
          k_path: [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.5, 0.25],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.0, 0.0],
          ],
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      return r ? `(${r.n_kpoints}, 2)` : null;
    },
  },
  {
    slug: 'two-atom-basis',
    title: 'Two-atom basis (square)',
    description:
      'Build a square crystal with two rods per unit cell using a raw TOML configuration.',
    primer:
      'Multiple atoms per unit cell turn a simple lattice into a richer crystal: a square lattice with two identical rods at $(0.25, 0.25)$ and $(0.75, 0.75)$ effectively *doubles* the band count and folds the bands of the underlying simple square. This is also the entry point for **sublattice symmetry breaking**, where the two atoms differ in radius or permittivity.',
    category: 'Lattices & Geometry',
    pyFile: 'two_atom.py',
    tomlFile: 'two_atom.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# Two rods per cell, one at (0.25, 0.25), one at (0.75, 0.75).
driver = BulkDriver("two_atom.toml")

for result in driver.run_streaming():
    print("atoms:", len(result["params"]["atoms"]))
    print(result["num_bands"], "bands ×", result["num_k_points"], "k-points")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          polarization: 'TM',
          n_bands: 8,
          atoms: [
            { pos: [0.25, 0.25], radius: 0.15, eps_inside: 8.9 },
            { pos: [0.75, 0.75], radius: 0.15, eps_inside: 8.9 },
          ],
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      if (!r) return null;
      return `atoms: 2\n${r.n_bands} bands × ${r.n_kpoints} k-points`;
    },
  },

  // ---- Parameter Sweeps ----------------------------------------------------
  {
    slug: 'radius-sweep',
    title: 'Sweep over rod radius',
    description:
      'Pass a [start, stop, step] list to sweep a parameter and get one result per value.',
    primer:
      'A `[start, stop, step]` triple expands the call into a sweep and returns a list of `BandResult` objects, one per value (the endpoint is *included*). The radius is constrained to $(0, 0.5)$ in units of $a$ so neighbouring rods cannot overlap. Watch how the lower bands stiffen as the rods grow: the filling fraction climbs and pulls the photonic gap up in frequency.',
    category: 'Parameter Sweeps',
    pyFile: 'radius_sweep.py',
    code: `import blaze

# Sweep the rod radius from 0.2 to 0.4 in steps of 0.05.
results = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=[0.2, 0.4, 0.05],
    polarization="TM",
    n_bands=8,
)

for r in results:
    print(f"r={r.radius_atom}: band1 max = {r.freqs[:, 0].max():.4f}")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: [0.2, 0.4, 0.05],
          polarization: 'TM',
          n_bands: 8,
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const valid = results.filter(Boolean);
      return valid.length === 0
        ? null
        : valid
            .map((r) => `r=${r.radius_atom}: band1 max = ${bandMax(r, 0).toFixed(4)}`)
            .join('\n');
    },
  },
  {
    slug: 'resolution-convergence',
    title: 'Resolution convergence study',
    description:
      'Sweep the grid resolution to watch the band frequencies converge.',
    primer:
      'Frequencies converge *from above* as the real-space grid is refined: at low resolution, the dielectric step is poorly resolved and shifts band edges upward by a few percent. A convergence sweep is the cheapest way to pick a working resolution: keep doubling until the band edges stop moving within your target tolerance.',
    category: 'Parameter Sweeps',
    pyFile: 'convergence.py',
    code: `import blaze

# Increase the grid resolution and watch the bands converge.
results = blaze.solve(
    lattice_type="square",
    resolution=[16, 48, 16],
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=0.2,
    polarization="TM",
)

for r in results:
    print(f"res={r.resolution}: band1 max = {r.freqs[:, 0].max():.5f}")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          resolution: [16, 48, 16],
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const valid = results.filter(Boolean);
      return valid.length === 0
        ? null
        : valid
            .map((r) => `res=${r.resolution}: band1 max = ${bandMax(r, 0).toFixed(5)}`)
            .join('\n');
    },
  },
  {
    slug: 'nested-sweeps',
    title: 'Nested sweeps: who is on top?',
    description:
      'Sweep epsilon AND radius at once. See how the flat results list is ordered, which sweep is the outer loop and which varies fastest.',
    primer:
      'Sweeps are expanded into a single *flat* list over the full Cartesian product, not nested lists. The sweep listed **first** in your config is the outer loop and changes slowest; the sweep listed **last** is the inner loop and varies fastest as you walk the results. Every result also carries its `sweep_values` dict, so you can always recover the $(\\varepsilon, r)$ coordinate that produced it without relying on index math.',
    category: 'Parameter Sweeps',
    pyFile: 'nested_sweep.py',
    tomlFile: 'sweep.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# Two [[sweeps]] are declared in sweep.toml. The driver expands them into a
# FLAT list over the full grid. The sweep listed LAST in the file is the
# INNER loop, it varies fastest as you walk the results list.
driver = BulkDriver("sweep.toml")
results = list(driver.run_streaming())

print("sweep order:", results[0]["sweep_order"])
for i, r in enumerate(results):
    # Each result also carries the exact sweep coordinate that produced it.
    print(f"[{i}] eps_atom={r['sweep_values']['atom0.eps_inside']}"
          f"  radius={r['sweep_values']['atom0.radius']}")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: [6.0, 10.0, 2.0],
          radius_atom: [0.2, 0.3, 0.05],
          polarization: 'TM',
          n_bands: 6,
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const valid = results.filter(Boolean);
      if (valid.length === 0) return null;
      const lines = valid.map(
        (r, i) => `[${i}] eps_atom=${r.epsilon_atoms}  radius=${r.radius_atom}`,
      );
      return cappedLines(['sweep order: atom0.eps_inside × atom0.radius', ...lines], 20);
    },
  },
  {
    slug: 'two-dimensional-sweep',
    title: '2D sweep: radius x epsilon',
    description:
      'Sweep two parameters at once, Blaze expands the full grid of configurations.',
    primer:
      'A 2D sweep multiplies the cost: $3$ radii times $3$ permittivities is $9$ full band calculations. The result is a flat list ordered the same way nested sweeps are (*last-declared parameter varies fastest*), which is the natural layout for reshaping into an $(n_r, n_\\varepsilon)$ grid for contour plots of band-edge frequencies or bandgap widths.',
    category: 'Parameter Sweeps',
    pyFile: 'grid_sweep.py',
    code: `import blaze

# Sweep both radius and atom permittivity → a grid of jobs.
results = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=[6.0, 10.0, 2.0],
    radius_atom=[0.2, 0.3, 0.05],
    polarization="TM",
    n_bands=6,
)

print(f"{len(results)} configurations computed")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: [6.0, 10.0, 2.0],
          radius_atom: [0.2, 0.3, 0.05],
          polarization: 'TM',
          n_bands: 6,
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const n = results.filter(Boolean).length;
      return n === 0 ? null : `${n} configurations computed`;
    },
  },

  // ---- Performance ---------------------------------------------------------
  {
    slug: 'set-number-of-workers',
    title: 'Set the number of workers',
    description:
      'Control parallelism across sweep jobs with the threads argument (single-core in the browser).',
    primer:
      'Blaze parallelises across sweep **JOBS**, not across the linear algebra within a single job. With `threads=N` and a sweep of $M$ jobs, up to $N$ jobs run concurrently on the desktop. The browser build is *single-threaded* by design (one WASM core), so jobs run sequentially here, but the same Python code scales straight to your laptop or a workstation without changes.',
    category: 'Performance',
    pyFile: 'workers.py',
    singleCore: true,
    code: `import blaze

# On the desktop, threads=N distributes sweep jobs across N workers.
# In the browser WASM build there is a single core, so the thread
# setting is ignored for live compute, but the API is identical.
results = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=[0.2, 0.35, 0.05],
    polarization="TM",
    n_bands=8,
    threads=4,
)

print(f"{len(results)} jobs done")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: [0.2, 0.35, 0.05],
          polarization: 'TM',
          n_bands: 8,
          threads: 4,
        },
        { displayKind: 'list', resultLabel: 'list[BandResult]', resultVar: 'results' },
      ),
    output: ({ results }) => {
      const n = results.filter(Boolean).length;
      return n === 0 ? null : `${n} jobs done`;
    },
  },

  // ---- Streaming & Inspection ---------------------------------------------
  {
    slug: 'live-k-point-streaming',
    title: 'Stream k-points live',
    description:
      'Watch each k-point being solved as the band diagram fills in, in real time.',
    primer:
      'In streaming mode the engine emits a callback *the moment* a k-point is solved, so you can show progress (or feed a long sweep into a database) without waiting for the whole job to finish. The Python iterator yields one finalized dict per job, but the WASM build additionally surfaces **per-k-point updates** to the host, which is what drives the progressive plot you see here.',
    category: 'Streaming & Inspection',
    pyFile: 'streaming.py',
    tomlFile: 'crystal.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# run_streaming() yields one dict per job. Each dict carries job_index,
# num_k_points, and the full band data. In the browser, the engine also
# pushes per-k-point updates to the host, which drives the live plot.
driver = BulkDriver("crystal.toml")
for result in driver.run_streaming():
    print("job", result["job_index"], "->", result["num_k_points"], "k-points")`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
          n_bands: 8,
        },
        { displayKind: 'streamDict', resultLabel: 'Iterator[dict]', resultVar: 'kpoint' },
      ),
    output: ({ results, rawStream }) => {
      // Python prints one line per finalized job, not per k-point.
      const completed = results.filter(Boolean);
      if (completed.length > 0) {
        return completed
          .map((r, i) => `job ${i} -> ${r.n_kpoints} k-points`)
          .join('\n');
      }
      // While the first job is still running, surface the progress here too.
      if (rawStream.length === 0) return null;
      const last = rawStream[rawStream.length - 1];
      const k = (last.k_index as number) ?? 0;
      const total = (last.total_k_points as number) ?? 0;
      return `job 0 in progress (${k + 1}/${total || '?'} k-points)`;
    },
  },
  {
    slug: 'selective-k-points-bands',
    title: 'Select specific k-points & bands',
    description:
      'Filter the output to just the k-points and bands you care about for a leaner result.',
    primer:
      'Filtered streaming returns *only* the indices you ask for, which is essential when you are scanning thousands of configurations and only need (say) the lowest $4$ bands at the high-symmetry corners. The engine still solves the full eigenproblem, but transfer/storage cost drops **linearly** with the number of selected k-points and bands.',
    category: 'Streaming & Inspection',
    pyFile: 'selective.py',
    tomlFile: 'crystal.toml',
    showToml: true,
    code: `from blaze import BulkDriver

# Stream only k-points 0, 5, 10 and bands 0 to 3.
driver = BulkDriver("crystal.toml")
for result in driver.run_streaming_filtered(
    k_indices=[0, 5, 10],
    band_indices=[0, 1, 2, 3],
):
    print("k-points:", result["num_k_points"], "bands:", result["num_bands"])`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
          n_bands: 8,
        },
        {
          displayKind: 'streamDict',
          resultLabel: 'Iterator[dict]',
          resultVar: 'result',
          mode: 'filtered',
          kIndices: [0, 5, 10],
          bandIndices: [0, 1, 2, 3],
        },
      ),
    output: ({ rawStream }) => {
      if (rawStream.length === 0) return null;
      return rawStream
        .map((d) => {
          const nk = (d.num_k_points as number) ?? (d.k_path as unknown[])?.length ?? 0;
          const nb = (d.num_bands as number) ?? (d.bands as number[][])?.[0]?.length ?? 0;
          return `k-points: ${nk} bands: ${nb}`;
        })
        .join('\n');
    },
  },
  {
    slug: 'inspect-band-result',
    title: 'Inspect the full BandResult',
    description:
      'Explore every attribute of a BandResult object, arrays, shapes, and metadata.',
    primer:
      'A `BandResult` is the canonical output: `freqs` ($n_k \\times n_\\text{bands}$), `distances` (cumulative Euclidean along the path, used as the x-axis), `k_points` and `k_labels` for the path, plus the parameters that produced it. The expandable inspector on the right mirrors what you would see in a Jupyter notebook, so you can dig into any field *without leaving the page*.',
    category: 'Streaming & Inspection',
    pyFile: 'inspect.py',
    code: `import blaze

result = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=0.2,
    polarization="TM",
    n_bands=8,
)

# Every attribute is inspectable:
print(result.freqs)              # ndarray (n_k, n_bands)
print(result.distances)          # ndarray (n_k,)
print(result.k_points)           # ndarray (n_k, 2)
print(result.k_labels)           # ['Γ', 'X', 'M', 'Γ']
print(result.epsilon_atoms)      # 8.9`,
    prepare: () =>
      fromSolve(
        {
          lattice_type: 'square',
          epsilon_background: 1.0,
          epsilon_atoms: 8.9,
          radius_atom: 0.2,
          polarization: 'TM',
          n_bands: 8,
        },
        { displayKind: 'BandResult', resultLabel: 'BandResult', resultVar: 'result' },
      ),
    output: ({ results, live }) => {
      const r = results[0] ?? live;
      if (!r) return null;
      return [
        `freqs:         ndarray (${r.n_kpoints}, ${r.n_bands})`,
        `distances:     ndarray (${r.n_kpoints},)`,
        `k_points:      ndarray (${r.n_kpoints}, 2)`,
        `k_labels:      ${labelsLine(r)}`,
        `epsilon_atoms: ${r.epsilon_atoms}`,
      ].join('\n');
    },
  },
];

export function getExample(slug: string): Example | undefined {
  return EXAMPLES.find((e) => e.slug === slug);
}

export function getExampleSlugs(): string[] {
  return EXAMPLES.map((e) => e.slug);
}
