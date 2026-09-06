/**
 * Builds the "Run" playback: a solve propagating through the architecture.
 *
 * Calibration honesty:
 * - Per-k durations and iteration counts are REAL, from
 *   public/data/benchmarks/series4-iterations.json (32x32 grid, 8 bands,
 *   square lattice, TM). Wall time is scaled uniformly so the playback is
 *   watchable; proportions between stages are preserved.
 * - The split INSIDE one LOBPCG iteration is MODELED from operation counts
 *   (FFT counts, GEMM bytes, r^3 dense solve). The UI labels it as modeled.
 */

import type { FlatStage, RunStage, RunTimeline, SolveParams } from './types';

export interface PerKSample {
  kIndex: number;
  iterations: number;
  ms: number;
}

export interface BenchData {
  /** Real per-k samples (ordered by k index). */
  perK: PerKSample[];
  /** Where the numbers came from, shown in the caption bar. */
  source: string;
}

/** Fallback if the benchmark fetch fails: plausible warm-start decay. */
export function syntheticBench(kPoints: number): BenchData {
  const perK: PerKSample[] = [];
  for (let k = 0; k < kPoints; k++) {
    const iterations = k === 0 ? 12 : Math.max(3, Math.round(7 - Math.log2(1 + k)));
    perK.push({ kIndex: k, iterations, ms: iterations * 0.9 });
  }
  return { perK, source: 'modeled (benchmark data unavailable)' };
}

/** Parses public/data/benchmarks/series4-iterations.json. */
export function benchFromSeries4(json: unknown): BenchData | null {
  try {
    const root = json as {
      TM?: { blaze?: { k_points?: { k_index: number; iterations: number; elapsed_seconds: number }[] } };
    };
    const kPoints = root.TM?.blaze?.k_points;
    if (!kPoints || kPoints.length === 0) return null;
    return {
      perK: kPoints.map((k) => ({ kIndex: k.k_index, iterations: k.iterations, ms: k.elapsed_seconds * 1000 })),
      source: 'measured per-k wall times (series4, 32×32, 8 bands, TM)',
    };
  } catch {
    return null;
  }
}

/**
 * Modeled cost split of one LOBPCG iteration (fractions of iteration time).
 * Derived from operation counts for TM at 32x32 / 8 bands: preconditioner and
 * batched A·Q are FFT-bound; SVQB/GEMM are N·r^2; the dense solve is r^3.
 */
const ITER_SPLIT: [string, number, string[], string[]][] = [
  // [stage suffix, fraction, node ids, edge ids]
  ['resid', 0.1, ['lob.resid', 'lob.lazy'], ['e.block-resid', 'e.lazy-resid']],
  ['deflate', 0.04, ['lob.deflate'], ['e.resid-deflate']],
  [
    'precond',
    0.18,
    ['lob.precond', 'pc.fourier', 'cpu.fft'],
    ['e.deflate-precond', 'e.precond-pc', 'e.pcf-fft'],
  ],
  ['subspace', 0.05, ['lob.subspace', 'lob.w'], ['e.precond-subspace', 'e.w-subspace']],
  ['svqb', 0.14, ['lob.svqb', 'cpu.dot'], ['e.subspace-svqb', 'e.svqb-dot']],
  [
    'aq',
    0.28,
    ['lob.aq', 'theta.tm', 'cpu.fft', 'theta.scratch'],
    ['e.svqb-aq', 'e.aq-tm', 'e.tm-fft', 'e.fft-plans', 'e.fft-transpose'],
  ],
  ['upcast', 0.05, ['lob.upcast'], ['e.aq-upcast', 'e.upcast-gemm']],
  ['gemm', 0.09, ['lob.gemm'], []],
  ['eigen', 0.03, ['lob.eigen'], ['e.gemm-eigen']],
  ['downcast', 0.02, ['lob.downcast'], ['e.eigen-downcast', 'e.downcast-ritz']],
  ['ritz', 0.02, ['lob.ritz', 'lob.block'], ['e.ritz-block', 'e.ritz-w', 'e.eigen-evals']],
];

const LOBPCG_ALL_NODES = [
  'grp.lobpcg',
  'lob.block',
  'lob.resid',
  'lob.deflate',
  'lob.precond',
  'lob.subspace',
  'lob.svqb',
  'lob.aq',
  'lob.upcast',
  'lob.gemm',
  'lob.eigen',
  'lob.downcast',
  'lob.ritz',
  'lob.w',
  'theta.tm',
  'cpu.fft',
];

const LOBPCG_RING_EDGES = [
  'e.block-resid',
  'e.resid-deflate',
  'e.deflate-precond',
  'e.precond-subspace',
  'e.w-subspace',
  'e.subspace-svqb',
  'e.svqb-aq',
  'e.aq-upcast',
  'e.upcast-gemm',
  'e.gemm-eigen',
  'e.eigen-downcast',
  'e.downcast-ritz',
  'e.ritz-block',
  'e.ritz-w',
];

/** Uniform wall-time -> playback scale, chosen so total playback ~ TARGET_MS. */
const TARGET_MS = 70_000;

export function buildTimeline(params: SolveParams, bench: BenchData): RunTimeline {
  const stages: RunStage[] = [];

  const totalSolveMs = bench.perK.reduce((sum, k) => sum + k.ms, 0);
  // Fixed-length setup stages take ~14s of the playback; the k-sweep gets the rest.
  const sweepBudget = TARGET_MS - 14_000;
  const scale = sweepBudget / Math.max(totalSolveMs, 1);

  /* -------- setup -------- */
  stages.push({
    id: 'parse',
    label: 'Parse config',
    caption: 'One TOML file in: geometry, grid, k-path, sweeps, solver settings become a BulkConfig.',
    duration: 1_600,
    nodes: ['if.toml', 'bulk.parse'],
    edges: ['e.toml-parse'],
  });
  stages.push({
    id: 'expand',
    label: 'Expand sweeps',
    caption: `Sweep dimensions multiply out into ${params.jobs} independent jobs.`,
    duration: 1_400,
    nodes: ['bulk.expand'],
    edges: ['e.parse-expand'],
  });
  stages.push({
    id: 'fanout',
    label: 'Fan out jobs',
    caption: 'rayon runs jobs data-parallel on a dedicated pool. This is the ONLY parallelism level; we follow one job.',
    duration: 2_200,
    nodes: ['bulk.pool', 'bulk.lane1', 'bulk.lane2', 'bulk.lane3', 'bulk.lane4', 'bulk.adaptive'],
    edges: ['e.expand-pool', 'e.pool-lane1', 'e.pool-lane2', 'e.pool-lane3', 'e.pool-lane4', 'e.adaptive-pool'],
  });
  stages.push({
    id: 'geometry',
    label: 'Build geometry',
    caption: 'Lattice, basis atoms, and the Brillouin-zone path for this job.',
    duration: 1_800,
    nodes: ['geo.lattice', 'geo.crystal', 'geo.brillouin', 'drv.kloop', 'bulk.lane1'],
    edges: ['e.lane-driver', 'e.parse-lattice', 'e.lattice-crystal', 'e.lattice-bz', 'e.bz-kloop'],
  });
  stages.push({
    id: 'dielectric',
    label: 'Build dielectric (once per job)',
    caption: 'Analytic subpixel smoothing: exact interface geometry gives MPB-grade ε on a coarse grid. O(N·atoms), done once.',
    duration: 2_400,
    nodes: ['diel.smooth', 'diel.eps', 'geo.crystal'],
    edges: ['e.crystal-smooth', 'e.smooth-eps'],
  });

  /* -------- k-point sweep -------- */
  const perK = bench.perK;
  const kCount = perK.length;

  const kSetupNodes = ['drv.kloop', 'drv.epsclone', 'theta.ktables', 'grp.theta'];
  const kSetupEdges = ['e.eps-clone', 'e.clone-theta', 'e.kloop-block'];

  const first = perK[0];
  const firstMs = Math.max(first.ms * scale, 9_000);

  // k0: operator setup
  stages.push({
    id: 'k0.setup',
    label: 'k₀: operator setup',
    caption: 'Per k-point: clone ε, rebuild the k+G tables. The operator is never a matrix, just these tables plus FFTs.',
    duration: 1_600,
    nodes: kSetupNodes,
    edges: kSetupEdges,
  });

  // k0: two expanded LOBPCG iterations (modeled split), then the rest compressed.
  const expandedIters = Math.min(2, first.iterations);
  const iterMs = (firstMs * 0.55) / expandedIters;
  for (let iter = 0; iter < expandedIters; iter++) {
    for (const [suffix, fraction, nodes, edges] of ITER_SPLIT) {
      stages.push({
        id: `k0.it${iter}.${suffix}`,
        label: `k₀ · iteration ${iter + 1}: ${suffix}`,
        caption: iterationCaption(suffix),
        duration: Math.max(iterMs * fraction, 220),
        nodes,
        edges,
      });
    }
  }
  const restIters = first.iterations - expandedIters;
  if (restIters > 0) {
    stages.push({
      id: 'k0.rest',
      label: `k₀: iterations 3..${first.iterations}`,
      caption: `${restIters} more iterations to convergence (${first.iterations} total at k₀, measured). Intra-iteration split above is modeled from op counts.`,
      duration: firstMs * 0.45,
      nodes: LOBPCG_ALL_NODES,
      edges: LOBPCG_RING_EDGES,
    });
  }
  stages.push({
    id: 'k0.out',
    label: 'k₀: eigenpairs out',
    caption: 'Eigenvalues (always f64) leave the solver; converged X seeds the next k-point.',
    duration: 1_200,
    nodes: ['lob.evals', 'drv.track', 'lob.block', 'drv.warm'],
    edges: ['e.eigen-evals', 'e.evals-track', 'e.block-warm'],
  });

  // k1..k5 individually (real durations), warm-started.
  const detailed = Math.min(5, kCount - 1);
  for (let k = 1; k <= detailed; k++) {
    const sample = perK[k];
    stages.push({
      id: `k${k}`,
      label: `k${subscript(k)}: warm-started solve`,
      caption: `Warm start + subspace transport cut this k-point to ${sample.iterations} iterations (measured: ${sample.ms.toFixed(1)} ms).`,
      duration: Math.max(sample.ms * scale, 900),
      nodes: [...LOBPCG_ALL_NODES, 'drv.warm', 'drv.predict', 'drv.kloop'],
      edges: ['e.warm-block', 'e.predict-warm', ...LOBPCG_RING_EDGES, 'e.block-warm'],
    });
  }

  // Remaining k-points as one compressed sweep stage.
  if (kCount - 1 > detailed) {
    const rest = perK.slice(detailed + 1);
    const restMs = rest.reduce((sum, k) => sum + k.ms, 0);
    const restIterTotal = rest.reduce((sum, k) => sum + k.iterations, 0);
    stages.push({
      id: 'ksweep',
      label: `k${subscript(detailed + 1)}..k${subscript(kCount - 1)}: the sweep`,
      caption: `${rest.length} more k-points, ${restIterTotal} iterations total. Durations here are the real measured per-k wall times, compressed.`,
      duration: restMs * scale,
      nodes: [...LOBPCG_ALL_NODES, 'drv.warm', 'drv.predict', 'drv.kloop', 'drv.epsclone'],
      edges: ['e.warm-block', 'e.predict-warm', 'e.eps-clone', ...LOBPCG_RING_EDGES, 'e.eigen-evals', 'e.evals-track', 'e.block-warm'],
    });
  }

  /* -------- results out -------- */
  stages.push({
    id: 'track',
    label: 'Band tracking',
    caption: 'Bands are matched across k by overlap (Hungarian assignment), so indices stay physical through crossings.',
    duration: 1_600,
    nodes: ['drv.track', 'drv.result'],
    edges: ['e.evals-track', 'e.track-result'],
  });
  stages.push({
    id: 'out',
    label: 'Stream results',
    caption: 'One mutex-guarded writer streams CSV; Python gets rows over a crossbeam channel; the browser gets per-k callbacks.',
    duration: 2_600,
    nodes: ['bulk.writer', 'if.csv', 'if.diag', 'if.py.channel', 'if.py.iter', 'if.wasm.cb', 'drv.result'],
    edges: ['e.result-writer', 'e.writer-csv', 'e.writer-diag', 'e.result-channel', 'e.channel-iter', 'e.result-wasmcb'],
  });

  /* -------- flatten -------- */
  const flat: FlatStage[] = [];
  let cursor = 0;
  for (const stage of stages) {
    flat.push({ start: cursor, end: cursor + stage.duration, stage });
    cursor += stage.duration;
  }
  return { params, totalMs: cursor, flat };
}

function iterationCaption(suffix: string): string {
  switch (suffix) {
    case 'resid':
      return 'Residuals from cached A·X and B·X: zero fresh operator applies. Norms are skipped ~90% of the time.';
    case 'deflate':
      return 'Converged bands are projected out and soft-locked.';
    case 'precond':
      return 'P = M⁻¹R: 2 FFTs per band (TM). The preconditioner reuses the same FFT engine.';
    case 'subspace':
      return 'The trial subspace Z = [X, P, W]: current block, preconditioned residuals, history.';
    case 'svqb':
      return 'SVQB B-orthonormalization: Gram matrix + dense eigen, rank-dropping near-dependent directions.';
    case 'aq':
      return 'Batched A·Q: one operator apply per subspace vector. The FFT-heavy step of the iteration.';
    case 'upcast':
      return 'THE precision boundary: storage (f32) is materialized as f64 copies before any GEMM. N×r×16 bytes of pure data movement.';
    case 'gemm':
      return 'Aₛ = Qᴴ(AQ) in f64. faer runs sequentially by design: at this size, parallel GEMM is 5-10× slower.';
    case 'eigen':
      return 'The Rayleigh-Ritz step: an r×r Hermitian eigensolve. Tiny, dense, always f64.';
    case 'downcast':
      return 'Ritz coefficients drop back to storage precision (cscalar). The f64 detour is over.';
    case 'ritz':
      return 'X, B·X, A·X rebuilt by GEMM from A·Q: reuse instead of recompute.';
    default:
      return '';
  }
}

const SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉';
function subscript(n: number): string {
  return String(n)
    .split('')
    .map((d) => SUBSCRIPTS[Number(d)])
    .join('');
}
