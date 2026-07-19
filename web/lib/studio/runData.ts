/**
 * Run-history data model: types, labels, eviction, ref pruning, k-label
 * positions, and CSV/JSON exporters. Pure functions only; the zustand store
 * calls into these so Map-identity and cap rules live in one testable place.
 */

import type { BandResult } from '../examples/bandResult';

export interface JobResult {
  jobIndex: number;
  result: BandResult;
  sweepValues: [string, string | number][];
}

export type RunRecordStatus = 'running' | 'done' | 'error' | 'aborted';

export interface RunRecord {
  id: string;
  /** 1-based session-monotonic counter: "Run 3". */
  index: number;
  startedAt: number;
  label: string;
  /** Full TOML snapshot taken when Run was clicked. */
  configToml: string;
  status: RunRecordStatus;
  totalJobs: number;
  precision: string;
  polarization: string;
  solverType: string;
  /** Finished jobs only, keyed by real job index. */
  jobs: Map<number, JobResult>;
  computeMs: number | null;
  error: string | null;
}

export interface SeriesRef {
  runId: string;
  jobIndex: number;
}

export const seriesKey = (ref: SeriesRef): string => `${ref.runId}:${ref.jobIndex}`;

export function parseSeriesKey(key: string): SeriesRef | null {
  const i = key.lastIndexOf(':');
  if (i <= 0) return null;
  const jobIndex = Number(key.slice(i + 1));
  if (!Number.isFinite(jobIndex)) return null;
  return { runId: key.slice(0, i), jobIndex };
}

export interface PlotSpec {
  id: string;
  name: string;
  series: SeriesRef[];
  /** 1-based inclusive band range; null = all bands. */
  bandRange: [number, number] | null;
  showLegend: boolean;
  showGrid: boolean;
}

/** History caps (user decision: last 5 runs, ~500 jobs total). */
export const MAX_RUNS = 5;
export const MAX_JOBS_GLOBAL = 500;
/** Live overlay rolling window (user decision). */
export const LIVE_WINDOW = 20;

let runCounter = 0;

export function makeRunRecord(meta: {
  configToml: string;
  precision: string;
  polarization: string;
  solverType: string;
}): RunRecord {
  runCounter += 1;
  const startedAt = Date.now();
  const record: RunRecord = {
    id: `r_${startedAt.toString(36)}_${runCounter}`,
    index: runCounter,
    startedAt,
    label: '',
    configToml: meta.configToml,
    status: 'running',
    totalJobs: 0,
    precision: meta.precision,
    polarization: meta.polarization,
    solverType: meta.solverType,
    jobs: new Map(),
    computeMs: null,
    error: null,
  };
  record.label = runLabel(record);
  return record;
}

export function runLabel(run: RunRecord): string {
  const jobs = run.totalJobs > 1 ? ` · ${run.totalJobs} jobs` : '';
  return `Run ${run.index}${jobs} · ${run.polarization} · ${run.precision}`;
}

/** Short label for one job's series: sweep values if present, else job index. */
export function jobLabel(job: JobResult, run?: RunRecord): string {
  if (job.sweepValues.length > 0) {
    return job.sweepValues
      .map(([key, value]) => {
        const short = key.replace(/^atom\d+\./, '');
        const val = typeof value === 'number' ? trimNumber(value) : value;
        return `${short}=${val}`;
      })
      .join(' ');
  }
  if (run && run.totalJobs > 1) return `job ${job.jobIndex}`;
  return run ? `Run ${run.index}` : `job ${job.jobIndex}`;
}

function trimNumber(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toFixed(4)));
}

/**
 * Evict oldest runs until the caps are met. Never evicts `protectRunId`
 * (the live run). Returns a NEW array when anything changed.
 */
export function evictRuns(runs: RunRecord[], protectRunId: string | null): RunRecord[] {
  const overBudget = (list: RunRecord[]) =>
    list.length > MAX_RUNS ||
    list.reduce((sum, r) => sum + r.jobs.size, 0) > MAX_JOBS_GLOBAL;

  let out = runs;
  while (overBudget(out)) {
    const victim = out.findIndex((r) => r.id !== protectRunId);
    if (victim === -1) break;
    out = out.filter((_, i) => i !== victim);
  }
  return out;
}

/** Remove refs pointing at evicted runs/jobs from plot specs + selection. */
export function pruneRefs(
  runs: RunRecord[],
  specs: PlotSpec[],
  selection: Set<string>,
): { specs: PlotSpec[]; selection: Set<string> } {
  const byId = new Map(runs.map((r) => [r.id, r]));
  const alive = (ref: SeriesRef) => byId.get(ref.runId)?.jobs.has(ref.jobIndex) ?? false;

  let specsChanged = false;
  const nextSpecs = specs.map((spec) => {
    const series = spec.series.filter(alive);
    if (series.length !== spec.series.length) {
      specsChanged = true;
      return { ...spec, series };
    }
    return spec;
  });

  let selChanged = false;
  const nextSel = new Set<string>();
  for (const key of selection) {
    const ref = parseSeriesKey(key);
    if (ref && alive(ref)) nextSel.add(key);
    else selChanged = true;
  }

  return {
    specs: specsChanged ? nextSpecs : specs,
    selection: selChanged ? nextSel : selection,
  };
}

/* ------------------------------ k-path labels ----------------------------- */

/**
 * Positions of the high-symmetry labels along the streamed path.
 *
 * Derived from the stream itself (`distances` has one entry per solved
 * k-point) rather than from config, so it stays correct if the driver dedupes
 * joint points or drops the final Γ. Assumes uniform per-leg sampling.
 */
export function computeLabelDistances(distances: number[], labelCount: number): number[] {
  const n = distances.length;
  if (n === 0 || labelCount < 2) return [];
  const ppsEff = (n - 1) / (labelCount - 1);
  return Array.from({ length: labelCount }, (_, m) =>
    distances[Math.min(Math.round(m * ppsEff), n - 1)],
  );
}

/* -------------------------------- exporters ------------------------------- */

export interface ExportSeries {
  runLabel: string;
  jobIndex: number;
  sweepValues: [string, string | number][];
  result: BandResult;
}

export function collectExportSeries(
  runs: RunRecord[],
  refs: SeriesRef[],
): ExportSeries[] {
  const byId = new Map(runs.map((r) => [r.id, r]));
  const out: ExportSeries[] = [];
  for (const ref of refs) {
    const run = byId.get(ref.runId);
    const job = run?.jobs.get(ref.jobIndex);
    if (!run || !job) continue;
    out.push({
      runLabel: run.label,
      jobIndex: job.jobIndex,
      sweepValues: job.sweepValues,
      result: job.result,
    });
  }
  return out;
}

/**
 * CSV of one or more series. Columns: run, job, one column per swept
 * parameter, then k data and bands (padded to the widest series).
 */
export function seriesToCsv(series: ExportSeries[], bandRange?: [number, number] | null): string {
  const sweepKeys: string[] = [];
  for (const s of series) {
    for (const [key] of s.sweepValues) {
      if (!sweepKeys.includes(key)) sweepKeys.push(key);
    }
  }
  const maxBands = Math.max(0, ...series.map((s) => s.result.n_bands));
  const [bLo, bHi] = bandRange ?? [1, maxBands];
  const lo = Math.max(1, bLo);
  const hi = Math.min(maxBands, bHi);

  const header = ['run', 'job', ...sweepKeys, 'k_index', 'kx', 'ky', 'k_distance'];
  for (let b = lo; b <= hi; b++) header.push(`band${b}`);
  const rows: string[] = [header.join(',')];

  for (const s of series) {
    const sweepMap = new Map(s.sweepValues);
    const sweepCells = sweepKeys.map((key) => {
      const v = sweepMap.get(key);
      return v === undefined ? '' : String(v);
    });
    const r = s.result;
    for (let k = 0; k < r.freqs.length; k++) {
      const kp = r.k_points[k] ?? [0, 0];
      const cells: (string | number)[] = [
        csvQuote(s.runLabel),
        s.jobIndex,
        ...sweepCells,
        k,
        kp[0],
        kp[1],
        r.distances[k] ?? 0,
      ];
      for (let b = lo - 1; b < hi; b++) cells.push(r.freqs[k]?.[b] ?? '');
      rows.push(cells.join(','));
    }
  }
  return rows.join('\n');
}

function csvQuote(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function seriesToJsonPayload(series: ExportSeries[], projectName: string): unknown {
  return {
    project: projectName,
    exportedAt: new Date().toISOString(),
    series: series.map((s) => ({
      run: s.runLabel,
      jobIndex: s.jobIndex,
      sweepValues: Object.fromEntries(s.sweepValues),
      result: s.result,
    })),
  };
}

/* --------------------------------- misc ---------------------------------- */

export function timeAgo(ts: number): string {
  const s = Math.max(0, (Date.now() - ts) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

/** Min/max frequency across a result (for the Data tab range column). */
export function freqRange(result: BandResult): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of result.freqs) {
    for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return [0, 0];
  return [lo, hi];
}
