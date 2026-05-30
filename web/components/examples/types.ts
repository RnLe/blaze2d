export type ExampleRunMode = 'kpoint' | 'collect' | 'collect-filtered' | 'job-stream';

export interface AtomParams {
  index: number;
  pos: [number, number];
  radius: number;
  eps_inside: number;
}

export interface JobParams {
  eps_bg: number;
  resolution: number;
  polarization: 'TE' | 'TM';
  lattice_type?: string;
  atoms: AtomParams[];
}

export type SweepValue = number | string;

export interface MaxwellResult {
  result_type: 'maxwell';
  job_index: number;
  params: JobParams;
  sweep_values: Record<string, SweepValue>;
  sweep_order: string;
  k_path: [number, number][];
  distances: number[];
  bands: number[][];
  num_k_points: number;
  num_bands: number;
}

export interface KPointResult {
  stream_type: 'k_point';
  job_index: number;
  k_index: number;
  total_k_points: number;
  k_point: [number, number];
  distance: number;
  omegas: number[];
  bands: number[];
  iterations: number;
  is_gamma: boolean;
  num_bands: number;
  progress: number;
  params: JobParams;
  sweep_values: Record<string, SweepValue>;
  sweep_order: string;
}

export interface DriverStats {
  total_jobs: number;
  completed: number;
  failed: number;
  total_time_ms: number;
  total_time_secs: number;
  jobs_per_second?: number;
}

export interface DryRunResult {
  total_jobs: number;
  thread_mode: string;
  solver_type: string;
}

export interface ExampleRunRequest {
  id: string;
  toml: string;
  mode: ExampleRunMode;
  kIndices?: number[];
  bandIndices?: number[];
}

export type ExampleWorkerInMessage =
  | { type: 'init'; basePath: string }
  | ({ type: 'start' } & ExampleRunRequest);

export type ExampleWorkerOutMessage =
  | { type: 'ready' }
  | { type: 'metadata'; id: string; dryRun: DryRunResult; jobCount: number }
  | { type: 'kpoint'; id: string; result: KPointResult; progress: number }
  | { type: 'result'; id: string; result: MaxwellResult; progress: number }
  | { type: 'done'; id: string; results: MaxwellResult[]; stats: DriverStats }
  | { type: 'error'; id?: string; message: string };

export interface VariableEntry {
  name: string;
  type: string;
  value: unknown;
}
