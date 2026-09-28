/* Generated from blaze2d-interface. Run pnpm generate:contract to update. */

export type LatticeKind = "square" | "triangular" | "rectangular" | "oblique" | "custom";
export type ObjectKind = "circle";
export type KCoordinates = "reciprocal_fractional" | "cartesian_angular";
export type Quantity =
  "velocity" | "mass_tensor" | "r_derivatives" | "born_huang" | "slow_coefficient" | "exact_tm" | "overlap";
export type Reference = {
  source: "external";
};
export type Task = "bands" | "operators";
export type Event =
  | {
      event: "run_start";
      jobs: number;
      solves: number;
    }
  | {
      event: "job_start";
      job: PlannedJob;
      job_index: number;
    }
  | {
      band_point?: BandPoint | null;
      completed: number;
      converged: boolean;
      event: "progress";
      iterations: number;
      job_index: number;
      sample_index: number;
      total: number;
    }
  | {
      event: "result";
      result: ResultRecord;
    }
  | {
      error: JobFailure;
      event: "job_failure";
    }
  | {
      completed: number;
      event: "terminal";
      failed: number;
      status: RunStatus;
    };
export type DType = "float64" | "complex128";
export type RunStatus = "completed" | "completed_with_errors" | "failed" | "cancelled";
export type Platform = "native" | "browser";
export type Precision = "f64" | "f32";

export interface BrowserContract {
  config: Config;
  event: Event;
  job: PlannedJob;
  result: ResultRecord;
  validation: ValidationReport;
}
export interface Config {
  bands?: Bands | null;
  dielectric?: Dielectric;
  dimension?: number;
  eigensolver?: Eigensolver;
  geometry: Geometry;
  grid?: Grid;
  operators?: Operators | null;
  polarization?: "TM" | "TE";
  results?: Results;
  schema: string;
  sweeps?: Sweep[];
  task: Task;
}
export interface Bands {
  /**
   * Optional acceptance gate on the final physical eigenpairs.
   */
  accuracy?: BandAccuracy | null;
  count?: number;
  /**
   * Selecting a metric ball changes the finite numerical pencil.
   */
  discretization?:
    | {
        kind: "fft_rectangle";
      }
    | {
        /**
         * Cartesian angular cutoff in inverse reference length; no implicit 2*pi.
         */
        angular_cutoff: number;
        kind: "metric_ball";
        max_allocation_rounds?: number;
        max_coefficients?: number;
        max_dense_dimension?: number;
        max_embedding_points?: number;
        max_enumerated_indices?: number;
        /**
         * Numerical payload admission, not a process RSS limit.
         */
        max_estimated_payload_bytes?: number;
      };
  /**
   * Optional setup probes; ordinary solves do not compute condition estimates.
   */
  operator_diagnostics?: boolean;
  path?: KPath;
  symmetry?: Symmetry;
  tracking?: boolean;
  /**
   * Native parameter continuation. Omitted by default.
   */
  warm_start?: GeometryWarmStart | null;
}
/**
 * Acceptance is separate from eigenvalue-change stopping. Every mode must
 * meet either residual bound, and the whole block must be B-orthonormal.
 * The absolute branch remains meaningful for zero and near-zero Gamma modes.
 */
export interface BandAccuracy {
  /**
   * In inverse reference-length squared units, not normalized frequency.
   */
  max_absolute_residual?: number;
  max_orthogonality_defect?: number;
  max_relative_residual?: number;
}
export interface KPath {
  basis?: "reciprocal_fractional" | "cartesian_angular";
  intervals_per_segment?: number | null;
  labels?: string[];
  points?: number[][] | null;
  preset?: string | null;
  vertices?: number[][] | null;
}
export interface Symmetry {
  arithmetic?: "spatial_complex" | "fourier_complex" | "pt_real";
  coarse_start?: CoarseStart | null;
  continuation?: "off" | "first_point" | "previous_geometry";
  correction_projection_interval?: number;
  k_reuse?: "off" | "identical" | "unitary" | "time_reversal";
  k_traversal?: "given" | "nearest";
  max_cache_bytes?: number;
  /**
   * Total resident continuation payload across native worker chains.
   * Solver scratch and retained output use separate memory estimates.
   */
  max_continuation_bytes?: number;
  /**
   * Numerical payload admission for rectangular sector plans and solves.
   * Metric Fourier supports use bands.discretization's budget instead.
   */
  max_sector_workspace_bytes?: number;
  max_traversal_field_bytes?: number;
  plane_group?: string | null;
  policy?: "manual" | "auto";
  reconstruct_partners?: boolean;
  sector_guard_budget?: number | null;
  sector_kernel?: "full_grid" | "mirror_tm" | "mirror_tm_te";
  sector_storage?: "full_grid" | "orbit";
  sectors?: "off" | "conservative_mirror" | "adaptive_mirror" | "conservative_group" | "adaptive_group";
  seed_transfer?: "copy" | "rank_aware";
  share_sector_workspace?: boolean;
}
export interface CoarseStart {
  /**
   * Coarse, lifted and prepared seed fields; solver and FFT scratch are extra.
   */
  max_seed_field_bytes?: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution: [number, number];
}
/**
 * Bounded native history on a fixed physical grid and lattice.
 */
export interface GeometryWarmStart {
  anchors?: [number, number][];
  donor_selection?: "same_k" | "previous_k" | "screened";
  guards?: number;
  history?: number;
  innovation_columns?: number;
  innovation_relative_tolerance?: number;
  max_cache_bytes?: number;
  max_guards?: number | null;
  /**
   * Zero selects twice the retained positive window.
   */
  max_predictor_columns?: number;
  max_preparation_bytes?: number;
  method?: "cold" | "copy" | "metric" | "secant" | "history" | "residual" | "response" | "tangent" | "reduced_response";
  model?: GeometryResponseModel;
  parameter_order?: "grouped" | "serpentine";
  response_iterations?: number;
  response_tolerance?: number;
  retain_tm_actions?: boolean;
  scope?: "gamma" | "anchors" | "path";
  transport?: GeometryTransport | null;
}
/**
 * Reusable source response limits. These do not weaken final field accuracy.
 */
export interface GeometryResponseModel {
  basis?: "fourier" | "history" | "residual";
  columns?: number;
  max_relative_residual?: number;
  max_reuses?: number;
  max_step?: number;
  training_iterations?: number;
  training_tolerance?: number;
}
/**
 * Optional periodic shape transport, composed as an independent target trial.
 */
export interface GeometryTransport {
  interpolation?: "cubic" | "fourier";
  jacobian?: boolean;
  max_interpolation_terms?: number;
  max_stretch?: number;
}
export interface Dielectric {
  interface_tolerance?: number;
  mesh_size?: number;
  smoothing?: "analytic" | "subgrid" | "voronoi" | "none";
  source?: "geometry" | "external";
}
export interface Eigensolver {
  block_size?: number;
  max_iterations?: number | null;
  precision?: "f64" | "f32";
  tolerance?: number | null;
}
export interface Geometry {
  background_epsilon?: number;
  lattice: Lattice;
  objects?: Object[];
}
export interface Lattice {
  a?: number | null;
  angle_deg?: number | null;
  b?: number | null;
  type: LatticeKind;
  vectors?: number[][] | null;
}
export interface Object {
  center?: number[];
  epsilon: number;
  kind: ObjectKind;
  name: string;
  radius: number;
}
export interface Grid {
  resolution?: number | number[];
}
export interface Operators {
  band_lo?: number;
  fail_on_residual?: number | null;
  k_point: KPoint;
  k_stencil?: KStencil | null;
  quantities?: Quantity[];
  reference?: Reference | null;
  registry?: Registry | null;
  remote_bands?: number;
  retained_bands?: number;
}
export interface KPoint {
  basis: KCoordinates;
  value: number[];
}
export interface KStencil {
  half_width: number;
  points_per_axis: number;
}
export interface Registry {
  fd_step?: number;
  object: string;
  points?: number[][];
}
export interface Results {
  eigenvectors?: boolean;
}
export interface Sweep {
  linspace?: Linspace | null;
  name: string;
  target: string;
  values?: unknown[] | null;
}
export interface Linspace {
  count: number;
  start: number;
  stop: number;
}
export interface PlannedJob {
  index: number;
  multi_index: number[];
  /**
   * @minItems 2
   * @maxItems 2
   */
  registry: [number, number];
  registry_index?: number | null;
  resolved: ResolvedConfig;
  sweep: {
    [k: string]: unknown;
  };
}
export interface ResolvedConfig {
  config: Config;
  distances: number[];
  k_label_indices: number[];
  k_labels: string[];
  k_points_cartesian: [number, number][];
  k_points_fractional: [number, number][];
  /**
   * Direct lattice vectors, stored as rows for serialization.
   *
   * @minItems 2
   * @maxItems 2
   */
  lattice_vectors: [[number, number], [number, number]];
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution: [number, number];
  solved_bands: number;
}
export interface BandPoint {
  distance: number;
  frequencies: number[];
  /**
   * @minItems 2
   * @maxItems 2
   */
  k_point: [number, number];
}
export interface ResultRecord {
  arrays: {
    [k: string]: Array;
  };
  job_index: number;
  metadata: unknown;
  samples?: SampleRecord[];
  schema: string;
  task: Task;
}
export interface Array {
  data: number[];
  dimensions: string[];
  dtype: DType;
  order: string;
  shape: number[];
}
export interface SampleRecord {
  arrays: {
    [k: string]: Array;
  };
  metadata: unknown;
  sample_index: number;
}
export interface JobFailure {
  diagnostic: Diagnostic;
  job_index: number;
  partial_result?: ResultRecord | null;
}
export interface Diagnostic {
  code: string;
  message: string;
  path: string;
  /**
   * Half-open UTF-8 byte offsets in the original TOML document.
   *
   * @minItems 2
   * @maxItems 2
   */
  span?: [number, number] | null;
}
export interface ValidationReport {
  capabilities: Capabilities;
  config?: Config | null;
  errors: Diagnostic[];
  ok: boolean;
  resolved?: ResolvedConfig | null;
  summary?: PlanSummary | null;
}
export interface Capabilities {
  checkpoint_resume: boolean;
  dimensions: number[];
  external_inputs: boolean;
  geometries: ObjectKind[];
  k_stencil: boolean;
  platform: Platform;
  precisions: Precision[];
  registry: boolean;
  schema: string;
  tasks: Task[];
  version: string;
}
export interface PlanSummary {
  estimated_peak_bytes: number;
  jobs: number;
  memory_estimate_scope: string;
  precision: Precision;
  registry_points: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution: [number, number];
  solved_bands: number;
  solves: number;
  sweep_shape: number[];
  task: Task;
}
