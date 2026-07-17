/**
 * Renderer-independent description of the Blaze2D architecture board.
 *
 * The graph (nodes + edges) lives in model.ts; layout.ts resolves nested
 * local coordinates into world space; timeline.ts builds the "Run"
 * propagation sequence. Nothing in this directory imports pixi.
 */

/** Semantic role — drives the base color (see palette.ts). */
export type NodeKind = 'compute' | 'movement' | 'memory' | 'control' | 'interface';

/** Structural role — drives shape and nesting behavior. */
export type NodeType = 'group' | 'crate' | 'module' | 'kernel' | 'buffer' | 'port';

/** Precision behavior for the precision overlay. `generic` = monomorphized over R ∈ {f32, f64}. */
export type Precision = 'f32' | 'f64' | 'mixed' | 'generic';

export type ParallelMode =
  | 'rayon-jobs'
  | 'serial-k'
  | 'gemm-seq-by-design'
  | 'browser-worker'
  | 'py-worker-thread'
  | 'none';

/** Grid unit in world pixels. All node coordinates are in grid units. */
export const GRID = 16;

export interface ArchNode {
  /** Dot-separated, e.g. 'core.eigensolver.svqb'. */
  id: string;
  type: NodeType;
  kind: NodeKind;
  label: string;
  /** Tooltip one-liner. */
  short: string;
  /** Detail-panel prose. Plain text; `backticks` render as code. */
  description?: string;
  /** Nesting: parent group/crate id. Coordinates are local to the parent. */
  parent?: string;
  /** Position in LOCAL grid units (top-left corner within parent's content area). */
  x: number;
  y: number;
  /** Size in grid units. Groups auto-size from children when omitted. */
  w?: number;
  h?: number;
  /** Repo-relative path -> GitHub blob link in the detail panel. */
  file?: string;
  loc?: number;
  /** e.g. 'O(N log N)', '6 FFTs / apply'. */
  complexity?: string;
  precision?: Precision;
  parallel?: ParallelMode;
  /** Site-relative deep-dive page, e.g. '/architecture/eigensolver'. */
  subpage?: string;
  /** Extra key/value rows for the detail panel. */
  metrics?: [string, string][];
  /** Small corner badge text, e.g. 'Par::Seq', 'lock'. */
  badge?: string;
  /** Dashed frame + muted fill for stubs/placeholders. */
  stub?: boolean;
}

/** Parameters of the depicted solve; feeds byte formulas and the timeline. */
export interface SolveParams {
  nx: number;
  ny: number;
  bands: number;
  kPoints: number;
  jobs: number;
}

export type EdgeKind = 'data' | 'control';

export interface ArchEdge {
  id: string;
  from: string;
  to: string;
  kind: EdgeKind;
  label?: string;
  /** What travels, e.g. 'X block: N x r Complex<R>'. */
  payload?: string;
  /** Bytes moved per traversal; drives thickness and packet rate. */
  bytes?: (p: SolveParams) => number;
  /** Marks an f32<->f64 boundary for the precision overlay. */
  precisionChange?: 'upcast' | 'downcast';
  bidirectional?: boolean;
  /** Hand-tuned waypoints in WORLD grid units (overrides the auto-router). */
  route?: [number, number][];
}

export interface ArchModel {
  nodes: ArchNode[];
  edges: ArchEdge[];
}

/* ------------------------------- timeline ------------------------------- */

export interface RunStage {
  id: string;
  label: string;
  /** Narration line shown in the caption bar during playback. */
  caption?: string;
  /** Model-milliseconds at 1x playback speed. */
  duration: number;
  /** Node ids that pulse during this stage. */
  nodes: string[];
  /** Edge ids that carry packets during this stage. */
  edges: string[];
  /** Packets per second per edge (defaults derived from edge bytes). */
  packetRate?: Record<string, number>;
}

export interface FlatStage {
  start: number;
  end: number;
  stage: RunStage;
}

export interface RunTimeline {
  params: SolveParams;
  totalMs: number;
  flat: FlatStage[];
}

/* ----------------------------- resolved layout --------------------------- */

/** World-space rectangle (pixels) produced by layout.ts. */
export interface NodeRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ResolvedEdge {
  edge: ArchEdge;
  /** Polyline in world pixels. */
  points: [number, number][];
  /** Prefix-summed arc lengths, same length as points. */
  arc: number[];
  totalLength: number;
}

export interface ResolvedLayout {
  rects: Map<string, NodeRect>;
  edges: ResolvedEdge[];
  /** Depth of each node (0 = root group) for draw order. */
  depth: Map<string, number>;
  bounds: NodeRect;
}
