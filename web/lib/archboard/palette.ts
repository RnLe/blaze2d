/**
 * The board's visual language. One source of truth for all archboard
 * rendering (pixi) and chrome (React legend, chips, cards).
 *
 * Contrast-validated against the board surface #0a0a0a (dark): all four
 * chromatic roles hold >= 3:1 contrast and pairwise CVD distance.
 * Brand anchors: blaze orange #eb7929, blaze blue #317cb8.
 */

import type { NodeKind, Precision } from './types';

export const SURFACE = 0x0a0a0a;
export const NODE_FILL = 0x141414;
export const GROUP_FILL = 0x0f0f0f;
export const TEXT_PRIMARY = 0xffffff;
export const TEXT_SECONDARY = 0xb3b3b3;

export const KIND_COLORS: Record<NodeKind, number> = {
  compute: 0xe06e1c, // brand orange, dark-band
  movement: 0x3987e5, // brand blue, stepped up for dark surface
  memory: 0xc65b9b, // magenta-violet
  interface: 0x1baf7a, // aqua-green
  control: 0x8b949e, // recessive neutral; control edges are also dashed
};

/** Hover/glow accent (pure brand orange is reserved for this). */
export const ACCENT = 0xeb7929;

export const PRECISION_COLORS: Record<Precision, number> = {
  f32: 0xd9a13b, // amber: storage / half precision
  f64: 0x4a9eff, // blue: accumulate / full precision
  mixed: 0x4a9eff, // mixed renders as split fill f32|f64; this is the fallback
  generic: 0x6e7681, // gray: monomorphized over R
};

export const PACKET_COLORS: Record<'data' | 'control', number> = {
  data: 0xc9d1d9, // neutral: data payload with no declared precision
  control: 0xc9d1d9,
};

/** Packet tints for edges that declare a payload precision (brighter than node fills). */
export const PACKET_PRECISION_COLORS: Record<'f32' | 'f64', number> = {
  f32: 0xf5c04a,
  f64: 0x8fc1ff,
};

export const KIND_LABELS: Record<NodeKind, string> = {
  compute: 'Computation',
  movement: 'Data movement',
  memory: 'Memory / buffers',
  interface: 'Interface / boundary',
  control: 'Control / orchestration',
};

export const KIND_DESCRIPTIONS: Record<NodeKind, string> = {
  compute: 'Kernels that burn FLOPs: FFTs, GEMMs, dense eigensolves.',
  movement: 'Where bytes move: copies, casts, marshaling, channels.',
  memory: 'Long-lived state: blocks, caches, scratch workspaces.',
  interface: 'Boundaries: config in, results out, FFI surfaces.',
  control: 'Orchestration and decisions; no bulk data touched.',
};

/** CSS hex strings for React chrome (legend, chips, cards). */
export const cssColor = (c: number): string => `#${c.toString(16).padStart(6, '0')}`;

export const KIND_CSS: Record<NodeKind, string> = Object.fromEntries(
  Object.entries(KIND_COLORS).map(([k, v]) => [k, cssColor(v)])
) as Record<NodeKind, string>;

export const PRECISION_CSS: Record<Precision, string> = Object.fromEntries(
  Object.entries(PRECISION_COLORS).map(([k, v]) => [k, cssColor(v)])
) as Record<Precision, string>;

/** Dim factor applied to inactive elements during Run playback. */
export const RUN_DIM_ALPHA = 0.15;
