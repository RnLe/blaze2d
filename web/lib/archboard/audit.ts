/**
 * Dev-time routing audit: reports edge polylines that cut through leaf node
 * rects they don't connect. Rect bounds are reported in grid units so fixes
 * can be written directly as `route:` waypoints in model.ts.
 */

import type { ArchModel, NodeRect, ResolvedLayout } from './types';
import { GRID } from './types';

export interface EdgeCrossing {
  edgeId: string;
  nodeId: string;
  /** Length of the polyline segment inside the rect (px). */
  overlapPx: number;
  /** Crossed rect in grid units: "x y w h" (for authoring waypoints). */
  rectGrid: string;
}

/** Ignore grazes shorter than this (px); borders touch legitimately. */
const MIN_OVERLAP = 6;
/** Inset applied to rects so segments running along a border don't count. */
const INSET = 2;

function segmentRectOverlap(a: [number, number], b: [number, number], r: NodeRect): number {
  const x0 = r.x + INSET;
  const y0 = r.y + INSET;
  const x1 = r.x + r.w - INSET;
  const y1 = r.y + r.h - INSET;
  if (a[0] === b[0]) {
    // vertical
    if (a[0] <= x0 || a[0] >= x1) return 0;
    const lo = Math.min(a[1], b[1]);
    const hi = Math.max(a[1], b[1]);
    return Math.max(0, Math.min(hi, y1) - Math.max(lo, y0));
  }
  if (a[1] === b[1]) {
    // horizontal
    if (a[1] <= y0 || a[1] >= y1) return 0;
    const lo = Math.min(a[0], b[0]);
    const hi = Math.max(a[0], b[0]);
    return Math.max(0, Math.min(hi, x1) - Math.max(lo, x0));
  }
  return 0; // routes are Manhattan; diagonal segments don't occur
}

export function auditEdgeRoutes(layout: ResolvedLayout, model: ArchModel): EdgeCrossing[] {
  const parents = new Set(model.nodes.map((n) => n.parent).filter(Boolean));
  const leaves = model.nodes.filter((n) => !parents.has(n.id));

  const crossings: EdgeCrossing[] = [];
  for (const resolved of layout.edges) {
    const { edge, points } = resolved;
    for (const leaf of leaves) {
      if (leaf.id === edge.from || leaf.id === edge.to) continue;
      const rect = layout.rects.get(leaf.id);
      if (!rect) continue;
      let overlap = 0;
      for (let i = 1; i < points.length; i++) {
        overlap += segmentRectOverlap(points[i - 1], points[i], rect);
      }
      if (overlap >= MIN_OVERLAP) {
        const g = (v: number) => Math.round((v / GRID) * 4) / 4;
        crossings.push({
          edgeId: edge.id,
          nodeId: leaf.id,
          overlapPx: Math.round(overlap),
          rectGrid: `${g(rect.x)} ${g(rect.y)} ${g(rect.w)} ${g(rect.h)}`,
        });
      }
    }
  }
  return crossings.sort((a, b) => b.overlapPx - a.overlapPx);
}
