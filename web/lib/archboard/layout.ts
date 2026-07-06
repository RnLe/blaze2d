/**
 * Resolves the hand-tuned local-grid model into world-space pixels:
 * parent-chain offsets, group auto-sizing, and Manhattan edge routing.
 */

import type {
  ArchEdge,
  ArchModel,
  ArchNode,
  NodeRect,
  ResolvedEdge,
  ResolvedLayout,
} from './types';
import { GRID } from './types';

/** Padding inside a group frame around its children (grid units). */
const GROUP_PAD = 1.25;
/** Height reserved for the group title bar (grid units). */
const GROUP_TITLE = 2.25;

interface TreeNode {
  node: ArchNode;
  children: TreeNode[];
}

function buildTree(model: ArchModel): { roots: TreeNode[]; byId: Map<string, TreeNode> } {
  const byId = new Map<string, TreeNode>();
  for (const node of model.nodes) byId.set(node.id, { node, children: [] });
  const roots: TreeNode[] = [];
  for (const entry of byId.values()) {
    const parentId = entry.node.parent;
    if (parentId) {
      const parent = byId.get(parentId);
      if (!parent) throw new Error(`archboard: node '${entry.node.id}' has unknown parent '${parentId}'`);
      parent.children.push(entry);
    } else {
      roots.push(entry);
    }
  }
  return { roots, byId };
}

/**
 * Bottom-up size resolution: leaves use their declared w/h; groups wrap their
 * children plus padding and a title bar. Returns sizes in grid units.
 */
function resolveSize(tree: TreeNode, sizes: Map<string, { w: number; h: number }>): { w: number; h: number } {
  const { node, children } = tree;
  if (children.length === 0) {
    const size = { w: node.w ?? 10, h: node.h ?? 4 };
    sizes.set(node.id, size);
    return size;
  }
  let maxX = 0;
  let maxY = 0;
  for (const child of children) {
    const size = resolveSize(child, sizes);
    maxX = Math.max(maxX, child.node.x + size.w);
    maxY = Math.max(maxY, child.node.y + size.h);
  }
  const size = {
    w: (node.w ?? maxX + 2 * GROUP_PAD),
    h: (node.h ?? maxY + 2 * GROUP_PAD + GROUP_TITLE),
  };
  sizes.set(node.id, size);
  return size;
}

/** Top-down world placement (pixels). Children are offset by pad + title bar. */
function place(
  tree: TreeNode,
  originX: number,
  originY: number,
  sizes: Map<string, { w: number; h: number }>,
  rects: Map<string, NodeRect>,
  depth: Map<string, number>,
  level: number
): void {
  const { node, children } = tree;
  const size = sizes.get(node.id)!;
  const rect: NodeRect = {
    x: originX + node.x * GRID,
    y: originY + node.y * GRID,
    w: size.w * GRID,
    h: size.h * GRID,
  };
  rects.set(node.id, rect);
  depth.set(node.id, level);
  const childOriginX = rect.x + GROUP_PAD * GRID;
  const childOriginY = rect.y + (GROUP_PAD + GROUP_TITLE) * GRID;
  for (const child of children) {
    place(child, childOriginX, childOriginY, sizes, rects, depth, level + 1);
  }
}

const center = (r: NodeRect): [number, number] => [r.x + r.w / 2, r.y + r.h / 2];

/**
 * Attachment point on the rect border facing a target point, with Manhattan
 * routing between them: exit vertically or horizontally toward the target,
 * one bend at the midpoint when both axes differ.
 */
function routeEdge(from: NodeRect, to: NodeRect): [number, number][] {
  const [fx, fy] = center(from);
  const [tx, ty] = center(to);
  const dx = tx - fx;
  const dy = ty - fy;

  // Prefer the dominant axis for the exit side.
  if (Math.abs(dy) >= Math.abs(dx)) {
    const startY = dy > 0 ? from.y + from.h : from.y;
    const endY = dy > 0 ? to.y : to.y + to.h;
    if (Math.abs(dx) < 4) {
      return [
        [fx, startY],
        [fx, endY],
      ];
    }
    const midY = (startY + endY) / 2;
    return [
      [fx, startY],
      [fx, midY],
      [tx, midY],
      [tx, endY],
    ];
  }
  const startX = dx > 0 ? from.x + from.w : from.x;
  const endX = dx > 0 ? to.x : to.x + to.w;
  if (Math.abs(dy) < 4) {
    return [
      [startX, fy],
      [endX, fy],
    ];
  }
  const midX = (startX + endX) / 2;
  return [
    [startX, fy],
    [midX, fy],
    [midX, ty],
    [endX, ty],
  ];
}

function resolveEdge(edge: ArchEdge, rects: Map<string, NodeRect>): ResolvedEdge {
  const from = rects.get(edge.from);
  const to = rects.get(edge.to);
  if (!from || !to) throw new Error(`archboard: edge '${edge.id}' references unknown node`);

  const points: [number, number][] = edge.route
    ? edge.route.map(([gx, gy]) => [gx * GRID, gy * GRID] as [number, number])
    : routeEdge(from, to);

  const arc: number[] = [0];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    total += Math.hypot(x1 - x0, y1 - y0);
    arc.push(total);
  }
  return { edge, points, arc, totalLength: total };
}

export function resolveLayout(model: ArchModel): ResolvedLayout {
  const { roots } = buildTree(model);
  const sizes = new Map<string, { w: number; h: number }>();
  for (const root of roots) resolveSize(root, sizes);

  const rects = new Map<string, NodeRect>();
  const depth = new Map<string, number>();
  for (const root of roots) place(root, 0, 0, sizes, rects, depth, 0);

  const edges = model.edges.map((edge) => resolveEdge(edge, rects));

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const rect of rects.values()) {
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.w);
    maxY = Math.max(maxY, rect.y + rect.h);
  }

  return {
    rects,
    edges,
    depth,
    bounds: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
  };
}

/** Point at arclength t·total along a resolved polyline (for packets). */
export function pointAt(edge: ResolvedEdge, t: number): [number, number] {
  const target = t * edge.totalLength;
  const { points, arc } = edge;
  for (let i = 1; i < points.length; i++) {
    if (target <= arc[i] || i === points.length - 1) {
      const segLen = arc[i] - arc[i - 1] || 1;
      const local = (target - arc[i - 1]) / segLen;
      const [x0, y0] = points[i - 1];
      const [x1, y1] = points[i];
      return [x0 + (x1 - x0) * local, y0 + (y1 - y0) * local];
    }
  }
  return points[points.length - 1];
}
