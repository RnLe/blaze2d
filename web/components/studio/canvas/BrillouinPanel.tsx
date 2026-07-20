'use client';

import React, { useCallback, useMemo, useRef } from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { cssColor, KIND_COLORS } from '../../../lib/archboard/palette';
import {
  latticeVectors,
  reciprocal,
  brillouinZone,
  resolvePreset,
  PRESET_CORNERS,
  PRESET_LABELS,
  type Vec2,
  type LatticeBasis,
} from '../../../lib/studio/geometry';

const VIEW = 320;
const PAD = 34;

/**
 * The first Brillouin zone with the current high-symmetry path overlaid.
 *
 * In preset mode this is read-only. In custom-points mode the k-points become
 * draggable handles: click empty space to append a point, drag to move, and
 * click a handle to remove it.
 */
export function BrillouinPanel() {
  const lattice = useStudioStore((s) => s.config.geometry.lattice);
  const path = useStudioStore((s) => s.config.path);
  const patch = useStudioStore((s) => s.patchConfig);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ index: number; moved: boolean } | null>(null);

  const editing = path.mode === 'points';

  const { basis, recip, zone, scale } = useMemo(() => {
    const basis = latticeVectors(lattice);
    const recip = reciprocal(basis);
    const zoneCart = brillouinZone(basis);

    // Fit zone (+ a bit of margin for far points) into the viewBox.
    const xs = zoneCart.map((p) => p[0]);
    const ys = zoneCart.map((p) => p[1]);
    const span =
      Math.max(Math.max(...xs, 0.1) - Math.min(...xs, -0.1), Math.max(...ys, 0.1) - Math.min(...ys, -0.1)) || 1;
    const scale = (VIEW - 2 * PAD) / span;
    return { basis, recip, zone: zoneCart, scale };
  }, [lattice]);

  const toScreen = useCallback(
    (p: Vec2): Vec2 => [VIEW / 2 + p[0] * scale, VIEW / 2 - p[1] * scale],
    [scale],
  );
  // fractional reciprocal coords -> cartesian
  const fracToCart = useCallback(
    (f: Vec2): Vec2 => [
      f[0] * recip.a1[0] + f[1] * recip.a2[0],
      f[0] * recip.a1[1] + f[1] * recip.a2[1],
    ],
    [recip],
  );
  // cartesian -> fractional reciprocal
  const cartToFrac = useCallback(
    (c: Vec2): Vec2 => {
      const b1 = recip.a1;
      const b2 = recip.a2;
      const det = b1[0] * b2[1] - b1[1] * b2[0];
      if (Math.abs(det) < 1e-12) return [0, 0];
      const inv = 1 / det;
      return [
        (b2[1] * c[0] - b2[0] * c[1]) * inv,
        (-b1[1] * c[0] + b1[0] * c[1]) * inv,
      ];
    },
    [recip],
  );

  // Points to draw: preset corners or the custom point list.
  const { pts, labels } = useMemo(() => {
    if (editing) {
      return { pts: path.points as Vec2[], labels: [] as string[] };
    }
    const presetName = resolvePreset(path.preset, lattice.kind);
    const cornersFrac: Vec2[] = presetName ? (PRESET_CORNERS[presetName] ?? []) : [];
    const labels = presetName ? (PRESET_LABELS[presetName] ?? []) : [];
    return { pts: cornersFrac, labels };
  }, [editing, path.points, path.preset, lattice.kind]);

  const screenToFrac = useCallback(
    (clientX: number, clientY: number): Vec2 => {
      const svg = svgRef.current!;
      const rect = svg.getBoundingClientRect();
      const vx = ((clientX - rect.left) / rect.width) * VIEW;
      const vy = ((clientY - rect.top) / rect.height) * VIEW;
      const cart: Vec2 = [(vx - VIEW / 2) / scale, -(vy - VIEW / 2) / scale];
      const frac = cartToFrac(cart);
      return [Math.round(frac[0] * 1000) / 1000, Math.round(frac[1] * 1000) / 1000];
    },
    [scale, cartToFrac],
  );

  const onBackgroundClick = (e: React.MouseEvent) => {
    if (!editing) return;
    if (dragRef.current) return;
    const frac = screenToFrac(e.clientX, e.clientY);
    patch((d) => {
      d.path.points.push(frac);
    });
  };

  const onPointMove = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current.moved = true;
    const frac = screenToFrac(e.clientX, e.clientY);
    const idx = dragRef.current.index;
    patch((d) => {
      d.path.points[idx] = frac;
    });
  };

  const orange = cssColor(KIND_COLORS.compute);
  const blue = cssColor(KIND_COLORS.movement);

  const zonePath =
    zone.length > 0
      ? zone.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toScreen(p).join(' ')}`).join(' ') + ' Z'
      : '';

  const screenPts = pts.map((f) => toScreen(fracToCart(f)));

  return (
    <svg
      ref={svgRef}
      className="studio__canvas"
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      width={VIEW}
      height={VIEW}
      style={{ cursor: editing ? 'crosshair' : 'default' }}
      onClick={onBackgroundClick}
      onPointerMove={onPointMove}
      onPointerUp={(e) => {
        dragRef.current = null;
        (e.target as Element).releasePointerCapture?.(e.pointerId);
      }}
    >
      {zonePath ? (
        <path d={zonePath} fill="rgba(57,135,229,0.06)" stroke="rgba(57,135,229,0.55)" strokeWidth={1.4} />
      ) : null}

      {screenPts.length > 1 ? (
        <polyline
          points={screenPts.map((p) => p.join(',')).join(' ')}
          fill="none"
          stroke={orange}
          strokeWidth={1.8}
        />
      ) : null}

      {screenPts.map((s, i) => {
        if (!editing && i === screenPts.length - 1 && labels[i] === labels[0]) return null;
        return (
          <g key={i}>
            <circle
              cx={s[0]}
              cy={s[1]}
              r={editing ? 5 : 3.5}
              fill={blue}
              stroke={editing ? '#fff' : 'none'}
              strokeWidth={editing ? 1 : 0}
              style={{ cursor: editing ? 'grab' : 'default' }}
              onPointerDown={
                editing
                  ? (e) => {
                      e.stopPropagation();
                      dragRef.current = { index: i, moved: false };
                      (e.target as Element).setPointerCapture?.(e.pointerId);
                    }
                  : undefined
              }
              onClick={
                editing
                  ? (e) => {
                      e.stopPropagation();
                      // A click without drag removes the point.
                      if (dragRef.current?.moved) return;
                      patch((d) => {
                        d.path.points.splice(i, 1);
                      });
                    }
                  : undefined
              }
            />
            {!editing ? (
              <text
                x={s[0] + 7}
                y={s[1] - 6}
                fill="#d0d0d0"
                fontSize={12}
                fontFamily="var(--font-mono, ui-monospace, monospace)"
              >
                {labels[i]}
              </text>
            ) : (
              <text
                x={s[0] + 7}
                y={s[1] - 6}
                fill="#8a8a8a"
                fontSize={10}
                fontFamily="var(--font-mono, ui-monospace, monospace)"
              >
                {i}
              </text>
            )}
          </g>
        );
      })}

      {screenPts.length === 0 && !editing ? (
        <text x={VIEW / 2} y={VIEW / 2} fill="#7c7c7c" fontSize={12} textAnchor="middle">
          No standard path for {lattice.kind}
        </text>
      ) : null}
      {editing && screenPts.length === 0 ? (
        <text x={VIEW / 2} y={VIEW / 2} fill="#7c7c7c" fontSize={12} textAnchor="middle">
          Click to add k-points
        </text>
      ) : null}
    </svg>
  );
}

// keep the basis type import used
export type { LatticeBasis };
