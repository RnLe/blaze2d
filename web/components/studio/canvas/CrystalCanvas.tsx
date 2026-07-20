'use client';

import React, { useCallback, useMemo, useRef } from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { cssColor, KIND_COLORS } from '../../../lib/archboard/palette';
import { latticeVectors, cartToFrac, type Vec2 } from '../../../lib/studio/geometry';

const VIEW = 320; // svg viewBox size
const PAD = 28;

/** Interpolate a fill color for an atom by its permittivity contrast vs background. */
function atomFill(epsInside: number, epsBg: number): string {
  // Blue-ish for high-index rods, grey for near-background.
  const contrast = Math.min(1, Math.abs(epsInside - epsBg) / 12);
  const lo = [90, 96, 110];
  const hi = [57, 135, 229];
  const rgb = lo.map((c, i) => Math.round(c + (hi[i] - c) * contrast));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export function CrystalCanvas() {
  const geometry = useStudioStore((s) => s.config.geometry);
  const selectedAtom = useStudioStore((s) => s.ui.selectedAtom);
  const selectAtom = useStudioStore((s) => s.selectAtom);
  const patch = useStudioStore((s) => s.patchConfig);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ atom: number } | null>(null);

  const basis = useMemo(() => latticeVectors(geometry.lattice), [geometry.lattice]);

  // Fit a 3x3 tiling of the unit cell into the viewBox.
  const { toScreen, aScreen } = useMemo(() => {
    const corners: Vec2[] = [];
    for (let i = -1; i <= 2; i++) {
      for (let j = -1; j <= 2; j++) {
        corners.push([
          i * basis.a1[0] + j * basis.a2[0],
          i * basis.a1[1] + j * basis.a2[1],
        ]);
      }
    }
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const span = Math.max(maxX - minX, maxY - minY) || 1;
    const scale = (VIEW - 2 * PAD) / span;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const toScreen = (p: Vec2): Vec2 => [
      VIEW / 2 + (p[0] - cx) * scale,
      // flip y (screen y grows down)
      VIEW / 2 - (p[1] - cy) * scale,
    ];
    return { toScreen, aScreen: scale };
  }, [basis]);

  // Cartesian of a fractional position in cell (i, j).
  const cellPoint = useCallback(
    (frac: Vec2, i: number, j: number): Vec2 => [
      (frac[0] + i) * basis.a1[0] + (frac[1] + j) * basis.a2[0],
      (frac[0] + i) * basis.a1[1] + (frac[1] + j) * basis.a2[1],
    ],
    [basis],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current || !svgRef.current) return;
      const svg = svgRef.current;
      const rect = svg.getBoundingClientRect();
      // Screen px -> viewBox coords.
      const vx = ((e.clientX - rect.left) / rect.width) * VIEW;
      const vy = ((e.clientY - rect.top) / rect.height) * VIEW;
      // viewBox -> cartesian (invert toScreen).
      // toScreen: sx = VIEW/2 + (x - cx)*scale ; sy = VIEW/2 - (y - cy)*scale
      // We need cx, cy, scale — recompute inverse via a probe.
      const [ox] = toScreen([0, 0]);
      const [, oy] = toScreen([0, 0]);
      const cartX = (vx - ox) / aScreen;
      const cartY = -(vy - oy) / aScreen;
      const frac = cartToFrac(basis, [cartX, cartY]);
      const wrap = (v: number) => ((v % 1) + 1) % 1;
      patch((d) => {
        d.geometry.atoms[dragRef.current!.atom].pos = [wrap(frac[0]), wrap(frac[1])];
      });
    },
    [aScreen, basis, patch, toScreen],
  );

  const endDrag = useCallback((e: React.PointerEvent) => {
    dragRef.current = null;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  }, []);

  // Cell outline path (primary + neighbors).
  const cellPath = (i: number, j: number): string => {
    const c00 = toScreen(cellPoint([0, 0], i, j));
    const c10 = toScreen(cellPoint([1, 0], i, j));
    const c11 = toScreen(cellPoint([1, 1], i, j));
    const c01 = toScreen(cellPoint([0, 1], i, j));
    return `M ${c00[0]} ${c00[1]} L ${c10[0]} ${c10[1]} L ${c11[0]} ${c11[1]} L ${c01[0]} ${c01[1]} Z`;
  };

  const orange = cssColor(KIND_COLORS.compute);

  return (
    <svg
      ref={svgRef}
      className="studio__canvas"
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      width={VIEW}
      height={VIEW}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
    >
      {/* neighbor cells (dimmed) */}
      {[-1, 0, 1].flatMap((i) =>
        [-1, 0, 1].map((j) => {
          if (i === 0 && j === 0) return null;
          return (
            <path
              key={`cell-${i}-${j}`}
              d={cellPath(i, j)}
              fill="none"
              stroke="rgba(128,128,128,0.14)"
              strokeWidth={1}
            />
          );
        }),
      )}

      {/* neighbor atoms (dimmed, non-interactive) */}
      {[-1, 0, 1].flatMap((i) =>
        [-1, 0, 1].map((j) => {
          if (i === 0 && j === 0) return null;
          return geometry.atoms.map((atom, ai) => {
            const p = toScreen(cellPoint(atom.pos, i, j));
            return (
              <circle
                key={`a-${i}-${j}-${ai}`}
                cx={p[0]}
                cy={p[1]}
                r={atom.radius * aScreen}
                fill={atomFill(atom.eps_inside, geometry.eps_bg)}
                opacity={0.28}
              />
            );
          });
        }),
      )}

      {/* primary cell */}
      <path d={cellPath(0, 0)} fill="rgba(57,135,229,0.05)" stroke="rgba(57,135,229,0.5)" strokeWidth={1.4} />

      {/* lattice vectors */}
      {(() => {
        const o = toScreen([0, 0]);
        const a1 = toScreen(basis.a1);
        const a2 = toScreen(basis.a2);
        return (
          <>
            <line x1={o[0]} y1={o[1]} x2={a1[0]} y2={a1[1]} stroke={orange} strokeWidth={1.6} markerEnd="url(#arrow)" />
            <line x1={o[0]} y1={o[1]} x2={a2[0]} y2={a2[1]} stroke={orange} strokeWidth={1.6} markerEnd="url(#arrow)" />
          </>
        );
      })()}

      {/* primary atoms (draggable) */}
      {geometry.atoms.map((atom, ai) => {
        const p = toScreen(cellPoint(atom.pos, 0, 0));
        const isSel = selectedAtom === ai;
        return (
          <g key={`primary-${ai}`}>
            <circle
              cx={p[0]}
              cy={p[1]}
              r={atom.radius * aScreen}
              fill={atomFill(atom.eps_inside, geometry.eps_bg)}
              stroke={isSel ? orange : 'rgba(255,255,255,0.5)'}
              strokeWidth={isSel ? 2 : 1}
              style={{ cursor: 'grab' }}
              onPointerDown={(e) => {
                e.preventDefault();
                selectAtom(ai);
                dragRef.current = { atom: ai };
                (e.target as Element).setPointerCapture?.(e.pointerId);
              }}
            />
            <circle cx={p[0]} cy={p[1]} r={2} fill="rgba(255,255,255,0.7)" pointerEvents="none" />
          </g>
        );
      })}

      <defs>
        <marker id="arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={orange} />
        </marker>
      </defs>
    </svg>
  );
}
