'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { cssColor, KIND_COLORS } from '../../../lib/archboard/palette';
import { latticeVectors, cartToFrac, type Vec2 } from '../../../lib/studio/geometry';

const VIEW = 720; // svg viewBox size
const PAD = 16;
/** Cells visible across the viewport at zoom 1 ("slightly larger than 3x3"). */
const BASE_SPAN_CELLS = 3.4;
const ZOOM_MIN = 0.5; // ~6.8 cells visible
const ZOOM_MAX = 1.7; // ~2.0 cells visible
/** Periodic images iterated; covers the widest zoom-out. */
const IMAGE_RANGE: number[] = [-3, -2, -1, 0, 1, 2, 3, 4];

/** Interpolate a fill color for an atom by its permittivity contrast vs background. */
function atomFill(epsInside: number, epsBg: number): string {
  // Blue-ish for high-index rods, grey for near-background.
  const contrast = Math.min(1, Math.abs(epsInside - epsBg) / 12);
  const lo = [90, 96, 110];
  const hi = [57, 135, 229];
  const rgb = lo.map((c, i) => Math.round(c + (hi[i] - c) * contrast));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/**
 * The crystal patch: a viewport of ~3.4x3.4 unit cells centered on the
 * primary cell, with every periodic image drawn at full opacity wherever it
 * intersects the viewport (real boundary wrapping). Primary-cell atoms are
 * draggable; wheel zooms between ~2 and ~7 visible cells.
 */
export function CrystalCanvas() {
  const geometry = useStudioStore((s) => s.config.geometry);
  const selectedAtom = useStudioStore((s) => s.ui.selectedAtom);
  const selectAtom = useStudioStore((s) => s.selectAtom);
  const patch = useStudioStore((s) => s.patchConfig);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ atom: number } | null>(null);
  const [zoom, setZoom] = useState(1);

  const basis = useMemo(() => latticeVectors(geometry.lattice), [geometry.lattice]);

  const { toScreen, aScreen } = useMemo(() => {
    // Bounding box of one cell {0, a1, a2, a1+a2}.
    const corners: Vec2[] = [
      [0, 0],
      basis.a1,
      basis.a2,
      [basis.a1[0] + basis.a2[0], basis.a1[1] + basis.a2[1]],
    ];
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    const cellW = Math.max(...xs) - Math.min(...xs) || 1;
    const cellH = Math.max(...ys) - Math.min(...ys) || 1;

    const worldSpan = (BASE_SPAN_CELLS / zoom) * Math.max(cellW, cellH);
    const scale = (VIEW - 2 * PAD) / worldSpan;
    // Center on the primary cell's centroid.
    const cx = (basis.a1[0] + basis.a2[0]) / 2;
    const cy = (basis.a1[1] + basis.a2[1]) / 2;
    const toScreen = (p: Vec2): Vec2 => [
      VIEW / 2 + (p[0] - cx) * scale,
      // flip y (screen y grows down)
      VIEW / 2 - (p[1] - cy) * scale,
    ];
    return { toScreen, aScreen: scale };
  }, [basis, zoom]);

  // Cartesian of a fractional position in cell (i, j).
  const cellPoint = useCallback(
    (frac: Vec2, i: number, j: number): Vec2 => [
      (frac[0] + i) * basis.a1[0] + (frac[1] + j) * basis.a2[0],
      (frac[0] + i) * basis.a1[1] + (frac[1] + j) * basis.a2[1],
    ],
    [basis],
  );

  // --- wheel zoom (non-passive so preventDefault works; pinch = ctrl+wheel) ---
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      setZoom((z) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z * factor)));
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current || !svgRef.current) return;
      const svg = svgRef.current;
      const rect = svg.getBoundingClientRect();
      // Screen px -> viewBox coords (the svg is letterboxed to a square).
      const side = Math.min(rect.width, rect.height);
      const offX = rect.left + (rect.width - side) / 2;
      const offY = rect.top + (rect.height - side) / 2;
      const vx = ((e.clientX - offX) / side) * VIEW;
      const vy = ((e.clientY - offY) / side) * VIEW;
      // viewBox -> cartesian (invert toScreen via the origin probe).
      const [ox, oy] = toScreen([0, 0]);
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

  const orange = cssColor(KIND_COLORS.compute);

  /** Cull helper: is a circle at screen p with radius r anywhere near the view? */
  const visible = (p: Vec2, r: number): boolean =>
    p[0] + r > 0 && p[0] - r < VIEW && p[1] + r > 0 && p[1] - r < VIEW;

  // Lattice grid lines along both directions, clipped to the viewport.
  const gridLines = useMemo(() => {
    const lines: { key: string; a: Vec2; b: Vec2 }[] = [];
    const lo = IMAGE_RANGE[0];
    const hi = IMAGE_RANGE[IMAGE_RANGE.length - 1] + 1;
    for (let i = lo; i <= hi; i++) {
      lines.push({ key: `ga1-${i}`, a: toScreen(cellPoint([0, 0], i, lo)), b: toScreen(cellPoint([0, 0], i, hi)) });
      lines.push({ key: `ga2-${i}`, a: toScreen(cellPoint([0, 0], lo, i)), b: toScreen(cellPoint([0, 0], hi, i)) });
    }
    return lines;
  }, [toScreen, cellPoint]);

  // Primary-cell outline.
  const cellOutline = useMemo(() => {
    const c00 = toScreen(cellPoint([0, 0], 0, 0));
    const c10 = toScreen(cellPoint([1, 0], 0, 0));
    const c11 = toScreen(cellPoint([1, 1], 0, 0));
    const c01 = toScreen(cellPoint([0, 1], 0, 0));
    return `M ${c00[0]} ${c00[1]} L ${c10[0]} ${c10[1]} L ${c11[0]} ${c11[1]} L ${c01[0]} ${c01[1]} Z`;
  }, [toScreen, cellPoint]);

  return (
    <>
      <svg
        ref={svgRef}
        className="studio__canvas"
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        width={VIEW}
        height={VIEW}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onDoubleClick={() => setZoom(1)}
      >
        <defs>
          <clipPath id="studio-cell-viewport">
            <rect x={0} y={0} width={VIEW} height={VIEW} rx={10} />
          </clipPath>
          <marker id="arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={orange} />
          </marker>
        </defs>

        <g clipPath="url(#studio-cell-viewport)">
          {/* lattice grid */}
          {gridLines.map((l) => (
            <line
              key={l.key}
              x1={l.a[0]}
              y1={l.a[1]}
              x2={l.b[0]}
              y2={l.b[1]}
              stroke="rgba(128,128,128,0.10)"
              strokeWidth={1}
            />
          ))}

          {/* periodic-image atoms: full opacity, non-interactive */}
          {IMAGE_RANGE.flatMap((i) =>
            IMAGE_RANGE.map((j) => {
              if (i === 0 && j === 0) return null;
              return geometry.atoms.map((atom, ai) => {
                const p = toScreen(cellPoint(atom.pos, i, j));
                const r = atom.radius * aScreen;
                if (!visible(p, r)) return null;
                return (
                  <circle
                    key={`img-${i}-${j}-${ai}`}
                    cx={p[0]}
                    cy={p[1]}
                    r={r}
                    fill={atomFill(atom.eps_inside, geometry.eps_bg)}
                    stroke="rgba(255,255,255,0.10)"
                    strokeWidth={1}
                    pointerEvents="none"
                  />
                );
              });
            }),
          )}

          {/* primary cell highlight */}
          <path
            d={cellOutline}
            fill="rgba(57,135,229,0.05)"
            stroke="rgba(57,135,229,0.55)"
            strokeWidth={1.6}
          />

          {/* lattice vectors */}
          {(() => {
            const o = toScreen([0, 0]);
            const a1 = toScreen(basis.a1);
            const a2 = toScreen(basis.a2);
            return (
              <>
                <line x1={o[0]} y1={o[1]} x2={a1[0]} y2={a1[1]} stroke={orange} strokeWidth={1.8} markerEnd="url(#arrow)" />
                <line x1={o[0]} y1={o[1]} x2={a2[0]} y2={a2[1]} stroke={orange} strokeWidth={1.8} markerEnd="url(#arrow)" />
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
                  stroke={isSel ? orange : 'rgba(255,255,255,0.55)'}
                  strokeWidth={isSel ? 2.5 : 1.4}
                  style={{ cursor: 'grab' }}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    selectAtom(ai);
                    dragRef.current = { atom: ai };
                    (e.target as Element).setPointerCapture?.(e.pointerId);
                  }}
                />
                <circle cx={p[0]} cy={p[1]} r={2.5} fill="rgba(255,255,255,0.75)" pointerEvents="none" />
              </g>
            );
          })}
        </g>
      </svg>

      {zoom !== 1 ? (
        <button
          type="button"
          className="studio__iconbtn studio__canvas-reset"
          aria-label="Reset zoom"
          title="Reset zoom (double-click)"
          onClick={() => setZoom(1)}
        >
          ⌂
        </button>
      ) : null}
    </>
  );
}
