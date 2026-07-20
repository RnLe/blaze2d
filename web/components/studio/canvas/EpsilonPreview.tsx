'use client';

import React, { useEffect, useRef } from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { latticeVectors, cartToFrac } from '../../../lib/studio/geometry';

/**
 * A client-side raster of the dielectric map over one unit cell.
 * This is a preview: circles are drawn hard-edged (no subpixel smoothing),
 * which the engine does properly at solve time.
 */
export function EpsilonPreview({
  size = 96,
  displaySize = 88,
}: {
  /** Internal raster resolution. */
  size?: number;
  /** Rendered CSS size. */
  displaySize?: number;
}) {
  const geometry = useStudioStore((s) => s.config.geometry);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const SIZE = size;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const basis = latticeVectors(geometry.lattice);
    const img = ctx.createImageData(SIZE, SIZE);
    const epsBg = geometry.eps_bg;

    // eps range for coloring
    let epsMax = epsBg;
    for (const a of geometry.atoms) epsMax = Math.max(epsMax, a.eps_inside);
    const epsMin = Math.min(epsBg, ...geometry.atoms.map((a) => a.eps_inside));
    const span = Math.max(1e-6, epsMax - epsMin);

    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        // fractional coords across the cell
        const fx = px / SIZE;
        const fy = 1 - py / SIZE;
        // determine eps: check membership in any atom (accounting for wrap)
        let eps = epsBg;
        for (const atom of geometry.atoms) {
          // nearest periodic image distance in cartesian
          let dfx = fx - atom.pos[0];
          let dfy = fy - atom.pos[1];
          dfx -= Math.round(dfx);
          dfy -= Math.round(dfy);
          const cart: [number, number] = [
            dfx * basis.a1[0] + dfy * basis.a2[0],
            dfx * basis.a1[1] + dfy * basis.a2[1],
          ];
          const dist = Math.hypot(cart[0], cart[1]);
          if (dist <= atom.radius) {
            eps = atom.eps_inside;
            break;
          }
        }
        const t = (eps - epsMin) / span;
        // grey (low) -> blue (high)
        const r = Math.round(90 + (57 - 90) * t);
        const g = Math.round(96 + (135 - 96) * t);
        const b = Math.round(110 + (229 - 110) * t);
        const idx = (py * SIZE + px) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    void cartToFrac; // keep import used across refactors
  }, [geometry, SIZE]);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      style={{
        width: displaySize,
        height: displaySize,
        imageRendering: SIZE >= 200 ? 'auto' : 'pixelated',
        borderRadius: 8,
        border: '1px solid rgba(128,128,128,0.25)',
      }}
      aria-label="Dielectric map preview"
    />
  );
}
