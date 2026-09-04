'use client';
import { useEffect, useRef } from 'react';
import type { Preview } from '../../lib/compute/protocol';
import { useSize } from './useSize';

export function GeometryPreview({ preview }: { preview?: Preview }) {
  const { ref, width, height, dpr } = useSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!canvas.current || !preview || !width || !height) return;
    const ctx = canvas.current.getContext('2d');
    if (!ctx) return;
    canvas.current.width = Math.round(width * dpr);
    canvas.current.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [nx, ny] = preview.resolution;
    const raster = document.createElement('canvas'); raster.width = nx; raster.height = ny;
    const pixels = new ImageData(nx, ny);
    let minimum = Infinity, maximum = -Infinity;
    for (const value of preview.epsilon) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
    for (let i = 0; i < preview.epsilon.length; i++) {
      const t = (preview.epsilon[i] - minimum) / (maximum - minimum || 1);
      pixels.data.set([28 + 70 * t, 36 + 160 * t, 42 + 138 * t, 255], i * 4);
    }
    raster.getContext('2d')!.putImageData(pixels, 0, 0);
    const [a, b] = preview.lattice_vectors;
    const vertices = [[0, 0], a, [a[0] + b[0], a[1] + b[1]], b];
    const xs = vertices.map(v => v[0]), ys = vertices.map(v => v[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const scale = Math.min((width - 72) / (xmax - xmin), (height - 88) / (ymax - ymin));
    const x0 = (width - scale * (xmin + xmax)) / 2, y0 = (height + scale * (ymin + ymax)) / 2 - 12;
    ctx.save(); ctx.transform(a[0] * scale, -a[1] * scale, b[0] * scale, -b[1] * scale, x0, y0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(raster, 0, 0, 1, 1); ctx.restore();
    ctx.strokeStyle = '#9ac4be'; ctx.lineWidth = 1; ctx.beginPath();
    vertices.forEach((v, i) => { if (i === 0) ctx.moveTo(x0 + v[0] * scale, y0 - v[1] * scale); else ctx.lineTo(x0 + v[0] * scale, y0 - v[1] * scale); });
    ctx.closePath(); ctx.stroke();
    ctx.font = '13px system-ui'; ctx.fillStyle = '#c6d5d2'; ctx.textAlign = 'center';
    ctx.fillText(`ε: ${minimum.toPrecision(3)} to ${maximum.toPrecision(3)}  ·  ${nx} × ${ny} samples`, width / 2, height - 18);
  }, [preview, width, height, dpr]);
  return <figure className="wb-preview" ref={ref}>
    <canvas ref={canvas} aria-label="Dielectric distribution in one primitive cell" role="img" />
    {!preview && <p className="wb-placeholder">The dielectric preview appears after validation.</p>}
  </figure>;
}
