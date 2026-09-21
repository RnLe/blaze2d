'use client';
import { useEffect, useRef, useState } from 'react';
import type { Geometry } from '@/lib/contract/generated';
import type { Preview } from '@/lib/compute/protocol';
import { toCartesian, toFractional, type CoordinateSystem } from './coordinates';
import { useSize } from './useSize';
import { epsilonColor, epsilonRgb, theme } from '@/lib/theme';

export function GeometryPreview({ preview, geometry, coordinates = 'fractional' }: { preview?: Preview; geometry?: Geometry; coordinates?: CoordinateSystem }) {
  const { ref, width, height, dpr } = useSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null), transform = useRef({ x: 0, y: 0, scale: 1 });
  const [periods, setPeriods] = useState(5), [mode, setMode] = useState('geometry'), [cursor, setCursor] = useState<[number, number]>();
  useEffect(() => {
    if (!canvas.current || !width || !height) return;
    const ctx = canvas.current.getContext('2d');
    if (!ctx) return;
    // Canvas cannot resolve CSS custom properties, so the UI font is read back
    // from the element it inherits (next/font gives the family a hashed name).
    const uiFont = getComputedStyle(canvas.current).fontFamily || 'system-ui';
    canvas.current.width = Math.round(width * dpr); canvas.current.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    if (!preview) return;
    const [nx, ny] = preview.resolution;
    const raster = document.createElement('canvas'); raster.width = nx; raster.height = ny;
    const pixels = new ImageData(nx, ny);
    let minimum = Infinity, maximum = -Infinity;
    for (const value of preview.epsilon) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
    for (let i = 0; i < preview.epsilon.length; i++) {
      const t = (preview.epsilon[i] - minimum) / (maximum - minimum || 1);
      pixels.data.set([...epsilonRgb(t), 255], i * 4);
    }
    raster.getContext('2d')!.putImageData(pixels, 0, 0);
    const [a, b] = preview.lattice_vectors, lo = -Math.floor(periods / 2), hi = lo + periods;
    const corners = [[lo, lo], [hi, lo], [hi, hi], [lo, hi]].map(point => toCartesian(point, [a, b]));
    const xs = corners.map(v => v[0]), ys = corners.map(v => v[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const scale = Math.min((width - 54) / (xmax - xmin), (height - 100) / (ymax - ymin));
    const x0 = (width - scale * (xmin + xmax)) / 2, y0 = (height + scale * (ymin + ymax)) / 2 + 4;
    transform.current = { x: x0, y: y0, scale };
    const point = (u: number, v: number) => [x0 + (a[0] * u + b[0] * v) * scale, y0 - (a[1] * u + b[1] * v) * scale];
    const determinant = Math.abs(a[0] * b[1] - a[1] * b[0]);
    const copies = geometry?.objects?.map(object => ({ object, du: object.radius * Math.hypot(...b) / determinant, dv: object.radius * Math.hypot(...a) / determinant })) ?? [];
    const vector = mode === 'geometry' && geometry && copies.reduce((count, {du,dv}) => count + (periods + 2 * Math.ceil(du) + 3) * (periods + 2 * Math.ceil(dv) + 3), 0) <= 4096;
    if (vector) {
      const epsilon = [geometry.background_epsilon!, ...copies.map(({ object }) => object.epsilon)];
      const low = Math.min(...epsilon), high = Math.max(...epsilon);
      const color = (value: number) => epsilonColor((value - low) / (high - low || 1));
      ctx.save(); ctx.beginPath();
      [[lo,lo],[hi,lo],[hi,hi],[lo,hi]].forEach(([u,v],i) => { const [x,y] = point(u,v); if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y); });
      ctx.closePath(); ctx.clip(); ctx.fillStyle = color(geometry.background_epsilon!); ctx.fillRect(0,0,width,height);
      for (const { object, du, dv } of copies) {
        const [u,v] = object.center!;
        ctx.fillStyle = color(object.epsilon); ctx.strokeStyle = theme.accent + '55'; ctx.lineWidth = .7;
        for (let iy = Math.floor(lo-v-dv); iy <= Math.ceil(hi-v+dv); iy++) for (let ix = Math.floor(lo-u-du); ix <= Math.ceil(hi-u+du); ix++) {
          const [x,y] = point(ix+u,iy+v); ctx.beginPath(); ctx.arc(x,y,object.radius*scale,0,Math.PI*2); ctx.fill(); ctx.stroke();
        }
      }
      ctx.restore();
    } else {
      for (let iy = lo; iy < hi; iy++) for (let ix = lo; ix < hi; ix++) {
        const [x, y] = point(ix, iy);
        ctx.save(); ctx.transform(a[0] * scale, -a[1] * scale, b[0] * scale, -b[1] * scale, x, y);
        ctx.imageSmoothingEnabled = false; ctx.drawImage(raster, 0, 0, 1, 1); ctx.restore();
      }
    }
    ctx.lineWidth = .7; ctx.strokeStyle = theme.accent + '26'; ctx.beginPath();
    for (let n = lo; n <= hi; n++) {
      ctx.moveTo(...point(n, lo) as [number, number]); ctx.lineTo(...point(n, hi) as [number, number]);
      ctx.moveTo(...point(lo, n) as [number, number]); ctx.lineTo(...point(hi, n) as [number, number]);
    }
    ctx.stroke(); ctx.beginPath(); ctx.strokeStyle = theme.textPrimary; ctx.lineWidth = 1.5;
    [[0,0],[1,0],[1,1],[0,1]].forEach(([u,v], i) => { const [x,y] = point(u,v); if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y); });
    ctx.closePath(); ctx.stroke();
    ctx.fillStyle = theme.textPrimary; ctx.font = `11px ${uiFont}`; ctx.textAlign = 'left';
    const [labelX, labelY] = point(0, 1); ctx.fillText('unit cell', labelX + 5, labelY - 9);
    ctx.fillStyle = theme.textSecondary; ctx.font = `12px ${uiFont}`; ctx.textAlign = 'center';
    ctx.fillText(`ε ${minimum.toPrecision(3)}–${maximum.toPrecision(3)}  ·  ${vector ? 'circle geometry' : `${nx} × ${ny} samples / cell`}`, width / 2, height - 17);
  }, [preview, geometry, width, height, dpr, periods, mode]);
  const displayed = cursor && preview ? coordinates === 'fractional' ? toFractional(cursor, preview.lattice_vectors) : cursor : undefined;
  return <figure className="wb-preview" ref={ref}>
    <div className="wb-preview-toolbar">{geometry && <div className="wb-segmented" role="group" aria-label="Preview rendering">{['geometry', 'sampled'].map(value => <button key={value} aria-pressed={mode === value} onClick={() => setMode(value)}>{value === 'geometry' ? 'Geometry' : 'Dielectric grid'}</button>)}</div>}
      <div className="wb-segmented" role="group" aria-label="Preview periods">{[3, 5, 7].map(value => <button key={value} aria-pressed={periods === value} onClick={() => setPeriods(value)}>{value} × {value}</button>)}</div></div>
    <canvas ref={canvas} aria-label={`${geometry && mode === 'geometry' ? 'Circle geometry' : 'Dielectric distribution'} across ${periods} by ${periods} primitive cells`} role="img"
      onPointerMove={event => { const rect = event.currentTarget.getBoundingClientRect(), { x, y, scale } = transform.current; setCursor([(event.clientX - rect.left - x) / scale, (y - event.clientY + rect.top) / scale]); }} onPointerLeave={() => setCursor(undefined)} />
    {displayed && <output className="wb-coordinate-readout">{coordinates === 'fractional' ? 'u, v' : 'x, y'}: {displayed.map(value => value.toFixed(3)).join(', ')}</output>}
    {!preview && <p className="wb-placeholder">Loading dielectric geometry…</p>}
  </figure>;
}
