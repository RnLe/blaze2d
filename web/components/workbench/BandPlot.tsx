'use client';
import { useId, useMemo, useRef } from 'react';
import { useSize } from './useSize';
import { downloadBlob } from '@/lib/util/download';
import { bandColors, theme } from '@/lib/theme';
import { useUiScale } from '@/lib/use-ui-scale';
import { BandTooltip, nearestBandPoint, useBandHover } from './BandTooltip';

export interface BandPlotData { distances: ArrayLike<number>; frequencies: ArrayLike<number>; bands: number; labels: string[]; labelIndices: number[]; pathDistances?: ArrayLike<number>; polarization: string }
export function BandPlot({ data, title = 'Band structure' }: { data: BandPlotData; title?: string }) {
  const { ref, width } = useSize<HTMLDivElement>();
  const svg = useRef<SVGSVGElement>(null), clip = useId().replace(/:/g, '');
  // Laid out in design units and scaled to the display at the <svg>, so the
  // plot keeps the same proportions as the interface around it.
  const scale = Math.max(1, useUiScale());
  const w = Math.max(width / scale, 280), h = Math.max(280, Math.min(440, w * 0.55));
  const left = 52, right = w - 22, top = 26, bottom = h - 48;
  // Streaming samples grow, but the planned reciprocal-space path stays fixed.
  const path = data.pathDistances ?? data.distances;
  const maximumX = path[path.length - 1] || 1;
  const maximumY = useMemo(() => {
    let maximum = 0;
    for (let i = 0; i < data.frequencies.length; i++) if (Number.isFinite(data.frequencies[i])) maximum = Math.max(maximum, data.frequencies[i]);
    return maximum * 1.06 || 1;
  }, [data.frequencies]);
  const plotted = useMemo(() => {
    const stride = Math.max(1, Math.ceil(data.distances.length / 5000));
    const indices = new Set(data.labelIndices.filter(index => index >= 0 && index < data.distances.length));
    for (let i = 0; i < data.distances.length; i += stride) indices.add(i);
    if (data.distances.length) indices.add(data.distances.length - 1);
    return [...indices].sort((a, b) => a - b);
  }, [data.distances, data.labelIndices]);
  const x = (value: number) => left + value / maximumX * (right - left);
  const y = (value: number) => bottom - value / maximumY * (bottom - top);
  const { cursor, handlers } = useBandHover({ left, right, top, bottom });
  const nearest = cursor ? nearestBandPoint([data], cursor, x, y) : undefined;
  return <div className="wb-plot" ref={ref}>
    <div className="wb-row"><h3>{title}</h3><button onClick={() => {
      if (svg.current) { const copy = svg.current.cloneNode(true) as SVGSVGElement; copy.querySelectorAll('[data-interactive]').forEach(node => node.remove());
        downloadBlob(new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' }), 'bands.svg'); }
    }}>Export plot</button></div>
    <svg ref={svg} xmlns="http://www.w3.org/2000/svg" width={w * scale} height={h * scale} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${title}. Reduced frequency along the reciprocal-space path.`} data-x-max={maximumX} data-samples={data.distances.length}
      {...handlers}>
      <rect width={w} height={h} fill={theme.surface} />
      <defs><clipPath id={clip}><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
      {Array.from({ length: 5 }, (_, i) => maximumY * i / 4).map(value => <g key={value}><line x1={left} x2={right} y1={y(value)} y2={y(value)} stroke={theme.gridLine} />
        <text x={left - 8} y={y(value) + 4} fill={theme.textSecondary} textAnchor="end" fontFamily="inherit" fontSize={11}>{value.toFixed(2)}</text></g>)}
      {data.labels.map((label, i) => { const distance = path[data.labelIndices[i]]; return distance === undefined ? null : <g key={i}>
        <line x1={x(distance)} x2={x(distance)} y1={top} y2={bottom} stroke={theme.axisLine} />
        <text x={x(distance)} y={bottom + 23} fill={theme.textPrimary} textAnchor="middle" fontFamily="inherit" fontSize={13}>{label}</text></g>; })}
      <text x={left} y={16} fill={theme.textPrimary} fontFamily="inherit" fontSize={12}>Frequency (c/reference length)</text>
      <g clipPath={`url(#${clip})`}>{Array.from({ length: data.bands }, (_, band) => <polyline key={band} fill="none" stroke={bandColors[band % bandColors.length]} strokeWidth={1.6}
        points={plotted.map(index => `${x(data.distances[index])},${y(data.frequencies[index * data.bands + band])}`).join(' ')} />)}</g>
      {nearest && cursor && <BandTooltip point={nearest} cursor={cursor} width={w} height={h} />}
    </svg>
    {plotted.length < data.distances.length && <p className="wb-muted">Plot overview uses {plotted.length} samples. Tooltips and exported data retain every point.</p>}
  </div>;
}
