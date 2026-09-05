'use client';
import { useId, useMemo, useRef, useState } from 'react';
import { useSize } from './useSize';
import { download } from '../../lib/compute/export';

export interface BandPlotData { distances: ArrayLike<number>; frequencies: ArrayLike<number>; bands: number; labels: string[]; labelIndices: number[] }
const colors = ['#84d4bd', '#75a9ee', '#d4b37a', '#bd98db', '#e38e91', '#81bed1'];
export function BandPlot({ data, title = 'Band structure' }: { data: BandPlotData; title?: string }) {
  const { ref, width } = useSize<HTMLDivElement>();
  const svg = useRef<SVGSVGElement>(null), clip = useId().replace(/:/g, '');
  const [point, setPoint] = useState(0), [band, setBand] = useState(0);
  const w = Math.max(width, 280), h = Math.max(280, Math.min(440, w * 0.55));
  const left = 52, right = w - 22, top = 26, bottom = h - 48;
  const maximumX = data.distances[data.distances.length - 1] || 1;
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
  const selected = Math.min(point, Math.max(0, data.distances.length - 1));
  return <div className="wb-plot" ref={ref}>
    <div className="wb-row"><h3>{title}</h3><button onClick={() => {
      if (svg.current) { const copy = svg.current.cloneNode(true) as SVGSVGElement; copy.querySelectorAll('[data-interactive]').forEach(node => node.remove());
        download(new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' }), 'bands.svg'); }
    }}>Export plot</button></div>
    <svg ref={svg} xmlns="http://www.w3.org/2000/svg" width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${title}. Reduced frequency along the reciprocal-space path.`}
      onPointerMove={event => { const bounds = event.currentTarget.getBoundingClientRect(); const position = (event.clientX - bounds.left) * w / bounds.width;
        const distance = (position - left) / (right - left) * maximumX;
        let low = 0, high = data.distances.length - 1;
        while (low < high) { const middle = Math.floor((low + high) / 2); if (data.distances[middle] < distance) low = middle + 1; else high = middle; }
        setPoint(low > 0 && Math.abs(data.distances[low - 1] - distance) < Math.abs(data.distances[low] - distance) ? low - 1 : low); }}>
      <rect width={w} height={h} fill="#101616" />
      <defs><clipPath id={clip}><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
      {Array.from({ length: 5 }, (_, i) => maximumY * i / 4).map(value => <g key={value}><line x1={left} x2={right} y1={y(value)} y2={y(value)} stroke="#2d3736" />
        <text x={left - 8} y={y(value) + 4} fill="#bccbc7" textAnchor="end" fontFamily="system-ui" fontSize={11}>{value.toFixed(2)}</text></g>)}
      {data.labels.map((label, i) => { const distance = data.distances[data.labelIndices[i]]; return distance === undefined ? null : <g key={i}>
        <line x1={x(distance)} x2={x(distance)} y1={top} y2={bottom} stroke="#3a4743" />
        <text x={x(distance)} y={bottom + 23} fill="#d6e2dd" textAnchor="middle" fontFamily="system-ui" fontSize={13}>{label}</text></g>; })}
      <text x={left} y={16} fill="#d6e2dd" fontFamily="system-ui" fontSize={12}>Frequency (c/reference length)</text>
      <g clipPath={`url(#${clip})`}>{Array.from({ length: data.bands }, (_, band) => <polyline key={band} fill="none" stroke={colors[band % colors.length]} strokeWidth={1.6}
        points={plotted.map(index => `${x(data.distances[index])},${y(data.frequencies[index * data.bands + band])}`).join(' ')} />)}</g>
      {data.distances.length > 0 && <line data-interactive x1={x(data.distances[selected])} x2={x(data.distances[selected])} y1={top} y2={bottom} stroke="#e2ece8" strokeDasharray="3 5" />}
    </svg>
    {plotted.length < data.distances.length && <p className="wb-muted">Plot overview uses {plotted.length} samples. The readout and exported data retain every point.</p>}
    <div className="wb-plot-readout">
      <label>Point <input aria-label="Selected k-point" type="range" min={0} max={Math.max(0, data.distances.length - 1)} value={selected} onChange={event => setPoint(Number(event.target.value))} /></label>
      <label>Band <select aria-label="Readout band" value={band} onChange={event => setBand(Number(event.target.value))}>{Array.from({ length: data.bands }, (_, index) => <option key={index} value={index}>{index}</option>)}</select></label>
      <output>k[{selected}] · f = {data.frequencies[selected * data.bands + Math.min(band, data.bands - 1)]?.toPrecision(6) ?? 'pending'}</output>
    </div>
  </div>;
}
