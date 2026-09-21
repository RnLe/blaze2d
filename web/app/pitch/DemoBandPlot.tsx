'use client';
import { useId, useRef } from 'react';
import { Download } from 'lucide-react';
import { useSize } from '@/components/workbench/useSize';
import { useUiScale } from '@/lib/use-ui-scale';
import { theme } from '@/lib/theme';
import { downloadBlob } from '@/lib/util/download';
import { BandTooltip, nearestBandPoint, useBandHover } from '@/components/workbench/BandTooltip';
import type { DemoPath, DemoSeries } from './demo-model';

export default function DemoBandPlot({ path, series, title, maximumY, pending }: {
  path?: DemoPath; series: DemoSeries[]; title: string; maximumY: number; pending: boolean;
}) {
  const { ref, width, height } = useSize<HTMLDivElement>();
  const svg = useRef<SVGSVGElement>(null), clip = useId().replace(/:/g, '');
  const scale = useUiScale(), w = width ? width / scale : 600, h = height ? height / scale : 400;
  const compact = w < 300, left = compact ? 35 : 54, right = w - 14, top = 31, bottom = h - 40;
  // Use the complete native path before the first sample and throughout both jobs.
  const maximumX = path?.distances.at(-1) || 1;
  const x = (distance: number) => left + distance / maximumX * (right - left);
  const y = (frequency: number) => bottom - frequency / maximumY * (bottom - top);
  const { cursor, handlers } = useBandHover({ left, right, top, bottom });
  const nearest = cursor ? nearestBandPoint(series, cursor, x, y) : undefined;
  return <figure className="pitch-demo-plot">
    <div className="pitch-demo-plot-heading"><h3>{title}</h3><button type="button" aria-label={`Export ${title}`} disabled={!series.length} onClick={() => {
      if (svg.current) {
        const copy = svg.current.cloneNode(true) as SVGSVGElement;
        copy.querySelectorAll('[data-interactive]').forEach(node => node.remove());
        downloadBlob(new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' }), `${title.toLowerCase().replaceAll(' ', '-')}.svg`);
      }
    }}><Download size={14} aria-hidden="true" /><span>SVG</span></button></div>
    <div className="pitch-demo-plot-stage" ref={ref}>
      <svg ref={svg} xmlns="http://www.w3.org/2000/svg" width={width || '100%'} height={height || '100%'} viewBox={`0 0 ${w} ${h}`}
        role="img" aria-label={`${title}. Frequency in units of c/a along the crystal's high-symmetry path.`} data-x-max={maximumX} {...handlers}>
        <rect width={w} height={h} fill={theme.surface} />
        <defs><clipPath id={clip}><rect x={left} y={top} width={Math.max(0, right - left)} height={Math.max(0, bottom - top)} /></clipPath></defs>
        {[0, 1, 2, 3, 4].map(index => {
          const value = maximumY * index / 4;
          return <g key={index}><line x1={left} x2={right} y1={y(value)} y2={y(value)} stroke={theme.gridLine} />
            <text x={left - 7} y={y(value) + 4} textAnchor="end" fill={theme.textSecondary} fontSize={compact ? 10 : 12}>{value.toFixed(2)}</text></g>;
        })}
        {path?.k_labels.map((label, index) => {
          const distance = path.distances[path.k_label_indices[index]];
          return <g key={index} data-k-label={label} data-distance={distance}>
            <line x1={x(distance)} x2={x(distance)} y1={top} y2={bottom} stroke={theme.axisLine} />
            <text x={x(distance)} y={bottom + 24} textAnchor="middle" fill={theme.textPrimary} fontSize={compact ? 12 : 14}>{label}</text>
          </g>;
        })}
        <text x={left} y="17" fill={theme.textPrimary} fontSize={compact ? 11 : 13}>ωa / (2πc)</text>
        <g clipPath={`url(#${clip})`}>{series.map(data => <g key={data.polarization} data-polarization={data.polarization} data-samples={data.distances.length}>
          {Array.from({ length: data.bands }, (_, band) => <polyline key={band} fill="none"
            stroke={data.polarization === 'TM' ? theme.brandBlue : theme.brandOrange} strokeWidth="1.7" strokeLinejoin="round"
            points={Array.from({ length: data.distances.length }, (_, index) => `${x(data.distances[index])},${y(data.frequencies[index * data.bands + band])}`).join(' ')} />)}
        </g>)}</g>
        {nearest && cursor && <BandTooltip point={nearest} cursor={cursor} width={w} height={h} />}
      </svg>
      {!series.length && <p className="pitch-demo-plot-placeholder">{pending ? 'Waiting for band data…' : 'Calculate to draw the bands.'}</p>}
    </div>
  </figure>;
}
