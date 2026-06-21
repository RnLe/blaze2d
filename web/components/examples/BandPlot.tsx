'use client';

import { useMemo } from 'react';
import type { BandResult } from '../../lib/examples/bandResult';

export interface BandPlotProps {
  /** One or more results to overlay (e.g. TM vs TE, or a sweep). */
  results: BandResult[];
  width?: number;
  height?: number;
  /** Pin the x-axis maximum (in distance units). Useful for streaming
   *  examples so the axis doesn't grow as more k-points arrive. */
  xMaxOverride?: number;
}

const SERIES_COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb7185'];

/**
 * Lightweight SVG band-diagram plot. Renders one or more `BandResult`s,
 * overlaying their bands and drawing high-symmetry tick labels.
 */
export default function BandPlot({ results, width = 560, height = 300, xMaxOverride }: BandPlotProps) {
  const valid = results.filter((r) => r && r.freqs.length > 0);

  const { paths, xTicks, yMax, padL, padR, padT, padB, plotW, plotH } = useMemo(() => {
    const padL = 56;
    const padR = 16;
    const padT = 16;
    const padB = 36;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    let xMax = 0;
    let yMax = 0;
    for (const r of valid) {
      const d = r.distances;
      if (d.length > 0) xMax = Math.max(xMax, d[d.length - 1]);
      for (const row of r.freqs) for (const v of row) if (v > yMax) yMax = v;
    }
    if (xMaxOverride && xMaxOverride > 0) xMax = xMaxOverride;
    if (xMax === 0) xMax = 1;
    yMax = yMax > 0 ? yMax * 1.05 : 1;

    const xScale = (x: number) => padL + (x / xMax) * plotW;
    const yScale = (y: number) => padT + plotH - (y / yMax) * plotH;

    const paths: { d: string; color: string }[] = [];
    valid.forEach((r, si) => {
      const color = SERIES_COLORS[si % SERIES_COLORS.length];
      const nBands = r.n_bands;
      for (let b = 0; b < nBands; b++) {
        let d = '';
        for (let k = 0; k < r.freqs.length; k++) {
          const x = xScale(r.distances[k]);
          const y = yScale(r.freqs[k][b]);
          d += `${k === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)} `;
        }
        paths.push({ d, color });
      }
    });

    // Use the first result with labels for the x-axis ticks.
    const labelled = valid.find((r) => r.k_labels.length > 0);
    const xTicks =
      labelled?.k_labels.map((label, i) => ({
        label,
        x: xScale(labelled.k_label_distances[i] ?? 0),
      })) ?? [];

    return { paths, xTicks, yMax, padL, padR, padT, padB, plotW, plotH };
  }, [valid, width, height, xMaxOverride]);

  if (valid.length === 0) {
    return (
      <div
        style={{
          width: '100%',
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#6b7280',
          border: '1px solid #1f2937',
          borderRadius: '10px',
          fontSize: '0.85rem',
        }}
      >
        Run the example to see the band diagram.
      </div>
    );
  }

  const yTickCount = 5;
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => {
    const val = (yMax * i) / yTickCount;
    return { val, y: padT + plotH - (val / yMax) * plotH };
  });

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      style={{
        background: '#0a0a0a',
        border: '1px solid #1f2937',
        borderRadius: '10px',
        maxWidth: '100%',
        display: 'block',
      }}
      role="img"
      aria-label="Band diagram"
    >
      {/* Y grid + labels */}
      {yTicks.map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={width - padR}
            y1={t.y}
            y2={t.y}
            stroke="#1f2937"
            strokeWidth={0.5}
          />
          <text x={padL - 8} y={t.y + 3} fontSize={10} fill="#9ca3af" textAnchor="end">
            {t.val.toFixed(2)}
          </text>
        </g>
      ))}

      {/* X high-symmetry verticals + labels */}
      {xTicks.map((t, i) => (
        <g key={i}>
          <line x1={t.x} x2={t.x} y1={padT} y2={padT + plotH} stroke="#374151" strokeWidth={0.7} />
          <text x={t.x} y={height - padB + 18} fontSize={12} fill="#d1d5db" textAnchor="middle">
            {t.label}
          </text>
        </g>
      ))}

      {/* Axis label */}
      <text
        x={14}
        y={padT + plotH / 2}
        fontSize={11}
        fill="#9ca3af"
        textAnchor="middle"
        transform={`rotate(-90 14 ${padT + plotH / 2})`}
      >
        Frequency (c/a)
      </text>

      {/* Bands */}
      {paths.map((p, i) => (
        <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth={1.4} opacity={0.9} />
      ))}
    </svg>
  );
}
