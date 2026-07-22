'use client';

/**
 * Band-structure plot for the studio: multi-series overlay with legend,
 * band-range filtering, hover readout, and clean image export.
 *
 * EXPORT CONTRACT: every visual attribute is an inline SVG attribute or
 * inline style (no CSS classes inside the <svg>), and interactive-only
 * elements live under a <g data-export="omit"> that the export utility
 * strips. Keep it that way or exported files will not match the screen.
 */

import React, { useMemo, useState } from 'react';
import type { BandResult } from '../../../lib/examples/bandResult';
import {
  PLOT_AXIS,
  PLOT_FONT,
  PLOT_GRID,
  PLOT_LABEL_TEXT,
  PLOT_SURFACE,
  PLOT_TICK_TEXT,
} from './palette';

export interface PlotSeries {
  id: string;
  label: string;
  color: string;
  result: BandResult;
  /** Streaming series: drawn slightly thicker. */
  live?: boolean;
}

export interface StudioBandPlotProps {
  series: PlotSeries[];
  width: number;
  height?: number;
  /** 1-based inclusive band range; null/undefined = all bands. */
  bandRange?: [number, number] | null;
  xMaxOverride?: number;
  title?: string;
  showLegend?: boolean;
  showGrid?: boolean;
  interactive?: boolean;
  svgRef?: React.Ref<SVGSVGElement>;
}

interface HoverState {
  x: number;
  y: number;
  seriesLabel: string;
  color: string;
  band: number;
  freq: number;
  kText: string;
}

const PAD_L = 56;
const PAD_R = 14;
const PAD_B = 34;
const LEGEND_ROW_H = 18;

export default function StudioBandPlot({
  series,
  width,
  height = 340,
  bandRange,
  xMaxOverride,
  title,
  showLegend = true,
  showGrid = true,
  interactive = true,
  svgRef,
}: StudioBandPlotProps) {
  const [hover, setHover] = useState<HoverState | null>(null);

  const valid = useMemo(
    () => series.filter((s) => s.result.freqs.length > 1),
    [series],
  );

  const padT = title ? 30 : 14;

  // Legend rows (below the plot area).
  const legendItems = showLegend && valid.length > 1 ? valid : [];
  const legendCols = Math.max(1, Math.floor((width - PAD_L - PAD_R) / 150));
  const legendRows = legendItems.length > 0 ? Math.ceil(legendItems.length / legendCols) : 0;
  const legendH = legendRows > 0 ? legendRows * LEGEND_ROW_H + 6 : 0;

  const plotW = width - PAD_L - PAD_R;
  const plotH = height - padT - PAD_B - legendH;

  const geom = useMemo(() => {
    if (valid.length === 0 || plotW <= 0 || plotH <= 0) return null;

    const bandsOf = (r: BandResult): [number, number] => {
      const lo = Math.max(1, bandRange?.[0] ?? 1);
      const hi = Math.min(r.n_bands, bandRange?.[1] ?? r.n_bands);
      return [lo, hi];
    };

    let yMax = 0;
    let xMax = 0;
    for (const s of valid) {
      const r = s.result;
      const [lo, hi] = bandsOf(r);
      for (const row of r.freqs) {
        for (let b = lo - 1; b < hi; b++) {
          const v = row[b];
          if (v !== undefined && v > yMax) yMax = v;
        }
      }
      const last = r.distances[r.distances.length - 1] ?? 0;
      if (last > xMax) xMax = last;
    }
    if (xMaxOverride && xMaxOverride > 0) xMax = Math.max(xMax, xMaxOverride);
    if (yMax <= 0) yMax = 1;
    yMax *= 1.05;
    if (xMax <= 0) xMax = 1;

    const sx = (d: number) => PAD_L + (d / xMax) * plotW;
    const sy = (f: number) => padT + plotH - (f / yMax) * plotH;

    // Band paths per series.
    const paths: { key: string; d: string; color: string; widthPx: number }[] = [];
    for (const s of valid) {
      const r = s.result;
      const [lo, hi] = bandsOf(r);
      for (let b = lo - 1; b < hi; b++) {
        let d = '';
        for (let k = 0; k < r.freqs.length; k++) {
          const v = r.freqs[k]?.[b];
          if (v === undefined) continue;
          const cmd = d === '' ? 'M' : 'L';
          d += `${cmd} ${sx(r.distances[k] ?? 0).toFixed(2)} ${sy(v).toFixed(2)} `;
        }
        if (d !== '') {
          paths.push({
            key: `${s.id}-b${b}`,
            d,
            color: s.color,
            widthPx: s.live ? 2 : 1.4,
          });
        }
      }
    }

    // Symmetry ticks from the first series that carries label distances.
    const labelled = valid.find(
      (s) => s.result.k_labels.length > 0 && s.result.k_label_distances.length > 0,
    )?.result;
    const ticks =
      labelled?.k_labels
        .map((label, i) => ({ label, x: sx(labelled.k_label_distances[i] ?? 0) }))
        .filter((t) => Number.isFinite(t.x)) ?? [];

    // Y gridlines.
    const yTicks = Array.from({ length: 5 }, (_, i) => {
      const f = (yMax / 5) * (i + 1);
      return { f, y: sy(f) };
    });

    return { sx, sy, xMax, yMax, paths, ticks, yTicks, bandsOf };
  }, [valid, plotW, plotH, padT, bandRange, xMaxOverride]);

  if (!geom) {
    return (
      <div
        style={{
          width: '100%',
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#6c6c6c',
          fontSize: 13,
          background: PLOT_SURFACE,
          border: `1px solid ${PLOT_GRID}`,
          borderRadius: 8,
        }}
      >
        Waiting for data…
      </div>
    );
  }

  const { sx, sy, paths, ticks, yTicks, bandsOf } = geom;

  const onPointerMove = (e: React.PointerEvent<SVGRectElement>) => {
    if (!interactive) return;
    const svg = (e.target as SVGRectElement).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const py = ((e.clientY - rect.top) / rect.height) * height;

    // Nearest sample across all series (nearest k by x, nearest band by y).
    let best: HoverState | null = null;
    let bestDist = Infinity;
    for (const s of valid) {
      const r = s.result;
      // nearest k index by x
      let kBest = 0;
      let kDist = Infinity;
      for (let k = 0; k < r.distances.length; k++) {
        const dx = Math.abs(sx(r.distances[k] ?? 0) - px);
        if (dx < kDist) {
          kDist = dx;
          kBest = k;
        }
      }
      const [lo, hi] = bandsOf(r);
      for (let b = lo - 1; b < hi; b++) {
        const v = r.freqs[kBest]?.[b];
        if (v === undefined) continue;
        const dy = Math.abs(sy(v) - py);
        const dist = kDist + dy;
        if (dist < bestDist) {
          bestDist = dist;
          best = {
            x: sx(r.distances[kBest] ?? 0),
            y: sy(v),
            seriesLabel: s.label,
            color: s.color,
            band: b + 1,
            freq: v,
            kText: `k ${kBest}/${r.n_kpoints - 1}`,
          };
        }
      }
    }
    setHover(bestDist < 60 ? best : null);
  };

  const readoutText = hover
    ? `${hover.seriesLabel} · band ${hover.band} · ω = ${hover.freq.toFixed(4)} · ${hover.kText}`
    : '';
  const readoutW = readoutText.length * 6.4 + 16;
  const readoutX = hover ? Math.min(Math.max(hover.x - readoutW / 2, PAD_L), width - PAD_R - readoutW) : 0;
  const readoutAbove = hover ? hover.y > padT + 40 : false;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      role="img"
      aria-label={title ?? 'Band diagram'}
      style={{ maxWidth: '100%', display: 'block', background: PLOT_SURFACE, borderRadius: 8 }}
    >
      {/* frame */}
      <rect
        x={PAD_L}
        y={padT}
        width={plotW}
        height={plotH}
        fill="none"
        stroke={PLOT_AXIS}
        strokeWidth={1}
      />

      {title ? (
        <text x={PAD_L} y={19} fontSize={12.5} fontWeight={600} fill={PLOT_LABEL_TEXT} fontFamily={PLOT_FONT}>
          {title}
        </text>
      ) : null}

      {/* y grid + labels */}
      {yTicks.map((t) => (
        <g key={`y-${t.f}`}>
          {showGrid ? (
            <line x1={PAD_L} y1={t.y} x2={PAD_L + plotW} y2={t.y} stroke={PLOT_GRID} strokeWidth={1} />
          ) : null}
          <text
            x={PAD_L - 8}
            y={t.y + 3.5}
            fontSize={10.5}
            fill={PLOT_TICK_TEXT}
            textAnchor="end"
            fontFamily={PLOT_FONT}
          >
            {t.f.toFixed(2)}
          </text>
        </g>
      ))}
      <text
        x={PAD_L - 8}
        y={padT + plotH + 3.5}
        fontSize={10.5}
        fill={PLOT_TICK_TEXT}
        textAnchor="end"
        fontFamily={PLOT_FONT}
      >
        0.00
      </text>

      {/* symmetry ticks */}
      {ticks.map((t, i) => (
        <g key={`t-${i}`}>
          <line x1={t.x} y1={padT} x2={t.x} y2={padT + plotH} stroke={PLOT_GRID} strokeWidth={1} />
          <text
            x={t.x}
            y={padT + plotH + 16}
            fontSize={11.5}
            fill={PLOT_LABEL_TEXT}
            textAnchor="middle"
            fontFamily={PLOT_FONT}
          >
            {t.label}
          </text>
        </g>
      ))}

      {/* y-axis label */}
      <text
        x={14}
        y={padT + plotH / 2}
        fontSize={11}
        fill={PLOT_TICK_TEXT}
        textAnchor="middle"
        fontFamily={PLOT_FONT}
        transform={`rotate(-90 14 ${padT + plotH / 2})`}
      >
        Frequency (c/a)
      </text>

      {/* bands */}
      {paths.map((p) => (
        <path key={p.key} d={p.d} fill="none" stroke={p.color} strokeWidth={p.widthPx} opacity={0.92} />
      ))}

      {/* legend */}
      {legendItems.map((s, i) => {
        const col = i % legendCols;
        const row = Math.floor(i / legendCols);
        const lx = PAD_L + col * 150;
        const ly = padT + plotH + 28 + row * LEGEND_ROW_H;
        return (
          <g key={`lg-${s.id}`}>
            <line x1={lx} y1={ly} x2={lx + 16} y2={ly} stroke={s.color} strokeWidth={2.5} />
            <text x={lx + 22} y={ly + 3.5} fontSize={10.5} fill={PLOT_TICK_TEXT} fontFamily={PLOT_FONT}>
              {s.label.length > 22 ? `${s.label.slice(0, 21)}…` : s.label}
            </text>
          </g>
        );
      })}

      {/* hover crosshair + readout (stripped from exports) */}
      {interactive ? (
        <g data-export="omit">
          {hover ? (
            <>
              <line
                x1={hover.x}
                y1={padT}
                x2={hover.x}
                y2={padT + plotH}
                stroke="rgba(255,255,255,0.25)"
                strokeWidth={1}
              />
              <circle cx={hover.x} cy={hover.y} r={4} fill="none" stroke={hover.color} strokeWidth={2} />
              <g transform={`translate(${readoutX}, ${readoutAbove ? hover.y - 30 : hover.y + 12})`}>
                <rect width={readoutW} height={19} rx={4} fill="rgba(16,16,16,0.95)" stroke={PLOT_AXIS} />
                <text x={8} y={13} fontSize={10.5} fill={PLOT_LABEL_TEXT} fontFamily={PLOT_FONT}>
                  {readoutText}
                </text>
              </g>
            </>
          ) : null}
          <rect
            x={PAD_L}
            y={padT}
            width={plotW}
            height={plotH}
            fill="transparent"
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
          />
        </g>
      ) : null}
    </svg>
  );
}
