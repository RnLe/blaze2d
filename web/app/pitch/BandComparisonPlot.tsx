'use client';
import { useEffect, useId, useState } from 'react';
import Link from 'next/link';
import { theme } from '@/lib/theme';
import { useUiScale } from '@/lib/use-ui-scale';
import { getAssetPath } from '@/lib/paths';
import { useSize } from '@/components/workbench/useSize';
import { useSequence, phase, smooth } from './useSequence';

type Point = { k_distance: number; frequencies: number[] };
type Bands = { mpb: Point[]; blaze_f64: Point[]; blaze_f32: Point[] };
type Data = { parameters: { resolution: number; num_bands: number; computed_bands: number; k_points_per_segment: number; mpb_tolerance: number; blaze_f32_tolerance: number }; TM: Bands; TE: Bands };

/** The polarizations, in the site's two brand colours. */
const POLARIZATIONS = [
  { key: 'TM' as const, base: theme.brandBlue, light: theme.brandBlueLight },
  { key: 'TE' as const, base: theme.brandOrange, light: theme.brandOrangeLight },
];

/**
 * One panel per solver, each drawing its own TM then TE, and then the two
 * panels come together. Splitting by solver is what makes the overlay mean
 * something: each panel is a whole, independent calculation, and the claim is
 * that they land on each other.
 *
 * Blaze is the dotted, lighter pass, drawn second so it sits over the reference.
 */
const SOLVERS = [
  { key: 'mpb' as const, label: 'MPB', dotted: false },
  { key: 'blaze_f32' as const, label: 'Blaze', dotted: true },
];
/**
 * When each polarization starts drawing: sequential within a panel, identical
 * between them. Both solvers lay down their TM together and then their TE
 * together, so the two panels are read side by side as one comparison rather
 * than one after the other.
 */
const POLARIZATION_STARTS = [200, 1100];
/** Short dots with a wide gap: at a band's curvature a tighter dash reads solid. */
const DOTS = '1.5 5';
const DRAW_MS = 1150;
const MERGE_AT = 2700;
const MERGE_MS = 1400;

export default function BandComparisonPlot() {
  const [data, setData] = useState<Data>(), [error, setError] = useState('');
  const { ref: sizeRef, width } = useSize<HTMLDivElement>();
  const { ref, elapsed } = useSequence(4900, !!data);
  const id = useId().replace(/:/g, '');
  useEffect(() => {
    const controller = new AbortController();
    void fetch(getAssetPath('/data/benchmarks/series6-accuracy.json'), { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('Band comparison data is unavailable.'); return response.json();
    }).then(setData).catch(error => { if (!controller.signal.aborted) setError(error.message); });
    return () => controller.abort();
  }, []);
  const scale = useUiScale();
  const w = Math.max(270, width / scale), mobile = w < 640, merge = smooth(phase(elapsed, MERGE_AT, MERGE_MS));
  const gap = 30;
  // Fixed for the whole sequence. Growing the panel as the two come together
  // would rescale the axes and stretch the curves, so only the position moves.
  const panelW = mobile ? w : (w - gap) / 2, panelH = mobile ? 265 : 380;
  const centreX = (w - panelW) / 2;
  const totalH = mobile ? panelH + (panelH + gap) * (1 - merge) : panelH;

  const maxX = Math.max(...POLARIZATIONS.map(p => data?.[p.key].mpb.at(-1)?.k_distance ?? 1));
  const maxY = Math.ceil(Math.max(1, ...POLARIZATIONS.flatMap(p =>
    [...(data?.[p.key].mpb ?? []), ...(data?.[p.key].blaze_f32 ?? [])].flatMap(point => point.frequencies))) * 10) / 10;

  function panel(solver: typeof SOLVERS[number], index: number) {
    const xOffset = mobile ? 0 : (index * (panelW + gap)) * (1 - merge) + centreX * merge;
    const yOffset = mobile ? index * (panelH + gap) * (1 - merge) : 0;
    const left = 58, right = panelW - 16, top = 46, bottom = panelH - 52;
    const x = (distance: number) => left + distance / maxX * (right - left);
    const y = (frequency: number) => bottom - frequency / maxY * (bottom - top);
    // Both panels carry axes while they stand apart; the second set goes as
    // they converge so the overlaid plot is not drawn twice over itself.
    const axesOpacity = index === 0 ? 1 : 1 - merge;
    return <g key={solver.key} transform={`translate(${xOffset},${yOffset})`} data-solver={solver.label}>
      <defs>
        {POLARIZATIONS.map((polarization, order) => {
          const progress = smooth(phase(elapsed, POLARIZATION_STARTS[order], DRAW_MS));
          return <clipPath key={polarization.key} id={`${id}-${index}-${polarization.key}`}>
            <rect x={left} y={top - 1} width={(right - left) * progress} height={bottom - top + 2} />
          </clipPath>;
        })}
      </defs>
      <g opacity={axesOpacity}>
        {Array.from({ length: 5 }, (_, i) => <g key={i}>
          <line x1={left} x2={right} y1={y(maxY * i / 4)} y2={y(maxY * i / 4)} stroke="var(--series-grid)" />
          <text x={left - 10} y={y(maxY * i / 4) + 6} textAnchor="end" fill={theme.textPrimary} fontSize={16}>{(maxY * i / 4).toFixed(2)}</text>
        </g>)}
        {['Γ', 'X', 'M', 'Γ'].map((label, i) => {
          const reference = data?.TM.mpb ?? [];
          const distance = reference[Math.min(i * (data?.parameters.k_points_per_segment ?? 15), Math.max(0, reference.length - 1))]?.k_distance ?? 0;
          return <g key={i}>
            <line x1={x(distance)} x2={x(distance)} y1={top} y2={bottom} stroke="var(--series-grid)" />
            <text x={x(distance)} y={bottom + 30} textAnchor="middle" fill={theme.textPrimary} fontSize={18}>{label}</text>
          </g>;
        })}
        <text x={left} y={22} fill={theme.textPrimary} fontSize={16}>Frequency (c/a)</text>
      </g>
      <text x={right} y={22} textAnchor="end" fill={theme.textPrimary} fontSize={18} opacity={1 - merge}>{solver.label}</text>
      {POLARIZATIONS.map(polarization => {
        const points = data?.[polarization.key][solver.key] ?? [];
        return <g key={polarization.key} clipPath={`url(#${id}-${index}-${polarization.key})`}
                  fill="none" stroke={solver.dotted ? polarization.light : polarization.base}
                  strokeWidth={solver.dotted ? 2.4 : 2.2} strokeDasharray={solver.dotted ? DOTS : undefined}
                  strokeLinecap={solver.dotted ? 'round' : undefined} opacity={solver.dotted ? 1 : 0.85}>
          {Array.from({ length: data?.parameters.num_bands ?? 0 }, (_, band) =>
            <polyline key={band} points={points.map(point => `${x(point.k_distance).toFixed(3)},${y(point.frequencies[band]).toFixed(3)}`).join(' ')} />)}
        </g>;
      })}
    </g>;
  }

  return <div className="pitch-comparison" ref={ref}>
    <div ref={sizeRef} className="pitch-band-stage">
      {error ? <p role="alert">{error}</p> : !data ? <p role="status">Loading recorded band data…</p> :
        <svg width={w * scale} height={totalH * scale} viewBox={`0 0 ${w} ${totalH}`} role="img"
             aria-label="Independent MPB and mixed-precision Blaze band calculations, each drawing its TM and TE polarizations before the two are overlaid.">
          {SOLVERS.map(panel)}
        </svg>}
    </div>
    {/* A drawn sample rather than a box-drawing character, so the legend shows
        the same dotting the plot does instead of approximating it. */}
    <div className="pitch-legend">
      {SOLVERS.map(solver => <span key={solver.key}>
        {POLARIZATIONS.map(polarization => <b key={polarization.key} style={{ color: solver.dotted ? polarization.light : polarization.base }}>
          <svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true">
            <line x1="1" y1="5" x2="29" y2="5" stroke="currentColor" strokeWidth="2.4"
                  strokeDasharray={solver.dotted ? DOTS : undefined} strokeLinecap={solver.dotted ? 'round' : undefined} />
          </svg>
          {polarization.key} {solver.label}
        </b>)}
      </span>)}
      <span className="pitch-comparison-state">{merge === 1 ? 'Independent calculations, overlaid' : 'Drawing recorded band structures'}</span>
    </div>
    {data && <p className="pitch-caption">Square lattice · {data.parameters.resolution} × {data.parameters.resolution} grid · {data.parameters.num_bands} of {data.parameters.computed_bands} bands, both polarizations · Blaze in mixed precision · <Link href="/blaze#accuracy-validation">Recorded comparison ↗</Link></p>}
  </div>;
}
