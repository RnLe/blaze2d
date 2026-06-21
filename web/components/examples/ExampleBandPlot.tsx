'use client';

import styles from './Examples.module.css';
import type { KPointResult, MaxwellResult } from './types';

interface ExampleBandPlotProps {
  results: MaxwellResult[];
  kPoints: KPointResult[];
}

const COLORS = ['#57c7b1', '#8aa8ff', '#ffb35f', '#f2738c', '#d4d966'];
const WIDTH = 720;
const HEIGHT = 260;
const PAD = { left: 44, right: 14, top: 14, bottom: 34 };

function partialResult(kPoints: KPointResult[]): MaxwellResult | null {
  if (kPoints.length === 0) return null;
  const first = kPoints[0];
  return {
    result_type: 'maxwell',
    job_index: first.job_index,
    params: first.params,
    sweep_values: first.sweep_values ?? {},
    sweep_order: first.sweep_order ?? '',
    k_path: kPoints.map((point) => point.k_point),
    distances: kPoints.map((point) => point.distance),
    bands: kPoints.map((point) => point.omegas),
    num_k_points: kPoints.length,
    num_bands: first.num_bands,
  };
}

function pathForBand(result: MaxwellResult, bandIndex: number, xMax: number, yMax: number) {
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const points = result.bands
    .map((bandsAtK, kIndex) => {
      const frequency = bandsAtK[bandIndex];
      if (frequency === undefined) return null;
      const x = PAD.left + ((result.distances[kIndex] ?? 0) / xMax) * innerW;
      const y = PAD.top + innerH - (frequency / yMax) * innerH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .filter(Boolean);

  return points.length > 0 ? `M ${points.join(' L ')}` : '';
}

function resultLabel(result: MaxwellResult, index: number) {
  const sweep = result.sweep_order ? ` ${result.sweep_order}` : '';
  return `${result.params?.polarization ?? `job ${index + 1}`}${sweep}`;
}

export default function ExampleBandPlot({ results, kPoints }: ExampleBandPlotProps) {
  const partial = partialResult(kPoints);
  const plotted = results.length > 0 ? results : partial ? [partial] : [];
  if (plotted.length === 0) return null;

  const xMax = Math.max(1e-9, ...plotted.flatMap((result) => result.distances));
  const yMax = Math.max(0.1, ...plotted.flatMap((result) => result.bands.flat())) * 1.12;
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const x0 = PAD.left;
  const y0 = PAD.top + innerH;

  return (
    <div className={styles.plotShell}>
      <div className={styles.plotTitle}>
        <span>Band diagram</span>
        <span>
          {plotted.length} result{plotted.length === 1 ? '' : 's'}
        </span>
      </div>
      <svg className={styles.plotSvg} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Computed band diagram">
        <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} fill="rgba(255,255,255,0.025)" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = PAD.top + innerH - tick * innerH;
          return (
            <g key={tick}>
              <line x1={PAD.left} x2={PAD.left + innerW} y1={y} y2={y} stroke="rgba(255,255,255,0.07)" />
              <text x={PAD.left - 8} y={y + 4} textAnchor="end" fill="rgba(255,255,255,0.5)" fontSize="11">
                {(tick * yMax).toFixed(2)}
              </text>
            </g>
          );
        })}
        <line x1={x0} y1={PAD.top} x2={x0} y2={y0} stroke="rgba(255,255,255,0.35)" />
        <line x1={x0} y1={y0} x2={PAD.left + innerW} y2={y0} stroke="rgba(255,255,255,0.35)" />
        <text x={18} y={PAD.top + innerH / 2} fill="rgba(255,255,255,0.58)" fontSize="12" transform={`rotate(-90 18 ${PAD.top + innerH / 2})`} textAnchor="middle">
          omega a / 2 pi c
        </text>
        <text x={PAD.left + innerW / 2} y={HEIGHT - 8} fill="rgba(255,255,255,0.58)" fontSize="12" textAnchor="middle">
          k-path distance
        </text>
        {plotted.map((result, resultIndex) =>
          Array.from({ length: result.num_bands }).map((_, bandIndex) => (
            <path
              key={`${result.job_index}-${resultIndex}-${bandIndex}`}
              d={pathForBand(result, bandIndex, xMax, yMax)}
              fill="none"
              stroke={COLORS[resultIndex % COLORS.length]}
              strokeOpacity={0.92 - bandIndex * 0.09}
              strokeWidth={resultIndex === 0 ? 1.8 : 1.35}
            />
          ))
        )}
        {plotted.slice(0, 4).map((result, index) => (
          <g key={`legend-${index}`} transform={`translate(${PAD.left + 8}, ${PAD.top + 16 + index * 18})`}>
            <line x1="0" x2="20" y1="0" y2="0" stroke={COLORS[index % COLORS.length]} strokeWidth="3" />
            <text x="27" y="4" fill="rgba(255,255,255,0.72)" fontSize="12">
              {resultLabel(result, index)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
