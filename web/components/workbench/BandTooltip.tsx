'use client';
import { useState, type PointerEvent } from 'react';
import { theme } from '@/lib/theme';

type Position = { x: number; y: number };
export type HoverSeries = { distances: ArrayLike<number>; frequencies: ArrayLike<number>; bands: number; polarization: string };
export type BandPoint = Position & { sample: number; band: number; polarization: string; frequency: number };

/** Find the closest computed point in screen space, across all bands and polarizations. */
export function nearestBandPoint(series: HoverSeries[], cursor: Position, x: (distance: number) => number, y: (frequency: number) => number): BandPoint | undefined {
  let nearest: BandPoint | undefined, best = Infinity;
  for (const data of series) {
    let low = 0, high = data.distances.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (x(data.distances[middle]) < cursor.x) low = middle + 1; else high = middle;
    }
    // Search outwards from the cursor's k position. Once the horizontal distance
    // exceeds the best total distance, no further sample on that side can win.
    for (const direction of [-1, 1]) {
      for (let sample = direction < 0 ? low - 1 : low; sample >= 0 && sample < data.distances.length; sample += direction) {
        const px = x(data.distances[sample]), dx2 = (px - cursor.x) ** 2;
        if (dx2 > best) break;
        for (let band = 0; band < data.bands; band++) {
          const frequency = data.frequencies[sample * data.bands + band];
          if (!Number.isFinite(frequency)) continue;
          const py = y(frequency), distance = dx2 + (py - cursor.y) ** 2;
          if (distance < best) { best = distance; nearest = { x: px, y: py, sample, band, polarization: data.polarization, frequency }; }
        }
      }
    }
  }
  return nearest;
}

export function useBandHover(bounds: { left: number; right: number; top: number; bottom: number }) {
  const [cursor, setCursor] = useState<Position>();
  function move(event: PointerEvent<SVGSVGElement>) {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    setCursor(point.x >= bounds.left && point.x <= bounds.right && point.y >= bounds.top && point.y <= bounds.bottom ? { x: point.x, y: point.y } : undefined);
  }
  return { cursor, handlers: { onPointerMove: move, onPointerDown: move, onPointerLeave: () => setCursor(undefined), onPointerCancel: () => setCursor(undefined) } };
}

export function BandTooltip({ point, cursor, width, height }: { point: BandPoint; cursor: Position; width: number; height: number }) {
  const boxWidth = Math.min(205, width - 12), boxHeight = 58;
  const compact = boxWidth < 175;
  const left = Math.max(6, Math.min(cursor.x + 14, width - boxWidth - 6));
  const top = Math.max(6, Math.min(cursor.y - boxHeight - 12, height - boxHeight - 6));
  return <g data-interactive="tooltip" role="tooltip" pointerEvents="none">
    <circle cx={point.x} cy={point.y} r="4" fill={theme.textPrimary} stroke={theme.surface} strokeWidth="1.5" />
    <g transform={`translate(${left},${top})`}>
      <rect width={boxWidth} height={boxHeight} rx="5" fill={theme.surface} stroke={theme.guideLine} />
      <text x="10" y="22" fill={theme.textPrimary} fontSize={compact ? 10 : 13} fontWeight="600">Band index {point.band} · {point.polarization}</text>
      <text x="10" y="42" fill={theme.textSecondary} fontSize={compact ? 10 : 12}>{!compact && `k[${point.sample}] · `}f = {point.frequency.toPrecision(6)}</text>
    </g>
  </g>;
}
