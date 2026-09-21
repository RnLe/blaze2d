'use client';
import { useEffect, useRef, useState } from 'react';
import { useUiScale } from '@/lib/use-ui-scale';

/**
 * Measures a chart's container and reports the width to draw at, in design
 * units.
 *
 * A chart does all its arithmetic -- margins, tick sizes, label sizes -- in the
 * units it was designed in, and is scaled to the display only at the `<svg>`
 * boundary, through `scale`. That keeps one set of numbers in the chart code
 * while still matching the surrounding interface on a 1080p or a 4K screen.
 *
 * Below `minimum` the chart stops shrinking and its container scrolls
 * horizontally, so scientific axes stay legible instead of collapsing.
 */
export function useChartWidth(maximum: number, minimum = 560) {
  const scale = useUiScale();
  const ref = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(maximum);

  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(entries => {
      const size = entries[0]?.contentRect.width;
      if (size) setAvailable(size);
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  // The observer reports rendered pixels; convert back to design units.
  const design = available / scale;
  const width = Math.min(maximum, Math.max(Math.min(minimum, maximum), design));
  return { ref, width, scale };
}
