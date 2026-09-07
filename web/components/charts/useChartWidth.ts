'use client';
import { useEffect, useRef, useState } from 'react';

/** Preserve legible scientific axes with local scrolling below the minimum. */
export function useChartWidth(maximum: number, minimum = 560) {
  const ref = useRef<HTMLDivElement>(null), [available, setAvailable] = useState(maximum);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(entries => { const size = entries[0]?.contentRect.width; if (size) setAvailable(size); });
    observer.observe(ref.current); return () => observer.disconnect();
  }, []);
  return { ref, width: Math.min(maximum, Math.max(Math.min(minimum, maximum), available)) };
}
