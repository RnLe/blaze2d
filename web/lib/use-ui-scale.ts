'use client';
import { useEffect, useState } from 'react';

/**
 * The current UI scale: 1 at the design anchor (a 2560x1440 viewport), 0.75 on
 * a 1080p screen, 1.5 on a 4K one.
 *
 * CSS gets this for free, because the layout is written in rem and the root font
 * size carries the scale (see app/styles/tokens.css). Graphics drawn from
 * JavaScript have no such luck and must read it back, which is what this is for.
 *
 * It starts at 1 so server and client render the same markup; the real value
 * arrives on the first effect, which is also when a chart first measures its
 * container.
 */
export function useUiScale(): number {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const read = () => {
      const root = parseFloat(getComputedStyle(document.documentElement).fontSize);
      if (root > 0) setScale(root / 16);
    };
    read();
    window.addEventListener('resize', read);
    return () => window.removeEventListener('resize', read);
  }, []);

  return scale;
}
