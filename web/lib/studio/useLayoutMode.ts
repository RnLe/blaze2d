'use client';

import { useEffect, useState } from 'react';

export type LayoutMode = 'wide' | 'mid' | 'narrow';

/**
 * wide  (>= 1100px): three columns side by side.
 * mid   (900-1099px): left + center; the TOML pane becomes an overlay drawer.
 * narrow (< 900px):  center only; config panel AND TOML become drawers.
 */
export function useLayoutMode(): LayoutMode {
  const [mode, setMode] = useState<LayoutMode>('wide');

  useEffect(() => {
    const wide = window.matchMedia('(min-width: 1100px)');
    const mid = window.matchMedia('(min-width: 900px)');
    const update = () => setMode(wide.matches ? 'wide' : mid.matches ? 'mid' : 'narrow');
    update();
    wide.addEventListener('change', update);
    mid.addEventListener('change', update);
    return () => {
      wide.removeEventListener('change', update);
      mid.removeEventListener('change', update);
    };
  }, []);

  return mode;
}
