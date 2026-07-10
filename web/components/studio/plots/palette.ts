/**
 * Series colors for studio plots: Okabe-Ito base (color-blind safe among the
 * first eight), brightened where needed for the dark plot surface, extended
 * to twenty via lightness variants for large sweeps.
 */

const BASE = [
  '#56b4e9', // sky blue
  '#e69f00', // orange
  '#00c48f', // green (brightened bluish green)
  '#f0e442', // yellow
  '#d98cb3', // pink (brightened reddish purple)
  '#3f8fd2', // blue (brightened)
  '#e8702a', // vermillion (brightened)
  '#b3b3b3', // grey
];

const EXTENDED = [
  ...BASE,
  '#8dd3c7',
  '#fb8072',
  '#bebada',
  '#fdb462',
  '#80b1d3',
  '#b3de69',
  '#fccde5',
  '#c9b2f0',
  '#ffed6f',
  '#66d9cf',
  '#f2a3a3',
  '#a6d96a',
];

export function seriesColor(index: number): string {
  return EXTENDED[index % EXTENDED.length];
}

export const PLOT_SURFACE = '#0a0a0a';
export const PLOT_GRID = '#1f2937';
export const PLOT_AXIS = '#374151';
export const PLOT_TICK_TEXT = '#9ca3af';
export const PLOT_LABEL_TEXT = '#d1d5db';
export const PLOT_FONT = "-apple-system, 'Segoe UI', system-ui, sans-serif";
