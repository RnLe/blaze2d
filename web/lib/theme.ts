/**
 * Theme values for graphics drawn from JavaScript.
 *
 * SVG can read CSS custom properties, but a 2D canvas cannot, so the palette
 * needs a literal form. These values mirror `app/styles/tokens.css` -- change
 * one and change the other. Nothing else in the codebase should hold a hex
 * literal for a colour.
 */

/** Blue accent and the neutrals used by plots and canvases. */
export const theme = {
  accent: '#59b6ff',
  brandBlue: '#317cb8',
  brandOrange: '#eb7929',
  /** The brand pair lightened, for a second series drawn over its own base. */
  brandBlueLight: '#8eb7d8',
  brandOrangeLight: '#f4b589',
  figurePaper: '#ffffff',
  surface: '#090909',
  gridLine: '#242424',
  axisLine: '#333333',
  textPrimary: '#ededed',
  textSecondary: '#b5b5b5',
  textMuted: '#a1a1a1',
  textSubtle: '#8f8f8f',
  /** Reference and fit lines drawn over data. */
  guideLine: '#8f8f8f',
  /** A measurement that went the wrong way; the only non-blue signal colour. */
  negative: '#ef4444',
  /** Light panel behind figures exported on a transparent background. */
  paper: '#f0f0f0',
  paperInk: '#666666',
} as const;

/**
 * Series colours, ordered from "external baseline" to "ours, highlighted".
 *
 * Used by the benchmark charts on the technical report, the pitch animations and
 * the Workbench band plot, so a reader meets the same colour for the same thing
 * wherever it appears.
 */
export const series = {
  reference: '#5477c4', // MPB, or any external baseline
  primary: '#a3befa', // Blaze, full precision
  highlight: '#eaf1fe', // Blaze, mixed precision
  muted: '#bbc1cb', // a secondary comparison, deliberately desaturated
} as const;

/** Distinct colours for an arbitrary number of bands, cycled in order. */
export const bandColors = [
  '#59b6ff',
  '#a3befa',
  '#7fd4e8',
  '#8ea8e8',
  '#b8d8ff',
  '#6b9fd4',
] as const;

/**
 * Colour ramp for a dielectric map: near-black at the lowest permittivity,
 * accent blue at the highest.
 *
 * `t` is the normalised permittivity in [0, 1].
 */
const EPSILON_LOW = [7, 7, 10] as const;
const EPSILON_HIGH = [89, 182, 255] as const;

export function epsilonRgb(t: number): [number, number, number] {
  return [
    Math.round(EPSILON_LOW[0] + (EPSILON_HIGH[0] - EPSILON_LOW[0]) * t),
    Math.round(EPSILON_LOW[1] + (EPSILON_HIGH[1] - EPSILON_LOW[1]) * t),
    Math.round(EPSILON_LOW[2] + (EPSILON_HIGH[2] - EPSILON_LOW[2]) * t),
  ];
}

export function epsilonColor(t: number): string {
  const [r, g, b] = epsilonRgb(t);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * TOML editor syntax colours.
 *
 * Syntax highlighting needs hues the accent alone cannot supply; these stay in
 * the cool half of the wheel so the editor still reads as part of the site.
 */
export const syntax = {
  comment: '#8f9299',
  string: '#a9d6f5',
  number: '#e3bd87',
  keyword: theme.accent,
  property: '#d2d6dd',
  punctuation: '#a7adb8',
  selection: '#1d3d55',
} as const;
