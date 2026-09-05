import { examples } from './catalog.generated';
export { examples };
export type Example = typeof examples[number];

// Preserve entry links from earlier documentation while directing readers to current examples.
export const relocatedExamples: Record<string, string> = {
  'first-band-diagram': 'square-rods', 'crystal-from-toml': 'square-rods', 'tm-vs-te': 'radius-sweep',
  'hexagonal-lattice': 'triangular-holes', 'honeycomb-toml': 'triangular-holes', 'custom-k-path': 'oblique-cell',
  'two-atom-basis': 'square-rods', 'resolution-convergence': 'rectangular-cell', 'nested-sweeps': 'radius-sweep',
  'two-dimensional-sweep': 'radius-sweep', 'set-number-of-workers': 'radius-sweep', 'live-k-point-streaming': 'square-rods',
  'selective-k-points-bands': 'square-rods', 'inspect-band-result': 'square-rods',
};
export function getExample(slug: string) { return examples.find(example => example.slug === slug); }
export function getExampleSlugs() { return [...examples.map(example => example.slug), ...Object.keys(relocatedExamples)]; }
