import { readFileSync, writeFileSync } from 'node:fs';

// An exact triangular Bravais lattice. Shared SVG patterns keep thousands of
// sites compact. 64a across, versus 17a in the landing-card loop.
const tokens = readFileSync(new URL('../app/styles/tokens.css', import.meta.url), 'utf8');
const token = name => tokens.match(new RegExp(`--${name}:\\s*(#[a-fA-F0-9]+)`))[1];
const a = 25, h = a * Math.sqrt(3) / 2;
const pattern = (id, colour) => `<pattern id="${id}" width="${a}" height="${2 * h}" patternUnits="userSpaceOnUse" x="${800 - a / 4}" y="${410 - h / 2}">
  <circle cx="${a / 4}" cy="${h / 2}" r="4.6" fill="${colour}"/>
  <circle cx="${3 * a / 4}" cy="${3 * h / 2}" r="4.6" fill="${colour}"/>
</pattern>`;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="820" viewBox="0 0 1600 820">
<title>A triangular bilayer at a six-degree twist</title>
<desc>Blue and orange triangular lattices, sixty-four lattice constants across. Solid sites reveal several moiré cells on a black background.</desc>
<defs>
  ${pattern('blue', token('brand-blue'))}
  ${pattern('orange', token('brand-orange'))}
  <radialGradient id="fade"><stop offset="40%" stop-color="${token('surface-base')}" stop-opacity="0"/><stop offset="100%" stop-color="${token('surface-base')}" stop-opacity="0.55"/></radialGradient>
</defs>
<rect width="1600" height="820" fill="${token('surface-base')}"/>
<rect x="-250" y="-400" width="2100" height="1620" fill="url(#blue)"/>
<g style="mix-blend-mode:screen" transform="rotate(6 800 410)"><rect x="-250" y="-400" width="2100" height="1620" fill="url(#orange)"/></g>
<rect width="1600" height="820" fill="url(#fade)"/>
</svg>\n`;
writeFileSync(new URL('../public/banners/thesis-triangular-bilayer.svg', import.meta.url), svg);
