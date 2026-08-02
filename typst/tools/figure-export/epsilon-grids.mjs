// Static stand-in for the page's interactive EpsilonGridViewer.
//
// The viewer paints the smoothed permittivity map onto a canvas and shows it
// under a resolution slider, so it has no print form of its own. This renders
// the same maps for a fixed set of resolutions as a row of pixel-exact vector
// tiles, reusing the viewer's own colour ramp and its "Resolution: N x N"
// caption styling, so the printed figure reads as the same widget frozen at
// four positions.
//
// Tiles are emitted as run-length-merged <rect>s rather than an embedded
// bitmap: that keeps the hard pixel edges the viewer gets from
// `image-rendering: pixelated`, stays vector at any zoom, and avoids shipping
// a raster (the repo gitignores **.png, and the thesis style guide asks for
// SVG figures).

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/** Colour ramp from EpsilonGridViewer: low epsilon (hole) to high (background). */
const COLOR_EPS_LOW = { r: 0xbb, g: 0xc1, b: 0xcb };
const COLOR_EPS_HIGH = { r: 0x43, g: 0x5f, b: 0x9d };

/** Accent used by the viewer for the active resolution readout. */
const ACCENT = '#a3befa';
/**
 * Muted label colour. The viewer uses #666, which survives on screen but goes
 * illegible once the four-up row is scaled down to a text column; #888 is the
 * muted tone the site's own chart captions use and reads at print size.
 */
const MUTED = '#888';

// Sized so the row lands close to a page's text width without much
// downscaling, which keeps the caption legible in print.
const TILE = 200;
const GAP = 20;
const LABEL_FONT = 18;
const LABEL_HEIGHT = 30;
const TILE_RADIUS = 4;

function interpolateColor(t) {
  const mix = (lo, hi) => Math.round(lo + t * (hi - lo));
  const r = mix(COLOR_EPS_LOW.r, COLOR_EPS_HIGH.r);
  const g = mix(COLOR_EPS_LOW.g, COLOR_EPS_HIGH.g);
  const b = mix(COLOR_EPS_LOW.b, COLOR_EPS_HIGH.b);
  return `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
}

/**
 * Emit one permittivity map as run-length-merged rects in a TILE x TILE box.
 * The stored grid runs bottom-up (iy = 0 is the bottom row), so rows are
 * flipped to match the viewer's canvas orientation.
 */
function renderTile(data) {
  const { nx, ny, grid, epsilon_background: high, epsilon_hole: low } = data;
  const range = high - low;
  const parts = [];

  // Cell boundaries are derived from a single edge function so neighbouring
  // rects share an exact coordinate. Combined with shape-rendering="crispEdges"
  // below, that removes the hairline antialiasing seams that show up between
  // abutting fills at fractional cell sizes.
  const edgeX = (i) => (i * TILE) / nx;
  const edgeY = (j) => (j * TILE) / ny;

  for (let iy = 0; iy < ny; iy += 1) {
    const row = ny - 1 - iy;                 // stored grid runs bottom-up
    const y = edgeY(row);
    const height = edgeY(row + 1) - y;
    let runStart = 0;
    let runColor = null;

    const flush = (end) => {
      if (runColor === null || end === runStart) return;
      const x = edgeX(runStart);
      parts.push(
        `<rect x="${x.toFixed(4)}" y="${y.toFixed(4)}" ` +
          `width="${(edgeX(end) - x).toFixed(4)}" height="${height.toFixed(4)}" fill="${runColor}"/>`,
      );
    };

    for (let ix = 0; ix < nx; ix += 1) {
      const eps = grid[iy * nx + ix];
      const color = interpolateColor(range > 0 ? (eps - low) / range : 0);
      if (color !== runColor) {
        flush(ix);
        runStart = ix;
        runColor = color;
      }
    }
    flush(nx);
  }

  return parts.join('');
}

/**
 * Build the full 1 x N row of permittivity maps.
 *
 * @param {object} options
 * @param {string} options.dataDir      web/public/data/epsilon-grids
 * @param {number[]} options.resolutions resolutions to show, left to right
 * @returns {string} a standalone SVG fragment (root <svg>, no background)
 */
export function renderEpsilonGridRow({ dataDir, resolutions }) {
  const width = resolutions.length * TILE + (resolutions.length - 1) * GAP;
  const height = TILE + LABEL_HEIGHT;

  const tiles = resolutions.map((resolution, index) => {
    const file = join(dataDir, `epsilon-res${resolution}.json`);
    const data = JSON.parse(readFileSync(file, 'utf8'));
    const x = index * (TILE + GAP);

    return (
      `<g transform="translate(${x}, 0)">` +
        `<clipPath id="eps-clip-${resolution}">` +
          `<rect x="0" y="0" width="${TILE}" height="${TILE}" rx="${TILE_RADIUS}"/>` +
        `</clipPath>` +
        `<g clip-path="url(#eps-clip-${resolution})" shape-rendering="crispEdges">${renderTile(data)}</g>` +
        `<text x="${TILE / 2}" y="${TILE + 21}" text-anchor="middle" font-size="${LABEL_FONT}" ` +
          `font-family="var(--font-sans), system-ui, sans-serif" fill="${MUTED}">` +
          `Resolution: <tspan font-weight="600" fill="${ACCENT}">` +
          `${resolution}×${resolution}</tspan>` +
        `</text>` +
      `</g>`
    );
  });

  return (
    `<svg width="${width}" height="${height}">${tiles.join('')}</svg>`
  );
}

export { TILE, GAP };
