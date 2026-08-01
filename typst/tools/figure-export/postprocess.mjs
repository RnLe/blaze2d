// Turns the markup react-dom/server produces for a chart component into a
// standalone SVG that Typst can embed unchanged.
//
// Three problems have to be solved, in this order:
//
//  1. Multi-panel components. MemoryUsageChart, MemoryRatioChart,
//     ThroughputScalingChart and SpeedupScalingChart each render two <svg>
//     roots inside a `display:flex; gap:20px` wrapper. They are composed side
//     by side with that same 20px gap so the exported figure matches the page.
//
//  2. Nested <svg> wrappers. @visx/text wraps every single label in
//     `<svg x y style="overflow:visible">`. Browsers honour overflow:visible;
//     Typst's SVG renderer clips nested viewports instead, which would silently
//     erase every y-axis tick label (they use text-anchor="end" at x=0 and so
//     extend to the left of their own viewport). Each wrapper is rewritten to
//     `<g transform="translate(...)">`, with em-valued offsets resolved against
//     the wrapper's own font-size.
//
//  3. Fonts and background. `--font-sans` is never defined, so
//     `font-family: var(--font-sans), system-ui, sans-serif` is invalid at
//     computed-value time and the text simply inherits <body>'s stack. That
//     inheritance cannot survive outside the page, so the inherited stack is
//     written in explicitly, led by the extended face that build-fonts.py
//     produces (see FONT_FAMILY). The charts are likewise designed against the
//     page's #000000 background, which is baked in here as a full-bleed rect.

import { readCoverage } from './font-coverage.mjs';

/** Padding baked around the composed chart, in SVG user units. */
const PADDING = 16;

/** Horizontal gap between panels of a multi-panel component (matches the flex gap). */
const PANEL_GAP = 20;

/** Page background the charts are drawn against (web/app/layout.tsx). */
const BACKGROUND = '#000000';

/**
 * The stack the site's chart text actually resolves to, inherited verbatim from
 * <body> in web/app/layout.tsx. The fallbacks matter: axis labels contain Greek
 * (omega a / 2 pi c, Gamma) which OpenAI Sans has no glyphs for, so the browser
 * falls through the list for those characters. Roboto is the first entry that
 * both this machine and Typst can resolve, so keeping the full chain reproduces
 * the page instead of rendering tofu.
 */
/**
 * The font stack written into every exported chart.
 *
 * The tail is verbatim what the site's chart text inherits from <body>, so the
 * SVG renders exactly like the page in any browser. 'OpenAI Sans Extended' is
 * prepended for Typst: resvg resolves one font per <text> element rather than
 * per glyph, so a single character outside the face -- the Greek in the axis
 * labels, the proportional sign in the fit annotations -- would drag the whole
 * label onto a fallback serif. The extended face (typst/assets/fonts, built by
 * tools/figure-export/build-fonts.py) is OpenAI Sans plus exactly those glyphs,
 * borrowed from the same Roboto that heads the site's own fallback chain. Where
 * it is not installed, the next entry is the shipped face and nothing changes.
 */
const FONT_FAMILY =
  "'OpenAI Sans Extended', 'OpenAI Sans', -apple-system, BlinkMacSystemFont, " +
  "'Segoe UI', Roboto, sans-serif";

const TAG_RE = /<(\/?)([a-zA-Z][\w:.-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)\s*(\/?)>/g;
const ATTR_RE = /([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)"/g;

function parseAttrs(source) {
  const attrs = new Map();
  for (const [, name, value] of source.matchAll(ATTR_RE)) attrs.set(name, value);
  return attrs;
}

function serializeAttrs(attrs) {
  return [...attrs].map(([name, value]) => ` ${name}="${value}"`).join('');
}

/**
 * Resolve an SVG coordinate that may carry a CSS unit.
 * visx emits dy values such as '0.33em' and '0.5em', which are relative to the
 * font-size declared on the very wrapper being flattened away.
 */
function resolveLength(value, fontSize) {
  if (value === undefined || value === null || value === '') return 0;
  if (typeof value === 'number') return value;
  const text = String(value).trim();
  const number = Number.parseFloat(text);
  if (!Number.isFinite(number)) return 0;
  if (text.endsWith('em')) return number * fontSize;
  return number;                       // bare user units, or 'px'
}

/** Trim floating point noise so the output stays readable and diff-friendly. */
function round(value) {
  return Number.parseFloat(value.toFixed(4));
}

/** Walk the markup and hand every <svg> open/close tag to the caller in order. */
function forEachSvgTag(markup, visit) {
  for (const match of markup.matchAll(TAG_RE)) {
    const [full, closing, name, attrSource, selfClosing] = match;
    if (name !== 'svg') continue;
    visit({
      full,
      isClosing: closing === '/',
      isSelfClosing: selfClosing === '/',
      attrSource,
      start: match.index,
      end: match.index + full.length,
    });
  }
}

/** Split the rendered markup into its top-level <svg> roots. */
function extractRoots(markup) {
  const roots = [];
  let depth = 0;
  let openTag = null;
  let contentStart = 0;

  forEachSvgTag(markup, (tag) => {
    if (tag.isSelfClosing) return;
    if (!tag.isClosing) {
      if (depth === 0) {
        openTag = tag;
        contentStart = tag.end;
      }
      depth += 1;
      return;
    }
    depth -= 1;
    if (depth === 0 && openTag) {
      roots.push({
        attrs: parseAttrs(openTag.attrSource),
        inner: markup.slice(contentStart, tag.start),
      });
      openTag = null;
    }
  });

  return roots;
}

/**
 * Replace every nested <svg> wrapper with an equivalent <g transform>.
 * The document root (depth 0) is left alone; only the visx label wrappers and
 * any other nested viewport are rewritten.
 */
function flattenNestedSvgs(markup) {
  let depth = 0;
  let cursor = 0;
  let out = '';
  const rewritten = [];

  forEachSvgTag(markup, (tag) => {
    if (tag.isSelfClosing) return;
    out += markup.slice(cursor, tag.start);
    cursor = tag.end;

    if (tag.isClosing) {
      depth -= 1;
      out += rewritten.pop() ? '</g>' : tag.full;
      return;
    }

    if (depth === 0) {
      out += tag.full;
      rewritten.push(false);
    } else {
      out += openGroupFor(tag.attrSource);
      rewritten.push(true);
    }
    depth += 1;
  });

  return out + markup.slice(cursor);
}

/** Build the <g> that stands in for one nested <svg> wrapper. */
function openGroupFor(attrSource) {
  const attrs = parseAttrs(attrSource);
  const fontSize = Number.parseFloat(attrs.get('font-size') ?? '0') || 0;
  const dx = resolveLength(attrs.get('x'), fontSize);
  const dy = resolveLength(attrs.get('y'), fontSize);

  const carried = new Map();
  if (attrs.has('font-size')) carried.set('font-size', attrs.get('font-size'));
  if (dx !== 0 || dy !== 0) {
    carried.set('transform', `translate(${round(dx)}, ${round(dy)})`);
  }
  return `<g${serializeAttrs(carried)}>`;
}

/** Write the resolved font family in place of the site's inherited custom property. */
function resolveFontFamily(markup) {
  return markup.replaceAll('var(--font-sans), system-ui, sans-serif', FONT_FAMILY);
}

/** Drop the overflow hints; the flattening above makes them meaningless. */
function stripOverflowStyles(markup) {
  return markup.replace(/\s*style="[^"]*overflow\s*:\s*visible[^"]*"/g, '');
}

/** Apply `fn` to every text node in a fragment, leaving tags untouched. */
function mapTextNodes(html, fn) {
  return html.replace(/(<[^>]*>)|([^<]+)/g, (_match, tag, text) => (tag === undefined ? fn(text) : tag));
}

/**
 * Report characters the primary face cannot draw.
 *
 * This is a guard, not a repair. Because resvg picks one font per <text>
 * element, a single uncovered character silently re-faces an entire label, and
 * the result reads as a styling bug rather than a missing glyph. Surfacing the
 * characters means a future post that introduces a new symbol shows up in the
 * export log instead of in the PDF.
 */
function auditGlyphCoverage(markup, coverage) {
  const missing = new Set();
  for (const match of markup.matchAll(/<text\b[^>]*>([\s\S]*?)<\/text>/g)) {
    mapTextNodes(match[1], (text) => {
      for (const char of text) {
        if (!coverage.has(char.codePointAt(0))) missing.add(char);
      }
      return text;
    });
  }
  return missing;
}

/**
 * Compose one or more rendered <svg> roots into a single padded, black-backed
 * SVG document.
 *
 * @param {string} markup             output of renderToStaticMarkup for one chart
 * @param {object} [options]
 * @param {Set<number>} [options.coverage] codepoints the primary font can render;
 *   supplying it turns on the missing-glyph audit
 * @returns {{svg: string, width: number, height: number, panels: number, missingGlyphs: string[]}}
 */
export function toStandaloneSvg(markup, { coverage } = {}) {
  const roots = extractRoots(markup);
  if (roots.length === 0) {
    throw new Error('no <svg> root found in rendered markup');
  }

  const panels = roots.map((root) => ({
    width: Number.parseFloat(root.attrs.get('width') ?? '0'),
    height: Number.parseFloat(root.attrs.get('height') ?? '0'),
    inner: root.inner,
  }));

  if (panels.some((p) => !Number.isFinite(p.width) || !Number.isFinite(p.height))) {
    throw new Error('an <svg> root is missing a numeric width/height');
  }

  const contentWidth =
    panels.reduce((sum, p) => sum + p.width, 0) + PANEL_GAP * (panels.length - 1);
  const contentHeight = Math.max(...panels.map((p) => p.height));

  let offset = 0;
  const body = panels
    .map((panel) => {
      const group =
        offset === 0
          ? panel.inner
          : `<g transform="translate(${round(offset)}, 0)">${panel.inner}</g>`;
      offset += panel.width + PANEL_GAP;
      return group;
    })
    .join('');

  const width = round(contentWidth + 2 * PADDING);
  const height = round(contentHeight + 2 * PADDING);

  const document =
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ` +
    `width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
    `<rect x="0" y="0" width="${width}" height="${height}" fill="${BACKGROUND}"/>` +
    `<g transform="translate(${PADDING}, ${PADDING})">${body}</g>` +
    `</svg>`;

  const svg = stripOverflowStyles(resolveFontFamily(flattenNestedSvgs(document)));
  const missing = coverage ? auditGlyphCoverage(svg, coverage) : new Set();

  return { svg, width, height, panels: panels.length, missingGlyphs: [...missing] };
}

export { PADDING, PANEL_GAP, BACKGROUND, FONT_FAMILY, readCoverage };
