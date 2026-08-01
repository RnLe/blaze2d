// Minimal TrueType `cmap` reader.
//
// The exported charts are labelled in OpenAI Sans, but that face carries no
// Greek: the band-structure axes use omega, pi and Gamma, the dielectric sweep
// uses epsilon, and the scaling fits use the proportional-to sign. In a browser
// those characters silently fall through the font-family list to the next face
// that has them. Renderers outside the browser (Typst's resvg, cairosvg) do not
// all perform that per-glyph fallback and draw tofu instead.
//
// Reading the coverage straight out of the .ttf that Typst itself loads lets the
// post-processor mark exactly the characters that need a fallback, and no
// others -- superscripts and the multiplication sign are covered by OpenAI Sans
// and must keep using it.

import { readFileSync } from 'node:fs';

function readTableDirectory(buf) {
  const tables = new Map();
  const numTables = buf.readUInt16BE(4);
  for (let i = 0; i < numTables; i += 1) {
    const record = 12 + i * 16;
    const tag = buf.toString('ascii', record, record + 4);
    tables.set(tag, { offset: buf.readUInt32BE(record + 8), length: buf.readUInt32BE(record + 12) });
  }
  return tables;
}

/** Pick the most capable cmap subtable: full Unicode first, then BMP. */
function selectSubtable(buf, cmapOffset) {
  const numSubtables = buf.readUInt16BE(cmapOffset + 2);
  const candidates = [];
  for (let i = 0; i < numSubtables; i += 1) {
    const record = cmapOffset + 4 + i * 8;
    candidates.push({
      platformId: buf.readUInt16BE(record),
      encodingId: buf.readUInt16BE(record + 2),
      offset: cmapOffset + buf.readUInt32BE(record + 4),
    });
  }

  const rank = ({ platformId, encodingId }) => {
    if (platformId === 3 && encodingId === 10) return 0;   // Windows, UCS-4
    if (platformId === 0 && encodingId >= 4) return 1;     // Unicode, full repertoire
    if (platformId === 3 && encodingId === 1) return 2;    // Windows, BMP
    if (platformId === 0) return 3;                        // Unicode, BMP
    return 4;
  };

  return candidates.sort((a, b) => rank(a) - rank(b))[0];
}

function readFormat4(buf, offset, covered) {
  const segCount = buf.readUInt16BE(offset + 6) / 2;
  const endCodes = offset + 14;
  const startCodes = endCodes + segCount * 2 + 2;
  const idDeltas = startCodes + segCount * 2;
  const idRangeOffsets = idDeltas + segCount * 2;

  for (let seg = 0; seg < segCount; seg += 1) {
    const end = buf.readUInt16BE(endCodes + seg * 2);
    const start = buf.readUInt16BE(startCodes + seg * 2);
    if (start === 0xffff) continue;                        // terminating segment
    const delta = buf.readInt16BE(idDeltas + seg * 2);
    const rangeOffset = buf.readUInt16BE(idRangeOffsets + seg * 2);

    for (let code = start; code <= end && code !== 0x10000; code += 1) {
      let glyph;
      if (rangeOffset === 0) {
        glyph = (code + delta) & 0xffff;
      } else {
        const index = idRangeOffsets + seg * 2 + rangeOffset + (code - start) * 2;
        if (index + 1 >= buf.length) continue;
        glyph = buf.readUInt16BE(index);
        if (glyph !== 0) glyph = (glyph + delta) & 0xffff;
      }
      if (glyph !== 0) covered.add(code);
    }
  }
}

function readFormat12(buf, offset, covered) {
  const groups = buf.readUInt32BE(offset + 12);
  for (let i = 0; i < groups; i += 1) {
    const group = offset + 16 + i * 12;
    const start = buf.readUInt32BE(group);
    const end = buf.readUInt32BE(group + 4);
    const startGlyph = buf.readUInt32BE(group + 8);
    if (startGlyph === 0 && start === 0) continue;
    for (let code = start; code <= end; code += 1) covered.add(code);
  }
}

/**
 * Codepoints a font can actually render.
 *
 * @param {string} fontPath path to a .ttf/.otf file
 * @returns {Set<number>}
 */
export function readCoverage(fontPath) {
  const buf = readFileSync(fontPath);
  const cmap = readTableDirectory(buf).get('cmap');
  if (!cmap) throw new Error(`no cmap table in ${fontPath}`);

  const subtable = selectSubtable(buf, cmap.offset);
  if (!subtable) throw new Error(`no usable cmap subtable in ${fontPath}`);

  const covered = new Set();
  const format = buf.readUInt16BE(subtable.offset);
  if (format === 4) readFormat4(buf, subtable.offset, covered);
  else if (format === 12) readFormat12(buf, subtable.offset, covered);
  else throw new Error(`unsupported cmap format ${format} in ${fontPath}`);

  return covered;
}
