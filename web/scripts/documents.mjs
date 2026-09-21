import { readFileSync, statSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

/**
 * Measures a PDF so the site can advertise its length and weight before a
 * reader commits to opening it.
 *
 * Page counts are read from the file rather than written by hand, because a
 * number in frontmatter goes stale the moment the document is rebuilt.
 */

/** Every FlateDecode stream in the file, inflated where possible. */
function inflatedStreams(buffer) {
  const chunks = [];
  const marker = Buffer.from('stream');
  const terminator = Buffer.from('endstream');
  let index = 0;
  while ((index = buffer.indexOf(marker, index)) !== -1) {
    let start = index + marker.length;
    if (buffer[start] === 0x0d) start++;
    if (buffer[start] === 0x0a) start++;
    const end = buffer.indexOf(terminator, start);
    if (end === -1) break;
    try {
      chunks.push(inflateSync(buffer.subarray(start, end)));
    } catch {
      // Not a deflate stream, or not one we can read: image data, fonts.
    }
    // Past the terminator, so the `stream` inside `endstream` is not matched
    // again -- doing so would skip every other stream in the file.
    index = end + terminator.length;
  }
  return chunks;
}

/**
 * The number of pages in a PDF.
 *
 * A PDF 1.5+ writer keeps the page tree inside compressed object streams, so a
 * plain text scan finds nothing; the streams have to be inflated first. The root
 * of the page tree holds the total, and it is the largest `/Count` attached to a
 * `/Type /Pages` dictionary. Counting `/Type /Page` objects is the fallback.
 */
export function pageCount(path) {
  const buffer = readFileSync(path);
  const texts = [buffer.toString('latin1'), ...inflatedStreams(buffer).map(chunk => chunk.toString('latin1'))];

  let total = 0;
  for (const text of texts) {
    for (const match of text.matchAll(/\/Type\s*\/Pages\b[\s\S]{0,400}?\/Count\s+(\d+)/g)) {
      total = Math.max(total, Number(match[1]));
    }
    for (const match of text.matchAll(/\/Count\s+(\d+)[\s\S]{0,400}?\/Type\s*\/Pages\b/g)) {
      total = Math.max(total, Number(match[1]));
    }
  }
  if (total) return total;

  let pages = 0;
  for (const text of texts) pages += [...text.matchAll(/\/Type\s*\/Page(?![s\w])/g)].length;
  return pages;
}

/**
 * Describes a PDF stored under `public/`.
 *
 * @param {URL} publicDirectory the site's public directory
 * @param {string} href the document's absolute site path, e.g. `/paper/x.pdf`
 */
export function describeDocument(publicDirectory, href) {
  const file = new URL('.' + href, publicDirectory);
  const pages = pageCount(file);
  if (!pages) throw new Error(`${href}: could not determine a page count`);
  return { href, pages, bytes: statSync(file).size };
}
