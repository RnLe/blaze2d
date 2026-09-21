import { documents } from './documents.generated';

export interface DocumentInfo {
  /** Absolute site path of the PDF, e.g. `/paper/blaze2d.pdf`. */
  href: string;
  pages: number;
  bytes: number;
}

/**
 * A PDF's measured length and weight.
 *
 * Measured at build time by scripts/documents.mjs, so the figures cannot drift
 * from the file that ships.
 */
export function getDocument(href: string): DocumentInfo {
  const info = documents[href];
  if (!info) throw new Error(`Unknown document: ${href}. Declare it in an article's frontmatter.`);
  return info;
}

/** Shows MB above a megabyte and KB below, e.g. `0.9 MB`, `340 KB`. */
export function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

/** The compact size line shown beside a document link, e.g. `11p. | 0.9 MB`. */
export function formatDocument({ pages, bytes }: Pick<DocumentInfo, 'pages' | 'bytes'>): string {
  return `${pages}p. | ${formatBytes(bytes)}`;
}
