// Synchronous replacement for the browser fetch() used by the chart data hooks.
//
// The website loads its benchmark JSON at runtime from web/public/data/. The
// exporter renders the very same components under react-dom/server, where
// useEffect never runs and fetch() is unavailable, so every data path is
// resolved here with a blocking read of the identical file. Reading the real
// public/data payload (rather than the FALLBACK_* constants embedded in
// lib/benchmark-data.ts) is what makes the exported figures match what a
// visitor actually sees.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const PUBLIC_DIR = process.env.BLAZE_WEB_PUBLIC_DIR;

if (!PUBLIC_DIR) {
  throw new Error('BLAZE_WEB_PUBLIC_DIR is not set; cannot locate web/public');
}

const cache = new Map<string, unknown>();

/** Load a JSON asset by its site-absolute path, e.g. '/data/benchmarks/single-core.json'. */
export function loadJson<T>(assetPath: string): T {
  const cached = cache.get(assetPath);
  if (cached !== undefined) return cached as T;

  const relative = assetPath.replace(/^\//, '');
  const parsed = JSON.parse(readFileSync(join(PUBLIC_DIR!, relative), 'utf8'));
  cache.set(assetPath, parsed);
  return parsed as T;
}
