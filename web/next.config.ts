import type { NextConfig } from 'next';
import createMDX from '@next/mdx';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * The deployment base path. GitHub Pages serves the site from /blaze2d, which is
 * the default; set NEXT_BASE_PATH to an empty string to serve from the root.
 *
 * It is read from the environment rather than a .env file because Next applies
 * .env after the shell environment, so a file silently wins over an explicit
 * export and the root-path build would quietly come out prefixed.
 */
const base = process.env.NEXT_BASE_PATH ?? '/blaze2d';

/** Written by scripts/build-wasm.mjs; ties the page to the solver it ships. */
const build = JSON.parse(readFileSync('./public/wasm-blaze/build.json', 'utf8'));

const nextConfig: NextConfig = {
  // A static export: no server at runtime, so every route is prerendered and
  // every asset is addressed relative to `base`.
  output: 'export',
  basePath: base,
  assetPrefix: base ? `${base}/` : '',
  // Emit /about/index.html rather than /about.html, which is what a static host
  // needs in order to serve /about/ directly.
  trailingSlash: true,

  // Mirrors of build-time facts that client components need; see lib/paths.ts.
  env: {
    NEXT_PUBLIC_BASE_PATH: base,
    NEXT_PUBLIC_BLAZE_REVISION: build.source_revision,
    NEXT_PUBLIC_BLAZE_VERSION: build.version,
  },

  turbopack: {
    resolveAlias: {
      // The PDF viewer reaches for the Node-only `canvas` package; the browser
      // build must not try to resolve it.
      canvas: './empty-module.ts',
      'next-mdx-import-source-file': './mdx-components.tsx',
    },
  },

  // GitHub Pages has no image optimizer.
  images: { unoptimized: true },

  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
};

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: ['remark-frontmatter', resolve('scripts/content.mjs'), 'remark-gfm', 'remark-math'],
    rehypePlugins: [
      'rehype-katex',
      ['rehype-pretty-code', { theme: 'github-dark-default', keepBackground: false }],
    ],
  },
});

export default withMDX(nextConfig);
