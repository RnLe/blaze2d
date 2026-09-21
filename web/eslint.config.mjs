import nextCoreWebVitals from 'eslint-config-next/core-web-vitals';
import nextTypeScript from 'eslint-config-next/typescript';

/**
 * Flat config. eslint-config-next ships flat arrays from v16, so no
 * compatibility shim is needed.
 */
const config = [
  {
    ignores: [
      '.next/**',
      'out/**',
      'node_modules/**',
      'next-env.d.ts',
      // Build outputs, not sources.
      'public/wasm-blaze/**',
      'app/(docs)/(content)/**',
      // Staging tree compiled by typst/tools/figure-export.
      '.figure-export/**',
    ],
  },
  ...nextCoreWebVitals,
  ...nextTypeScript,
  {
    rules: {
      // The site is a static export with `images.unoptimized`, so next/image
      // buys nothing over a plain <img> and would only add markup.
      '@next/next/no-img-element': 'off',

      // React Compiler rules, new in eslint-config-next 16. They flag patterns
      // that predate the upgrade and are correct as written but not
      // compiler-friendly: reading browser-only values after hydration
      // (DocsShell, PdfViewer), resetting derived state when inputs change
      // (Workbench, InteractiveBandDiagram, Fields), and keeping a callback
      // fresh in a ref (Workbench, TomlEditor). Warnings so the signal stays
      // visible; moving the Workbench to the compiler's model is its own task.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/refs': 'warn',
    },
  },
];

export default config;
