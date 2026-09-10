import type { NextConfig } from 'next';
import nextra from 'nextra'
import { readFileSync } from 'node:fs'

const base = process.env.NEXT_BASE_PATH ?? '/blaze2d';
const build = JSON.parse(readFileSync('./public/wasm-blaze/build.json', 'utf8'));

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {

    // EXPORT RELATED CONFIG
    output: 'export',
    // when exporting to gh pages, prefix all routes/assets with /blaze2d
    basePath: base,
    assetPrefix: base ? `${base}/` : '',
    trailingSlash: true,         // output /about/index.html instead of about.html
    
    // Make base path available to client-side code
    env: {
        NEXT_PUBLIC_BASE_PATH: base,
        NEXT_PUBLIC_BLAZE_REVISION: build.source_revision,
        NEXT_PUBLIC_BLAZE_VERSION: build.version,
    },
    
    // FUNCTIONALITY RELATED CONFIG
    transpilePackages: [],
    serverExternalPackages: ['pino'],
    turbopack: {
        resolveAlias: {
            canvas: "./empty-module.ts",
            'next-mdx-import-source-file': './mdx-components.jsx'
        }
    },
    // Webpack fallback for production builds
    webpack(config) {
        // Stub out 'canvas' for client and server bundles via fallback
        config.resolve.fallback = {
            ...(config.resolve.fallback ?? {}),
            canvas: false,
        };
        
        // Enable WASM support
        config.experiments = {
            ...config.experiments,
            asyncWebAssembly: true,
            layers: true,
        };
        
        // Handle WASM files properly for Next.js
        config.module.rules.push({
            test: /\.wasm$/,
            type: 'asset/resource',
        });
        
        // Resolve WASM imports
        config.resolve.extensions.push('.wasm');
        
        return config;
    },
    images: {
        unoptimized: true,          // disable image optimization. Necessary for GitHub Pages
    },
    pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
};

// Set up Nextra with its configuration
const withNextra = nextra({
// pick either preset; KaTeX is pre-rendered, MathJax hydrates client-side
  latex: true,                 // shorthand → { renderer: 'katex' }
  // latex: { renderer: 'mathjax' },
})
 
// Export the final Next.js config with Nextra included
export default withNextra(nextConfig)