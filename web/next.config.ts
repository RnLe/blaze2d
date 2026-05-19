// apps/website/next.config.ts

import type { NextConfig } from 'next';
import nextra from 'nextra'
import { existsSync } from 'node:fs'

const base = process.env.NEXT_BASE_PATH ?? '/blaze2d';

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
    },
    
    // FUNCTIONALITY RELATED CONFIG
    // GitHub Pages requires static export and serves from /msl, so we need Next to emit all HTML/CSS/JS under that path.
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
        
        // Copy WASM files to public directory during build
        if (config.mode === 'production') {
            const path = require('path');
            const CopyPlugin = require('copy-webpack-plugin');
            const wasmSource = path.resolve(__dirname, 'wasm');
            const wasmTarget = path.resolve(__dirname, 'out/wasm');
            
            if (existsSync(wasmSource)) {
                config.plugins.push(
                    new CopyPlugin({
                        patterns: [
                            {
                                from: wasmSource,
                                to: wasmTarget,
                            },
                        ],
                    })
                );
            }
        }
        
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