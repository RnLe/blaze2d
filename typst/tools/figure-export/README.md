# Figure export

Turns the website's chart components into standalone SVGs for the Typst reports.

```bash
node index.mjs blaze          # or: make figures REPORT=blaze  (from typst/)
```

Output lands in `typst/reports/<report>/figures/`, one `.svg` per entry in
`manifests/<report>.mjs`.

## Why it works this way

The charts in `web/components/charts/` are the single source of truth for how a
plot looks. Re-drawing them in Typst would drift from the site the moment either
side changed. Instead this renders those very components with
`react-dom/server`, so the exported figure is identical to the page by
construction, and re-exports pick up any change to the components or the
benchmark data automatically.

That is possible because the charts are pure `@visx` SVG: no canvas, no DOM
measurement, no tooltips. Only three modules assume a browser, and each has a
synchronous stand-in in `shims/`:

| Module | Shim | Why |
| --- | --- | --- |
| `lib/use-benchmarks.ts` | `shims/use-benchmarks.ts` | The real hooks start from the embedded `FALLBACK_*` data and swap in `public/data/*.json` from a `useEffect` fetch. That effect never runs server-side, so the untouched hooks would export the fallback numbers. |
| `lib/paths.ts` | `shims/paths.ts` | `getAssetPath` prepends the GitHub-Pages base path; the exporter reads from disk. |
| `swr` | `shims/swr-shim.ts` | `BandComparisonChart` and `DeviationBoxPlotChart` fetch through `useSWR`. |

The interactive `EpsilonGridViewer` has no print form and is the one component
not rendered through React: `epsilon-grids.mjs` redraws its permittivity maps
for a fixed set of resolutions, reusing the viewer's own colour ramp.

## How it runs

The tool lives under `typst/` so all report tooling stays in one place, but it
executes inside `web/`, where `node_modules` is. `index.mjs` writes a staging
tree to `web/.figure-export/` (gitignored), compiles it with the TypeScript
already installed there, and runs the result. Nothing outside that directory is
touched, and the chart components are copied verbatim apart from one rewritten
`swr` import.

## Post-processing

`postprocess.mjs` rewrites the rendered markup into a self-contained SVG:

- **Multi-panel components** (`MemoryUsageChart`, `ThroughputScalingChart`, the
  TM/TE pairs, ...) render two `<svg>` roots inside a `display:flex; gap:20px`
  wrapper. They are composed side by side with that same gap.
- **Nested `<svg>` wrappers** are flattened to `<g transform>`. `@visx/text`
  wraps every label in `<svg x y style="overflow:visible">`; browsers honour
  `overflow:visible`, Typst's renderer clips nested viewports instead, which
  would erase every y-axis tick label (they are anchored `end` at `x=0` and so
  sit left of their own viewport). `em`-valued offsets are resolved against the
  wrapper's font-size during the rewrite.
- **The font stack** is written in explicitly, since `--font-sans` is never
  defined and the site's chart text simply inherits `<body>`'s stack.
- **The page background** (`#000000`) and 16px of padding are baked in, so the
  figure is complete on its own.

## Fonts

`build-fonts.py` produces `typst/assets/fonts/` from `web/public/fonts/*.woff2`:

```bash
/home/renlephy/miniforge3/bin/python build-fonts.py    # needs fontTools + brotli
```

It unwraps WOFF2 (which Typst cannot read) and extends the faces with the Greek
and mathematical glyphs OpenAI Sans lacks but the axis labels need. That second
step is not cosmetic: Typst renders SVG through resvg, which resolves one font
per `<text>` element rather than per glyph, so a single uncovered character
drags an entire label onto a fallback serif. Donor glyphs come from Roboto,
which already heads the site's own fallback chain, with DejaVu Sans for the
mathematical operators Roboto lacks. The result is published as a separate
family, `OpenAI Sans Extended`, which the exporter lists ahead of plain
`OpenAI Sans`, so a browser without it renders exactly like the page.

`index.mjs` audits every exported label against that face and warns if a
character has no glyph; extend `EXTRA_RANGES` in `build-fonts.py` and rebuild.

## Adding a figure

Add an entry to `manifests/<report>.mjs` with the component name and the exact
props the MDX passes, then re-run. Keeping the props in the manifest is what
makes a figure reproducible after the page changes.
