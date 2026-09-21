# Blaze2D website

Next.js, React and MDX, exported as static HTML. Interactive calculations run in
Web Workers against the same Rust configuration contract that Python uses.

## Local development

Node.js 22+, pnpm 10, Rust and wasm-pack. From this directory:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

`pnpm dev` first runs `prepare:site`, which builds the WASM solver and
regenerates the contract, examples and content indexes; that is what makes the
first start slow. The dev server then watches `content/` and regenerates article
metadata, navigation and the search index as files change.

## Layout

```text
app/
  layout.tsx            <html>, the UI font, site metadata
  global.css            manifest: decides stylesheet order, holds no rules
  styles/               tokens, reset, base, shell, article, home, examples, charts
  (docs)/               the documentation shell
    (content)/          GENERATED route stubs, one per article
  examples/[slug]/      one example: intro plus an embedded Workbench
  workbench/            the full-page Workbench
  pitch/                the standalone landing page
components/
  site/                 shell, article frame, table of contents, footer
  article/              components MDX articles may use
  examples/             example library, code window
  workbench/            the browser calculation app
  charts/               benchmark charts for the technical report
content/                the articles themselves, as Markdown or MDX
lib/
  theme.ts              colours for graphics drawn from JavaScript
  compute/              worker protocol, execution controller, export
  contract/             GENERATED types from the Rust schema
scripts/                the generators listed below
```

Import with the `@/` alias (`@/components/site/DocsShell`) rather than relative
paths.

Files named `*.generated.*` and everything under `app/(docs)/(content)/` are
build output. Edit the source that produces them, then re-run the generator.

| Generated | From | Command |
| --- | --- | --- |
| `app/(docs)/(content)/`, `lib/posts.generated.ts`, `lib/navigation.generated.ts`, `lib/search.generated.json` | `content/` frontmatter | `pnpm generate:content` |
| `lib/examples/catalog.generated.ts`, `public/examples/` | `../examples/calculations/` | `pnpm generate:examples` |
| `lib/contract/generated.ts`, `lib/contract/schema.json` | the Rust config schema | `pnpm generate:contract` |
| `public/wasm-blaze/` | `crates/backend-wasm` | `pnpm build:wasm` |
| `public/data/epsilon-grids/` | the native CLI | `pnpm build:epsilon-grids` |

## Styling

Every colour, radius and measure is a token in `app/styles/tokens.css`. Nothing
else should contain a hex literal. Canvas cannot read CSS custom properties, so
graphics drawn from JavaScript read `lib/theme.ts` instead; the two files mirror
each other and must be changed together.

To re-theme the site, change `--accent*` in `tokens.css` and `accent` in
`theme.ts`.

`.article` is the only place bare HTML elements are styled, because MDX content
cannot carry class names. A component rendered inside an article therefore has to
set its own margins and link decoration rather than inherit prose defaults.

## Add an article

Add a Markdown or MDX file to `content/`. Its filename becomes its URL. Use
Markdown for prose, code, links and equations; use MDX when the article needs a
React figure or an interactive component.

```md
---
title: A new study
description: What this calculation investigates.
nav:
  group: Research
  order: 3
---

# A new study

Introduce the calculation, then describe its settings and results.
```

The title and description feed metadata and search. Headings generate the table
of contents. Navigation groups are `Use Blaze`, `Research` and `Project`, and
`order` must be unique within a group. Omit `nav` to leave an article out of
navigation, or set `draft: true` to exclude it from the build. A `card` supplies
navigation too, unless `nav: false` is set.

For a homepage card, copy the `card` block from an existing article. Sizes are
`1x1`, `2x1`, `1x2` and `2x2`; categories are `Use Blaze` and `Research`. Normal
articles use a readable measure, and `layout: wide` lets figures use more space.

Store images in `public/` and reference them with `/` paths. Internal links and
images pick up the deployment base path automatically. In React components, use
`next/link` for pages and `getAssetPath` for assets.

## Calculations and checks

Calculation examples live in `../examples/calculations/`. Their TOML, Python
source and lattice images are generated into the site together. Do not copy
numerical defaults into frontend code.

```sh
pnpm typecheck
pnpm lint
pnpm build:local     # static export served from the root
pnpm test:smoke      # eight essential Chromium checks for deployment
pnpm test:browser    # Playwright, all three browsers, against that export
```

`test:browser` lets Playwright start and stop the export server itself. While
iterating, leave one running instead and point the tests at it:

```sh
pnpm serve:test      # once, in its own terminal (port 3211)
pnpm test:quick      # chromium only, layout and site specs
```

`BLAZE_TEST_URL` makes `playwright.config.ts` skip its managed `webServer`
entirely, which is where the run-to-run variance lives: the eight layout and
site tests are about 33s of work, but a managed run has been seen to add over
two minutes of stall on top of them.

The production base path is `/blaze2d`, set in `next.config.ts`. Export from the
domain root with `NEXT_BASE_PATH=`, which is what `build:local` and `dev` do.

Pushes to `main` build the production `/blaze2d` export once, check types and
prose, and run eight `@smoke` tests in Chromium before publishing that same
artifact. These cover navigation, mobile reflow, static links, example redirects,
TOML validation/normalization, and real solver runs and exports.

Pull requests, package releases, and manual **Website checks** runs default to
the full suite in Chromium, Firefox, and WebKit for both base paths. To run this
before a larger direct-to-main release, use `gh workflow run site-checks.yml`.
The workflow caches Cargo dependencies, compiled outputs, and `wasm-pack`; it
still builds and verifies the solver's source revision on every run.

## Known follow-ups

- `eslint-config-next` 16 added the React Compiler rules
  `react-hooks/set-state-in-effect` and `react-hooks/refs`. They flag patterns
  that predate the upgrade and are correct as written, so they are configured as
  warnings; moving the Workbench to the compiler's model is its own task.
- ESLint stays on 9 and TypeScript on 5: `eslint-plugin-react` has not adopted
  ESLint 10, and typescript-eslint has not adopted TypeScript 7.
