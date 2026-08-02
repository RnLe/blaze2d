# Typst reports

PDF editions of the posts on the [Blaze2D website](https://rnle.github.io/blaze2d/).
Each report mirrors one page of `web/content/`, keeps the post's prose, and
reuses the plots the site itself renders.

```bash
make                        # build every report
make blaze                  # -> out/blaze2d-technical-report.pdf
make watch REPORT=blaze     # rebuild on save
```

A plain `make` needs only Typst: the figures and fonts are committed.

The Typst root is the **repository** root, not this directory, which is what an
editor extension uses when the workspace is opened at the repo. Shared files are
therefore addressed as `/typst/lib/...`, `/typst/assets/...` and
`/typst/references.bib`, so the same source compiles from the CLI and from the
editor without a per-tool root setting.

## Layout

```
lib/            shared template: theme.typ, macros.typ, report.typ
assets/         fonts and logos shared by every report
references.bib  shared bibliography (Typst prints only what is cited)
reports/<slug>/ one report per website page
tools/          the figure exporter (see tools/figure-export/README.md)
out/            build output, gitignored
```

`reports/` mirrors `web/content/` one to one, so `content/blaze.mdx` is
`reports/blaze/` and a future `content/architecture/eigensolver.mdx` would be
`reports/architecture/eigensolver/`.

## Two things this pipeline is deliberate about

**The prose is written by hand.** Nothing converts MDX to Typst. A report is
curated: the post's text is carried over, its figure references become Typst
cross-references, and the page's note boxes become callouts. The value is in the
editing pass, so there is no generator to fight with.

**The plots are not.** Redrawing the site's charts in Typst would guarantee
drift. Instead `tools/figure-export` renders the website's own React chart
components with `react-dom/server` and rewrites the output into standalone SVGs.
The figures are therefore identical to the page by construction, and re-running
the export picks up any change to the components or the benchmark data.

```bash
make figures REPORT=blaze   # re-export after the site's charts or data change
make fonts                  # only after the site's web fonts change
```

`make fonts` needs `fontTools` and `brotli`; point `PYTHON` at an environment
that has them, e.g. `make fonts PYTHON=~/miniforge3/bin/python`.

## Style

The document style follows the master thesis (`../../master-thesis/`) so a
report and the thesis chapter on the same subject read as one body of work:
the same colour palette and table treatment from its `StyleGuide.md`, the same
chapter banners and fading margin stripes, the same cover geometry, and the
prose conventions in its `WritingStyleGuide.md`.

The charts are the deliberate exception. They keep the website's own look,
white on black in OpenAI Sans, and sit on the page as rounded black cards rather
than being recoloured to the thesis palette.

## Adding a report

1. `mkdir -p reports/<slug>/sections reports/<slug>/figures`.
2. Copy `reports/blaze/figures.typ` across; it only defines the two figure
   helpers, and it must live next to the figures because Typst resolves an
   `image()` path against the file the literal is written in.
3. Write `reports/<slug>/main.typ`: `#show: report.with(...)` for the cover,
   imprint, contents and abstract, then `#include` one file per section. Import
   shared modules by their root-absolute path (`/typst/lib/report.typ`); only
   `figures.typ` is reached relatively, from the sections that use it.
4. Add `tools/figure-export/manifests/<slug>.mjs` listing each chart component
   and the exact props the MDX passes to it, then `make figures REPORT=<slug>`.
5. Optionally add `PDF_<slug> := <output-name>` to the `Makefile` to control the
   PDF filename; it defaults to the report's slug.

Cite with `@key` against the shared `references.bib`, and add any entry the post
needs that is not there yet.
