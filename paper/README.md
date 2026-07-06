# Blaze2D — Standalone Paper

A self-contained LaTeX source package describing **Blaze2D**.

## Build locally

Requires a TeX Live (or MiKTeX) installation with `lualatex`, `latexmk`, and `biber`. The figure PDFs are committed alongside the SVG sources, so no SVG converter is needed for a regular build.

```bash
cd paper
latexmk -lualatex main.tex          # or: make
```

The output is written to `build/main.pdf`.

The website serves a committed copy of the built PDF (CI does not build
LaTeX). After editing the paper, rebuild and re-copy it:

```bash
make && cp build/main.pdf ../web/public/paper/blaze2d.pdf
```

For continuous rebuilds on save:

```bash
make watch
```

## Regenerating figure PDFs from SVG

Only required if you edit one of the SVG sources. Uses
[`cairosvg`](https://cairosvg.org/) (a pure-Python converter — no
Inkscape required):

```bash
pip install cairosvg
make svg2pdf
```

## Build on Overleaf

1. Compress this folder into a `.zip`.
2. On Overleaf: *New Project → Upload Project → select the zip*.
3. In *Menu → Settings*, set **Compiler** to **LuaLaTeX** and
   **Main document** to `main.tex`.
4. Compile.

Overleaf needs no extra setup: `biber` and `lualatex` are part of the
default TeX Live distribution there, and the committed PDF figures
sidestep the SVG-conversion step.
