# Build the standalone Blaze2D paper.
#
# Targets:
#   make         -> build PDF into build/main.pdf
#   make watch   -> latexmk -pvc continuous rebuild
#   make svg2pdf -> regenerate figures/*.pdf from figures/*.svg (cairosvg)
#   make clean   -> remove build artefacts
#   make distclean -> clean + remove generated PDF figures

ENGINE   := lualatex
MAIN     := main.tex
BUILDDIR := build

SVGS := $(wildcard figures/*.svg)
PDFS := $(SVGS:.svg=.pdf)

.PHONY: all watch svg2pdf clean distclean

all: $(BUILDDIR)/main.pdf

$(BUILDDIR)/main.pdf: $(MAIN) preamble.tex metadata.tex sections/introduction.tex sections/blaze2d.tex sections/conclusion.tex references.bib | $(PDFS)
	latexmk -$(ENGINE) $(MAIN)

watch:
	latexmk -$(ENGINE) -pvc $(MAIN)

svg2pdf: $(PDFS)

figures/%.pdf: figures/%.svg
	cairosvg "$<" -o "$@"

clean:
	latexmk -C || true
	rm -rf $(BUILDDIR)

distclean: clean
	rm -f figures/*.pdf
