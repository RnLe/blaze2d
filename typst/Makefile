# Build the Typst reports.
#
#   make                       build every report
#   make blaze                 build one report -> out/blaze2d-technical-report.pdf
#   make figures REPORT=blaze  re-export that report's charts from the website
#   make fonts                 rebuild assets/fonts from web/public/fonts
#   make watch REPORT=blaze    rebuild on save
#   make clean                 remove out/
#
# Figures and fonts are committed, so a plain `make` needs nothing but Typst.

TYPST   ?= typst
PYTHON  ?= python3
NODE    ?= node
REPORT  ?= blaze

FONTS   := assets/fonts
EXPORT  := tools/figure-export
OUT     := out

# Every reports/<name>/main.typ is a report.
REPORTS := $(patsubst reports/%/main.typ,%,$(wildcard reports/*/main.typ))

# Output filename per report; defaults to the report's own name.
PDF_blaze := blaze2d-technical-report

pdf_name = $(if $(PDF_$(1)),$(PDF_$(1)),$(1))
pdf_path = $(OUT)/$(call pdf_name,$(1)).pdf

# The Typst root is the repository root, matching what an editor extension uses
# when the workspace is opened at the repo. Shared files are therefore addressed
# as /typst/lib/..., /typst/assets/... and /typst/references.bib, which resolve
# the same from the CLI and from the editor. --font-path supplies the faces the
# exported chart SVGs reference by name.
ROOT        := ..
TYPST_FLAGS := --root $(ROOT) --font-path $(FONTS)

.PHONY: all $(REPORTS) figures fonts watch clean help

all: $(REPORTS)

## Build one report.
$(REPORTS): %: $(OUT)
	@echo "==> $* -> $(call pdf_path,$*)"
	@$(TYPST) compile $(TYPST_FLAGS) reports/$*/main.typ $(call pdf_path,$*)

$(OUT):
	@mkdir -p $(OUT)

## Re-export a report's charts from the website's React components.
figures:
	@echo "==> exporting figures for $(REPORT)"
	@$(NODE) $(EXPORT)/index.mjs $(REPORT)

## Rebuild the extended OpenAI Sans faces the chart SVGs need.
fonts:
	@echo "==> building fonts"
	@$(PYTHON) $(EXPORT)/build-fonts.py

## Rebuild a report whenever a source file changes.
watch: $(OUT)
	@$(TYPST) watch $(TYPST_FLAGS) reports/$(REPORT)/main.typ $(call pdf_path,$(REPORT))

clean:
	@rm -rf $(OUT)

help:
	@echo "reports: $(REPORTS)"
	@echo "targets: all <report> figures fonts watch clean"
