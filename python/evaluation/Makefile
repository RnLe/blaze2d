# Evaluation harness for MPB vs. mpb2d runs.
# Usage:
#   make                         # runs default config (compare_hex_lowres)
#   make CONFIG=config/foo.mk run # run arbitrary config file
#   make foo                      # run config/foo.mk via its short name

CONFIG_DIR ?= config
DEFAULT_CONFIG := $(CONFIG_DIR)/compare_hex_lowres.mk
CONFIG ?= $(DEFAULT_CONFIG)

CONFIG_FILES := $(wildcard $(CONFIG_DIR)/*.mk)
ifeq ($(strip $(CONFIG_FILES)),)
  $(error No config files found under $(CONFIG_DIR))
endif

EVAL_TARGETS := $(patsubst $(CONFIG_DIR)/%.mk,%,$(CONFIG_FILES))

ifeq (,$(wildcard $(CONFIG)))
  $(error Config file '$(CONFIG)' not found. Override via CONFIG=path/to/config.mk)
endif

include $(CONFIG)

ifndef EVAL_NAME
  $(error EVAL_NAME must be defined inside $(CONFIG))
endif

.DEFAULT_GOAL := run

.PHONY: run mpb mpb2d list-config clean $(EVAL_TARGETS)

run: mpb mpb2d

$(EVAL_TARGETS):
	@echo "[evaluation] launching '$@' via $(CONFIG_DIR)/$@.mk"
	@$(MAKE) --no-print-directory mpb CONFIG=$(CONFIG_DIR)/$@.mk
	@$(MAKE) --no-print-directory mpb2d CONFIG=$(CONFIG_DIR)/$@.mk

mpb:
	@if [ -z "$(strip $(MPB_COMMAND))" ]; then \
		echo "MPB_COMMAND is empty in $(CONFIG)"; \
		exit 1; \
	fi
	@echo "[evaluation] running MPB for $(EVAL_NAME)"
	$(MPB_COMMAND)

mpb2d:
	@if [ -z "$(strip $(MPB2D_COMMAND))" ]; then \
		echo "MPB2D_COMMAND is empty in $(CONFIG)"; \
		exit 1; \
	fi
	@echo "[evaluation] running mpb2d for $(EVAL_NAME)"
	$(MPB2D_COMMAND)

list-config:
	@echo "Loaded evaluation: $(EVAL_NAME)"
	@echo "Config file: $(CONFIG)"
	@echo "MPB command: $(MPB_COMMAND)"
	@echo "mpb2d command: $(MPB2D_COMMAND)"

clean:
	@echo "Removing generated outputs under evaluation/reference-data"
	rm -f reference-data/*.json reference-data/*.csv
