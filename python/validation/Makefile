CARGO ?= cargo
VALIDATION_BIN ?= mpb2d-validation
ARTIFACT_ROOT := $(CURDIR)/artifacts
LATTICE_DIR := $(ARTIFACT_ROOT)/lattice_data
GEOMETRY_DIR := $(ARTIFACT_ROOT)/geometry_data
SMOOTHING_DIR := $(ARTIFACT_ROOT)/smoothing_data
KSPACE_DIR := $(ARTIFACT_ROOT)/kspace_data
PRECOMPUTE_DIR := $(ARTIFACT_ROOT)/precompute_data
RECIPROCAL_DIR := $(ARTIFACT_ROOT)/reciprocal_data
MK_DIR := $(CURDIR)/mk

include $(MK_DIR)/lattice.mk
include $(MK_DIR)/geometry.mk
include $(MK_DIR)/smoothing.mk
include $(MK_DIR)/kspace.mk
include $(MK_DIR)/precompute.mk
include $(MK_DIR)/reciprocal.mk

.PHONY: all clean

all: lattice-data geometry-data smoothing-data kspace-data precompute-data reciprocal-data

clean:
	rm -rf $(ARTIFACT_ROOT)
