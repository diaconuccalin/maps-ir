MAPS_TRANSLATE ?= ./build/tools/maps-translate/maps-translate
HEADER_NAME ?= magia_example
DATA_NAME ?= $(HEADER_NAME)_data
NUM_TOKENS ?= 1
NUM_TOKEN_SLOTS ?= 2

GENERATED_DIR ?= generated
PIPELINE_JSON ?= examples/magia_example.pipeline.json
PIPELINE_MLIR ?= $(GENERATED_DIR)/magia_example.pipeline.mlir
MAGIA_HEADER ?= $(GENERATED_DIR)/$(HEADER_NAME).h
MAGIA_DATA ?= $(GENERATED_DIR)/$(DATA_NAME).c
MAGIA_TRANSLATE_OPTS := --maps-magia-num-tokens=$(NUM_TOKENS) --maps-magia-num-token-slots=$(NUM_TOKEN_SLOTS)

.PHONY: all magia-example maps-translate maps-mlir magia-header magia-data clean-generated

all: magia-example

magia-example: magia-header magia-data

maps-translate:
	cmake --build build --target maps-translate

maps-mlir: maps-translate
	$(MAPS_TRANSLATE) --json-to-maps $(PIPELINE_JSON) -o $(PIPELINE_MLIR)

magia-header: maps-mlir
	$(MAPS_TRANSLATE) --maps-to-magia-header $(MAGIA_TRANSLATE_OPTS) $(PIPELINE_MLIR) -o $(MAGIA_HEADER)

magia-data: maps-mlir
	$(MAPS_TRANSLATE) --maps-to-magia-data $(MAGIA_TRANSLATE_OPTS) $(PIPELINE_MLIR) -o $(MAGIA_DATA)

clean-generated:
	rm -f $(PIPELINE_MLIR) $(MAGIA_HEADER) $(MAGIA_DATA)
