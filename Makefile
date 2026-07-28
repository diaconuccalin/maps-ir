MAPS_TRANSLATE ?= ./build/tools/maps-translate/maps-translate
OUTPUT_STEM ?= model
HEADER_NAME ?= $(OUTPUT_STEM)_generated
DATA_NAME ?= $(OUTPUT_STEM)_generated_data
NUM_TOKENS ?= 1
RANDOMIZE_INPUTS ?= 1

GENERATED_DIR ?= generated
PIPELINE_JSON ?= examples/magia_example.pipeline.json
PIPELINE_MLIR ?= $(GENERATED_DIR)/magia_example.pipeline.mlir
MAGIA_HEADER ?= $(GENERATED_DIR)/$(HEADER_NAME).h
MAGIA_DATA ?= $(GENERATED_DIR)/$(DATA_NAME).c
MAGIA_WEIGHTS_ASM ?= $(GENERATED_DIR)/$(OUTPUT_STEM)_weights.S
MAGIA_WEIGHTS ?= $(GENERATED_DIR)/$(OUTPUT_STEM).weights.bin
BUNDLE_WEIGHTS ?= $(basename $(PIPELINE_JSON)).weights.bin
MAGIA_PACKAGE_DIR ?= $(GENERATED_DIR)/$(OUTPUT_STEM).runtime
MAGIA_TRANSLATE_OPTS = --maps-magia-num-tokens=$(NUM_TOKENS) --maps-magia-output-stem=$(OUTPUT_STEM) $(if $(filter 1,$(RANDOMIZE_INPUTS)),--maps-magia-test-randomize-inputs,)

.PHONY: all magia-example magia-bundle magia-package maps-translate maps-mlir magia-header magia-data magia-weights-asm magia-weights clean-generated

all: magia-example

magia-example: OUTPUT_STEM = magia_example
magia-example: MAGIA_HEADER = $(GENERATED_DIR)/magia_example.h
magia-example: MAGIA_DATA = $(GENERATED_DIR)/magia_example_data.c
magia-example: magia-header magia-data

magia-bundle: RANDOMIZE_INPUTS = 0
magia-bundle: magia-header magia-data magia-weights-asm magia-weights

magia-package: RANDOMIZE_INPUTS = 0
magia-package: maps-translate
	$(MAPS_TRANSLATE) --json-to-magia-package $(MAGIA_TRANSLATE_OPTS) \
		--maps-magia-package-dir=$(MAGIA_PACKAGE_DIR) \
		--maps-magia-weights-file=$(BUNDLE_WEIGHTS) \
		$(PIPELINE_JSON) -o /dev/null

maps-translate:
	cmake --build build --target maps-translate

maps-mlir: maps-translate
	mkdir -p $(dir $(PIPELINE_MLIR))
	if [ ! -f $(PIPELINE_MLIR) ] || [ $(PIPELINE_JSON) -nt $(PIPELINE_MLIR) ] || [ $(MAPS_TRANSLATE) -nt $(PIPELINE_MLIR) ]; then \
		$(MAPS_TRANSLATE) --json-to-maps $(PIPELINE_JSON) -o $(PIPELINE_MLIR); \
	fi

magia-header: maps-mlir
	mkdir -p $(dir $(MAGIA_HEADER))
	$(MAPS_TRANSLATE) --maps-to-magia-header $(MAGIA_TRANSLATE_OPTS) $(PIPELINE_MLIR) -o $(MAGIA_HEADER)

magia-data: maps-mlir
	mkdir -p $(dir $(MAGIA_DATA))
	$(MAPS_TRANSLATE) --maps-to-magia-data $(MAGIA_TRANSLATE_OPTS) $(PIPELINE_MLIR) -o $(MAGIA_DATA)

magia-weights-asm: maps-mlir
	mkdir -p $(dir $(MAGIA_WEIGHTS_ASM))
	$(MAPS_TRANSLATE) --maps-to-magia-weights-asm $(MAGIA_TRANSLATE_OPTS) --maps-magia-weights-file=$(MAGIA_WEIGHTS) $(PIPELINE_MLIR) -o $(MAGIA_WEIGHTS_ASM)

magia-weights:
	mkdir -p $(dir $(MAGIA_WEIGHTS))
	if ! cmp -s $(BUNDLE_WEIGHTS) $(MAGIA_WEIGHTS); then cp $(BUNDLE_WEIGHTS) $(MAGIA_WEIGHTS); fi

clean-generated:
	rm -f $(PIPELINE_MLIR) $(MAGIA_HEADER) $(MAGIA_DATA) $(MAGIA_WEIGHTS_ASM) $(MAGIA_WEIGHTS)
