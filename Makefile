MAPS_TRANSLATE ?= ./build/tools/maps-translate/maps-translate
OUTPUT_STEM ?= model
HEADER_NAME ?= $(OUTPUT_STEM)_generated
DATA_NAME ?= $(OUTPUT_STEM)_generated_data
NUM_TOKENS ?= 1
RANDOMIZE_INPUTS ?= 1

GENERATED_DIR ?= generated
EXECUTION_PLAN_JSON ?= examples/magia_example.execution-plan.json
EXECUTION_PLAN_MLIR ?= $(GENERATED_DIR)/magia_example.execution-plan.mlir
MAGIA_HEADER ?= $(GENERATED_DIR)/$(HEADER_NAME).h
MAGIA_DATA ?= $(GENERATED_DIR)/$(DATA_NAME).c
MAGIA_WEIGHTS_ASM ?= $(GENERATED_DIR)/$(OUTPUT_STEM)_weights.S
MAGIA_WEIGHTS ?= $(GENERATED_DIR)/$(OUTPUT_STEM).weights.bin
BUNDLE_WEIGHTS ?= $(basename $(EXECUTION_PLAN_JSON)).weights.bin
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
		$(EXECUTION_PLAN_JSON) -o /dev/null

maps-translate:
	cmake --build build --target maps-translate

maps-mlir: maps-translate
	mkdir -p $(dir $(EXECUTION_PLAN_MLIR))
	if [ ! -f $(EXECUTION_PLAN_MLIR) ] || [ $(EXECUTION_PLAN_JSON) -nt $(EXECUTION_PLAN_MLIR) ] || [ $(MAPS_TRANSLATE) -nt $(EXECUTION_PLAN_MLIR) ]; then \
		$(MAPS_TRANSLATE) --json-to-maps $(EXECUTION_PLAN_JSON) -o $(EXECUTION_PLAN_MLIR); \
	fi

magia-header: maps-mlir
	mkdir -p $(dir $(MAGIA_HEADER))
	$(MAPS_TRANSLATE) --maps-to-magia-header $(MAGIA_TRANSLATE_OPTS) $(EXECUTION_PLAN_MLIR) -o $(MAGIA_HEADER)

magia-data: maps-mlir
	mkdir -p $(dir $(MAGIA_DATA))
	$(MAPS_TRANSLATE) --maps-to-magia-data $(MAGIA_TRANSLATE_OPTS) $(EXECUTION_PLAN_MLIR) -o $(MAGIA_DATA)

magia-weights-asm: maps-mlir
	mkdir -p $(dir $(MAGIA_WEIGHTS_ASM))
	$(MAPS_TRANSLATE) --maps-to-magia-weights-asm $(MAGIA_TRANSLATE_OPTS) --maps-magia-weights-file=$(MAGIA_WEIGHTS) $(EXECUTION_PLAN_MLIR) -o $(MAGIA_WEIGHTS_ASM)

magia-weights:
	mkdir -p $(dir $(MAGIA_WEIGHTS))
	if ! cmp -s $(BUNDLE_WEIGHTS) $(MAGIA_WEIGHTS); then cp $(BUNDLE_WEIGHTS) $(MAGIA_WEIGHTS); fi

clean-generated:
	rm -f $(EXECUTION_PLAN_MLIR) $(MAGIA_HEADER) $(MAGIA_DATA) $(MAGIA_WEIGHTS_ASM) $(MAGIA_WEIGHTS)
