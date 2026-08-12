PLAN_IMPORT ?= ./build/tools/maps-translate/maps-plan-import
CODEGEN ?= ./build/tools/maps-translate/maps-codegen

GENERATED_DIR ?= generated
APPLICATION_NAME ?= magia_example
EXECUTION_PLAN ?= examples/magia_example.execution-plan.json
INITIALIZERS ?= examples/magia_example.execution-plan.weights.bin
MAPS_MLIR ?= $(GENERATED_DIR)/$(APPLICATION_NAME).mlir
APPLICATION ?= $(GENERATED_DIR)/$(APPLICATION_NAME)

.PHONY: all application import-plan tools clean-generated

all: application

application: import-plan tools
	$(CODEGEN) $(MAPS_MLIR) \
		--target=magia-v2 \
		--maps-magia-output-stem=$(APPLICATION_NAME) \
		--maps-magia-weights-file=$(INITIALIZERS) \
		-o $(APPLICATION)

import-plan: tools
	mkdir -p $(dir $(MAPS_MLIR))
	$(PLAN_IMPORT) $(EXECUTION_PLAN) -o $(MAPS_MLIR)

tools:
	cmake --build build --target maps-plan-import maps-codegen

clean-generated:
	rm -f $(MAPS_MLIR)
