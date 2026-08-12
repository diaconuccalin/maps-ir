#ifndef MAPS_TOOLS_MAPS_TRANSLATE_EXECUTIONPLANIMPORTER_H
#define MAPS_TOOLS_MAPS_TRANSLATE_EXECUTIONPLANIMPORTER_H

#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
class Operation;

namespace maps {

OwningOpRef<Operation *> importExecutionPlan(llvm::StringRef input,
                                             MLIRContext *context);

} // namespace maps
} // namespace mlir

#endif
