#ifndef MAPS_TOOLS_MAPS_TRANSLATE_JSONTOMAPS_H
#define MAPS_TOOLS_MAPS_TRANSLATE_JSONTOMAPS_H

#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
class Operation;

namespace maps {

OwningOpRef<Operation *> importJsonToMaps(llvm::StringRef input,
                                          MLIRContext *context);
void registerJsonToMapsTranslation();

} // namespace maps
} // namespace mlir

#endif
