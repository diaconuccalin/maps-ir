#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_CODEGENERATION_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_CODEGENERATION_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;

namespace maps {

LogicalResult generateMagiaApplication(ModuleOp module,
                                       llvm::StringRef outputDirectory,
                                       llvm::StringRef target);

} // namespace maps
} // namespace mlir

#endif
