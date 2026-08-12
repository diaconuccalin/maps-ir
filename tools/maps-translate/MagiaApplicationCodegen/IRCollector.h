#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_IRCOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_IRCOLLECTOR_H

#include "ApplicationDescription.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::maps {

// Walks the maps module and fills the MAGIA Application description. The collector
// deliberately stops at a layout-neutral representation; C syntax is handled by
// the emitter module.
LogicalResult collectMagiaApplicationDescription(ModuleOp module,
                                                  MagiaApplicationDescription &application);

} // namespace mlir::maps

#endif
