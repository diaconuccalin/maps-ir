#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_IRCOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_IRCOLLECTOR_H

#include "Model.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::maps {

// Walks the MAPS module and fills the generated header model.  The collector
// deliberately stops at a layout-neutral representation; C syntax is handled by
// the emitter module.
LogicalResult collectMagiaHeaderModel(ModuleOp module,
                                      MagiaHeaderModel &model);

} // namespace mlir::maps

#endif
