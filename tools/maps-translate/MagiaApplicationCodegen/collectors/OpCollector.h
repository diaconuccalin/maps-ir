#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_OPCOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_OPCOLLECTOR_H

#include "CollectorSupport.h"

namespace mlir::maps {

LogicalResult collectComputeOp(Operation *op, RunCollectionState &state,
                               MagiaApplicationDescription &model);

} // namespace mlir::maps

#endif
