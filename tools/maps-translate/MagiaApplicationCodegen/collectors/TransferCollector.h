#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_TRANSFERCOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_TRANSFERCOLLECTOR_H

#include "CollectorSupport.h"

namespace mlir::maps {

LogicalResult collectTileRecvs(Operation *runOp, RunCollectionState &state,
                               MagiaApplicationDescription &application);
LogicalResult collectProducedSend(Operation *sendOp, RunCollectionState &state,
                                  MagiaApplicationDescription &application);

} // namespace mlir::maps

#endif
