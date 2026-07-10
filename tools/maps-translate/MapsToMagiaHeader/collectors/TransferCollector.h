#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_TRANSFERCOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_TRANSFERCOLLECTOR_H

#include "CollectorSupport.h"

namespace mlir::maps {

LogicalResult collectTileRecvs(Operation *runOp, RunCollectionState &state,
                               MagiaHeaderModel &model);
LogicalResult collectProducedSend(Operation *sendOp, RunCollectionState &state,
                                  MagiaHeaderModel &model);

} // namespace mlir::maps

#endif
