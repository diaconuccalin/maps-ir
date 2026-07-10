#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_SLICECOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_SLICECOLLECTOR_H

#include "CollectorSupport.h"

#include <limits>

namespace mlir::maps {

GeneratedSubslice makeSubsliceFromAttr(const GeneratedSlice &slice,
                                       ArrayRef<int64_t> sliceDims);
unsigned addWholeSubsliceForSlice(MagiaHeaderModel &model,
                                  unsigned sliceIndex);
unsigned getOrCreateProducedSlice(
    MagiaHeaderModel &model, llvm::StringMap<unsigned> &producedSliceIndices,
    StringRef uniqueKey, StringRef globalName, GeneratedGlobalKind kind,
    int64_t ownerHartid, RankedTensorType tensorType,
    ArrayRef<int64_t> fullOffset);

LogicalResult collectInitializerStore(
    Operation *storeOp, const llvm::StringMap<L2SourceInfo> &sources,
    MagiaHeaderModel &model);
LogicalResult collectInputRecv(Operation *recvOp,
                               const llvm::StringMap<L2SourceInfo> &sources,
                               RunCollectionState &state,
                               MagiaHeaderModel &model);
LogicalResult collectLoad(Operation *loadOp, RunCollectionState &state,
                          MagiaHeaderModel &model);
LogicalResult collectExtractSliceAlias(Operation *op, RunCollectionState &state,
                                       MagiaHeaderModel &model);
LogicalResult collectInsertSliceAlias(Operation *op, RunCollectionState &state,
                                      MagiaHeaderModel &model);

} // namespace mlir::maps

#endif
