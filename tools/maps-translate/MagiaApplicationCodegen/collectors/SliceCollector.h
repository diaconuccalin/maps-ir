#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_SLICECOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_SLICECOLLECTOR_H

#include "CollectorSupport.h"

#include <limits>

namespace mlir::maps {

GeneratedSubslice makeSubsliceFromAttr(const GeneratedSlice &slice,
                                       ArrayRef<int64_t> sliceDims);
unsigned addWholeSubsliceForSlice(MagiaApplicationDescription &model,
                                  unsigned sliceIndex);
unsigned getOrCreateProducedSlice(
    MagiaApplicationDescription &model, llvm::StringMap<unsigned> &producedSliceIndices,
    StringRef uniqueKey, StringRef globalName, GeneratedGlobalKind kind,
    int64_t ownerHartid, RankedTensorType tensorType,
    ArrayRef<int64_t> fullOffset);

LogicalResult collectInitializerStore(
    Operation *storeOp, const llvm::StringMap<L2SourceInfo> &sources,
    MagiaApplicationDescription &model);
LogicalResult collectInputRecv(Operation *recvOp,
                               const llvm::StringMap<L2SourceInfo> &sources,
                               RunCollectionState &state,
                               MagiaApplicationDescription &model);
LogicalResult collectLoad(Operation *loadOp, RunCollectionState &state,
                          MagiaApplicationDescription &model);
LogicalResult collectExtractSliceAlias(Operation *op, RunCollectionState &state,
                                       MagiaApplicationDescription &model);
LogicalResult collectInsertSliceAlias(Operation *op, RunCollectionState &state,
                                      MagiaApplicationDescription &model);

} // namespace mlir::maps

#endif
