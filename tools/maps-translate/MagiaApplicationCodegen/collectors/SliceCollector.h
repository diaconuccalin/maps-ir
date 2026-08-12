#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_SLICECOLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_SLICECOLLECTOR_H

#include "CollectorSupport.h"

#include <limits>

namespace mlir::maps {

GeneratedSubslice makeSubsliceFromAttr(const GeneratedSlice &slice,
                                       ArrayRef<int64_t> sliceDims);
unsigned addWholeSubsliceForSlice(MagiaApplicationDescription &application,
                                  unsigned sliceIndex);
unsigned getOrCreateProducedSlice(
    MagiaApplicationDescription &application, llvm::StringMap<unsigned> &producedSliceIndices,
    StringRef uniqueKey, StringRef globalName, GeneratedGlobalKind kind,
    int64_t ownerHartid, RankedTensorType tensorType,
    ArrayRef<int64_t> fullOffset);

LogicalResult collectInitializerStore(
    Operation *storeOp, const llvm::StringMap<L2SourceInfo> &sources,
    MagiaApplicationDescription &application);
LogicalResult collectInputRecv(Operation *recvOp,
                               const llvm::StringMap<L2SourceInfo> &sources,
                               RunCollectionState &state,
                               MagiaApplicationDescription &application);
LogicalResult collectLoad(Operation *loadOp, RunCollectionState &state,
                          MagiaApplicationDescription &application);
LogicalResult collectExtractSliceAlias(Operation *op, RunCollectionState &state,
                                       MagiaApplicationDescription &application);
LogicalResult collectInsertSliceAlias(Operation *op, RunCollectionState &state,
                                      MagiaApplicationDescription &application);

} // namespace mlir::maps

#endif
