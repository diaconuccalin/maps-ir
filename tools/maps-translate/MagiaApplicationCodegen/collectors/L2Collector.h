#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_L2COLLECTOR_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_L2COLLECTOR_H

#include "CollectorSupport.h"

namespace mlir::maps {

void collectL2Sources(Operation *regionOp,
                      llvm::StringMap<L2SourceInfo> &sources);
LogicalResult addL2ReadForSlice(MagiaApplicationDescription &model, unsigned sliceIndex,
                                const L2SourceInfo &source,
                                GeneratedGlobalKind globalKind,
                                unsigned numTokens, bool isInitializerRead);
void collectOutputInsertOffsets(
    Operation *runOp, llvm::StringMap<OutputInsertInfo> &outputInserts);
LogicalResult addOutputL2Write(MagiaApplicationDescription &model, StringRef globalName,
                               RankedTensorType outputGlobalType,
                               ArrayRef<int64_t> outputFullOffset,
                               unsigned srcSubsliceIndex, int64_t srcHartid);

} // namespace mlir::maps

#endif
