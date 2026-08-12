/*
Suppport operation file for collection from maps-ir

*/

#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_COLLECTORSUPPORT_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_COLLECTORSUPPORT_H

#include "../ApplicationDescription.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::maps {

struct L2SourceInfo {
  std::string globalName;
  RankedTensorType globalType;
  SmallVector<int64_t> fullOffset;
  bool isConstant = false;
  uint64_t packedImageOffset = 0;
  uint64_t packedByteSize = 0;
  std::string packedDtype;
  SmallVector<int64_t> packedShape;
  uint64_t requiredAlignment = kGeneratedL2AlignmentBytes;
};

struct ProducedSliceSource {
  Value rootValue;
  RankedTensorType rootType;
  SmallVector<int64_t> fullOffset;
};

struct TileRecvInfo {
  unsigned dstSliceIndex = 0;
  unsigned dstSubsliceIndex = 0;
  unsigned producerIdx = 0;
};

struct OutputInsertInfo {
  SmallVector<int64_t> fullOffset;
  RankedTensorType outputType;
};

struct RunCollectionState {
  llvm::StringMap<TileRecvInfo> tileRecvs;
  llvm::StringMap<unsigned> aggregateSliceIndices;
  llvm::StringMap<OutputInsertInfo> outputInserts;
  llvm::StringMap<unsigned> producedSliceIndices;
  llvm::DenseMap<Value, unsigned> producedValueIds;
  llvm::DenseMap<Value, unsigned> valueSubsliceIndices;
  unsigned nextProducedValueId = 0;
};

Operation *findParentNamed(Operation *op, StringRef name);
std::string getFlatSymbolAttr(Operation *op, StringRef name);
SmallVector<int64_t> getDenseI64Array(Operation *op, StringRef name);
std::string getGlobalNameForValue(Value value);
std::string getChannelBaseName(StringRef channel);
std::string getTileSlotKey(int64_t hartid, StringRef slot);
std::string getAggregateSliceKey(int64_t hartid, StringRef baseName,
                                 ArrayRef<int64_t> parentSlice);
SmallVector<int64_t> getPairwiseOffsets(ArrayRef<int64_t> values);
SmallVector<int64_t> getPairwiseLengths(ArrayRef<int64_t> values);
std::string getProducedValueKey(RunCollectionState &state, Value value);

} // namespace mlir::maps

#endif
