#include "Model.h"
#include "collectors/CollectorSupport.h"

#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <utility>

namespace mlir::maps {

std::string normalizeSnakeCase(StringRef value) {
  std::string normalized;
  bool separator = false;
  bool previousWasLowerOrDigit = false;
  for (char character : value) {
    if (!llvm::isAlnum(character)) {
      separator = !normalized.empty();
      previousWasLowerOrDigit = false;
      continue;
    }
    bool uppercase = llvm::isUpper(character);
    if (!normalized.empty() &&
        (separator || (uppercase && previousWasLowerOrDigit)) &&
        normalized.back() != '_')
      normalized.push_back('_');
    normalized.push_back(llvm::toLower(character));
    separator = false;
    previousWasLowerOrDigit =
        llvm::isLower(character) || llvm::isDigit(character);
  }
  while (!normalized.empty() && normalized.back() == '_')
    normalized.pop_back();
  if (normalized.empty())
    return "application";
  if (llvm::isDigit(normalized.front()))
    normalized.insert(0, "application_");
  return normalized;
}

static unsigned getOrCreateGlobalId(MagiaHeaderModel &model,
                                    StringRef globalName) {
  auto existing = model.globalIds.find(globalName);
  if (existing != model.globalIds.end())
    return existing->second;

  unsigned globalId = model.nextGlobalId++;
  model.globalIds[globalName] = globalId;
  return globalId;
}

static unsigned allocateSliceIdForHart(MagiaHeaderModel &model,
                                       int64_t hartid) {
  unsigned sliceId = model.nextSliceIdByHart[hartid];
  model.nextSliceIdByHart[hartid] = sliceId + 1;
  return sliceId;
}


unsigned addGeneratedSlice(MagiaHeaderModel &model, GeneratedSlice slice) {
  /*
  Add slice to plan
  */
  slice.modelIndex = model.slices.size();
  slice.globalId = getOrCreateGlobalId(model, slice.globalName);
  slice.sliceId = allocateSliceIdForHart(model, slice.ownerHartid);
  model.slices.push_back(std::move(slice));
  return model.slices.back().modelIndex;
}


unsigned addGeneratedSubslice(MagiaHeaderModel &model, GeneratedSubslice subslice) {
  /*
  Add subslice to plan
  */
  subslice.modelIndex = model.subslices.size();
  model.subslices.push_back(std::move(subslice));
  return model.subslices.back().modelIndex;
}


GeneratedSubslice makeWholeSubslice(const GeneratedSlice &slice) {
  /*
  Create a subslice consisting of the full slice. Used for L2 to L1 transfers for example
  */

  GeneratedSubslice subslice;
  subslice.sliceIndex = slice.modelIndex;
  subslice.offset.assign(slice.shape.size(), 0);
  subslice.shape = slice.shape;
  subslice.elementType = slice.elementType;
  subslice.elemBytes = slice.elemBytes;
  subslice.stridesBytes = slice.stridesBytes;
  return subslice;
}

uint64_t computeOffsetBytes(ArrayRef<int64_t> offsets,
                            ArrayRef<uint64_t> stridesBytes) {
  uint64_t result = 0;
  for (size_t i = 0; i < offsets.size() && i < stridesBytes.size(); ++i)
    result += static_cast<uint64_t>(offsets[i]) * stridesBytes[i];
  return result;
}

unsigned addGeneratedL2Read(MagiaHeaderModel &model, GeneratedL2Read read) {
  /*
  Add L2 read descriptor to plan
  */

  read.modelIndex = model.l2Reads.size();
  model.l2Reads.push_back(read);
  return model.l2Reads.back().modelIndex;

}


unsigned getOrCreateTransitionId(MagiaHeaderModel &model, StringRef channel) {
  auto existing = model.transitionIds.find(channel);
  if (existing != model.transitionIds.end())
    return existing->second;

  unsigned transitionId = model.nextTransitionId++;
  model.transitionIds[channel] = transitionId;
  return transitionId;
}

unsigned addGeneratedRecv(MagiaHeaderModel &model, GeneratedRecv recv) {
  recv.modelIndex = model.recvs.size();
  model.recvs.push_back(recv);
  return model.recvs.back().modelIndex;
}

unsigned addGeneratedSend(MagiaHeaderModel &model, GeneratedSend send) {
  send.modelIndex = model.sends.size();
  model.sends.push_back(send);
  return model.sends.back().modelIndex;
}

unsigned addGeneratedOp(MagiaHeaderModel &model, GeneratedOp op) {
  op.modelIndex = model.ops.size();
  model.ops.push_back(op);
  return model.ops.back().modelIndex;
}

unsigned addGeneratedL2Write(MagiaHeaderModel &model, GeneratedL2Write write) {
  write.modelIndex = model.l2Writes.size();
  model.l2Writes.push_back(write);
  return model.l2Writes.back().modelIndex;
}

static std::string getL2GlobalKey(GeneratedGlobalKind globalKind,
                                  StringRef globalName) {
  return (Twine(static_cast<int>(globalKind)) + ":" + globalName).str();
}


void noteTile(MagiaHeaderModel &model, int64_t hartid) {
  /*
  Add tile to plan
  */

  auto &plan = model.tilePlans[hartid];
  plan.hartid = hartid;
}


uint64_t alignTo(uint64_t value, uint64_t alignment) {
  /*
  Consider L1 allignment
  */
  return ((value + alignment - 1) / alignment) * alignment;
}


uint64_t product(ArrayRef<int64_t> values) {
  uint64_t result = 1;
  for (int64_t value : values)
    result *= static_cast<uint64_t>(value);
  return result;
}


uint64_t computeSlotBytes(const GeneratedSlice &slice) {
  /*
  number of bytes for a slot
  */
  return product(slice.shape) * slice.elemBytes;
}


unsigned getElementByteWidth(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16())
    return 2;
  if (type.isInteger(8))
    return 1;
  return 0;
}


SmallVector<uint64_t> computeContiguousStridesBytes(ArrayRef<int64_t> shape, unsigned elemBytes) {
  /*
  Returns the number of bytes per strided dimension considering the last dimension as the contiguous dimension

  e.g.
  shape = [3, 4, 5]
  elemBytes = 2

  dim 2 stride = 2
  dim 1 stride = 5 * 2 = 10
  dim 0 stride = 4 * 10 = 40

  result = [40, 10, 2]

  */
  SmallVector<uint64_t> strides(shape.size(), elemBytes);
  if (shape.empty())
    return strides;

  for (int64_t dim = static_cast<int64_t>(shape.size()) - 2; dim >= 0; --dim)
    strides[dim] = strides[dim + 1] * static_cast<uint64_t>(shape[dim + 1]);

  return strides;
}


LogicalResult getOrCreateL2Global(MagiaHeaderModel &model,
                                  StringRef globalName,
                                  GeneratedGlobalKind globalKind,
                                  RankedTensorType tensorType,
                                  unsigned numTokens,
                                  unsigned &l2GlobalIndex,
                                  const L2SourceInfo *sourceInfo) {

  /*
  check if an L2 allocation exists for the current global tensor. If not, create it
  */
  
  // build a unique key (for example, Input:arg1)
  std::string key = getL2GlobalKey(globalKind, globalName);
  
  // check if key exists, if it already exists, use it
  auto existing = model.l2GlobalIndices.find(key);
  if (existing != model.l2GlobalIndices.end()) {
    l2GlobalIndex = existing->second;
    return success();
  }

  // else
  // get bytes per element
  unsigned elemBytes = getElementByteWidth(tensorType.getElementType());

  // Create the global allocation
  GeneratedL2Global global;
  global.modelIndex = model.l2Globals.size();
  global.globalId = getOrCreateGlobalId(model, globalName);
  global.globalName = globalName.str();
  global.globalKind = globalKind;
  global.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
  global.elementType = tensorType.getElementType();
  global.elemBytes = elemBytes;
  global.tensorBytes = product(global.shape) * elemBytes;
  global.tokenStrideBytes = alignTo(global.tensorBytes, kGeneratedL2AlignmentBytes);
  global.numTokens = numTokens;
  global.stridesBytes = computeContiguousStridesBytes(global.shape, global.elemBytes);

  if (globalKind == GeneratedGlobalKind::Initializer && sourceInfo &&
      sourceInfo->isConstant) {
    global.isConstant = true;
    global.packedImageOffset = sourceInfo->packedImageOffset;
    global.packedByteSize = sourceInfo->packedByteSize;
    global.packedDtype = sourceInfo->packedDtype;
    global.packedShape = sourceInfo->packedShape;
    global.requiredAlignment = model.weightsAlignment;

    if (!model.hasPackedWeights ||
        global.packedImageOffset % global.requiredAlignment != 0 ||
        global.packedImageOffset + global.packedByteSize > model.weightsSize ||
        global.packedShape != global.shape ||
        global.packedByteSize != global.tensorBytes)
      return failure();

    global.l2Address = model.l2Base + global.packedImageOffset;
    model.l2GlobalIndices[key] = global.modelIndex;
    model.l2Globals.push_back(std::move(global));
    l2GlobalIndex = model.l2Globals.back().modelIndex;
    return success();
  }

  if (globalKind == GeneratedGlobalKind::Initializer &&
      (model.hasPackedWeights || !model.randomizeInputs))
    return failure();

  // allign
  model.nextL2Address = alignTo(model.nextL2Address,
                                kGeneratedL2AlignmentBytes);
  global.l2Address = model.nextL2Address;

  // Return error if memory is exceeded
  uint64_t allocationBytes = global.tokenStrideBytes * global.numTokens;
  if (global.l2Address + allocationBytes > model.l2End)
    return failure();

  // set starting adress for next allocation
  model.nextL2Address = global.l2Address + allocationBytes;

  // Advance the allocator
  model.l2GlobalIndices[key] = global.modelIndex;
  model.l2Globals.push_back(std::move(global));
  l2GlobalIndex = model.l2Globals.back().modelIndex;

  return success();

}


LogicalResult allocateSliceInTileL1(MagiaHeaderModel &model, unsigned sliceIndex) {
  /*
  Allocates space for a tile slice in tile's L1
  */
  GeneratedSlice &slice = model.slices[sliceIndex];

  noteTile(model, slice.ownerHartid);
  GeneratedTilePlan &plan = model.tilePlans[slice.ownerHartid];

  // Get the amount of bytes the slice needs
  slice.slotBytes = computeSlotBytes(slice);

  // Consider L1 allignment
  plan.nextL1OffsetBytes = alignTo(plan.nextL1OffsetBytes, kGeneratedL1AlignmentBytes);

  // get the adress for next tile's starting point
  slice.l1OffsetBytes = plan.nextL1OffsetBytes;

  // Guard for out-of-memory
  uint64_t allocationBytes = slice.slotBytes * slice.numSlots;
  uint64_t endOffset = slice.l1OffsetBytes + allocationBytes;
  if (plan.l1DataBaseOffset + endOffset > kGeneratedReadyFlagsOffset)
    return failure();

  // Populate plan with info derived
  plan.nextL1OffsetBytes = endOffset;
  plan.l1DataBytes = std::max(plan.l1DataBytes, endOffset);
  plan.sliceIndices.push_back(sliceIndex);
  return success();

}

} // namespace mlir::maps
