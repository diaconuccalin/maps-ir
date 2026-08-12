#include "SliceCollector.h"

#include "L2Collector.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <utility>


namespace mlir::maps {


GeneratedSubslice makeSubsliceFromAttr(const GeneratedSlice &slice,
                                       ArrayRef<int64_t> sliceDims) {
  // If the attribute is empty, the subslice is the whole slice
  if (sliceDims.empty())
    return makeWholeSubslice(slice);

  // Create a subslice that points inside the given slice
  GeneratedSubslice subslice;
  subslice.sliceIndex = slice.descriptionIndex;
  subslice.offset = getPairwiseOffsets(sliceDims);
  subslice.shape = getPairwiseLengths(sliceDims);
  subslice.elementType = slice.elementType;
  subslice.elemBytes = slice.elemBytes;
  subslice.stridesBytes = slice.stridesBytes;
  return subslice;
}


unsigned addWholeSubsliceForSlice(MagiaApplicationDescription &application, unsigned sliceIndex) {
  // Add a subslice that covers the entire generated slice
  return addGeneratedSubslice(application, makeWholeSubslice(application.slices[sliceIndex]));
}


unsigned getOrCreateProducedSlice(
    MagiaApplicationDescription &application, llvm::StringMap<unsigned> &producedSliceIndices,
    StringRef uniqueKey, StringRef globalName, GeneratedGlobalKind kind,
    int64_t ownerHartid, RankedTensorType tensorType,
    ArrayRef<int64_t> fullOffset) {

  // Reuse the produced slice if it was already created for this key
  auto existingSlice = producedSliceIndices.find(uniqueKey);
  if (existingSlice != producedSliceIndices.end())
    return existingSlice->second;

  // Initialize the descriptor for the produced tile-local slice
  GeneratedSlice slice;
  slice.globalName = globalName.str();
  slice.globalKind = kind;
  slice.ownerHartid = ownerHartid;
  slice.fullOffset.append(fullOffset.begin(), fullOffset.end());
  if (slice.fullOffset.empty())
    slice.fullOffset.assign(tensorType.getRank(), 0);
  slice.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
  slice.elementType = tensorType.getElementType();
  slice.elemBytes = getElementByteWidth(slice.elementType);
  if (slice.elemBytes == 0)
    return std::numeric_limits<unsigned>::max();

  // Runtime produced slices are token-buffered
  slice.numSlots = application.numTokenSlots;
  slice.stridesBytes =
      computeContiguousStridesBytes(slice.shape, slice.elemBytes);

  // Add the slice to the application and allocate it in the owner's L1
  unsigned sliceIndex = addGeneratedSlice(application, std::move(slice));
  if (failed(allocateSliceInTileL1(application, sliceIndex)))
    return std::numeric_limits<unsigned>::max();

  // Remember the slice so later users of the same produced value can reuse it
  producedSliceIndices[uniqueKey] = sliceIndex;
  return sliceIndex;
}


LogicalResult collectInitializerStore(
    Operation *storeOp, const llvm::StringMap<L2SourceInfo> &sources,
    MagiaApplicationDescription &application) {

  /*
  Add L2 read descriptor for initializer data
  */
    
  // Find the tile which the store operation belongs to
  Operation *tileOp = findParentNamed(storeOp, "maps.tile");
  if (!tileOp)
    return failure();
  
  // Get the id of the parent tile
  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  
  // Get value of stored tensor
  Value storedValue = storeOp->getOperand(0);

  // Get type of stored tensor
  auto tensorType = dyn_cast<RankedTensorType>(storedValue.getType());
  
  // initialize slice descriptor
  GeneratedSlice slice;
  const L2SourceInfo *sourceInfo = nullptr;

  if (Operation *recvOp = storedValue.getDefiningOp()) {

    // get the name of the communication channel and find the corresponding send in the hash map
    std::string recvChannel = getFlatSymbolAttr(recvOp, "channel");
    auto source = sources.find(recvChannel);

    // Populate slice with info from send
    if (source != sources.end()) { // if source exists
      slice.globalName = source->second.globalName;
      slice.fullOffset = source->second.fullOffset;
      sourceInfo = &source->second; // keep copy for later use
    }
  }

  slice.globalKind = GeneratedGlobalKind::Initializer;
  slice.ownerHartid = tileId.getInt();
  slice.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
  slice.elementType = tensorType.getElementType();
  slice.elemBytes = getElementByteWidth(slice.elementType);

  // Error handling for 0 byte initializer
  if (slice.elemBytes == 0)
    return failure();

  // As initializer slices are static inside tiles, they always get one slot
  slice.numSlots = 1;

  // Get the strides for every dimension
  slice.stridesBytes = computeContiguousStridesBytes(slice.shape, slice.elemBytes);

  // add slice to plan and allocate the space in tile's L1
  unsigned sliceIndex = addGeneratedSlice(application, std::move(slice));
  if (failed(allocateSliceInTileL1(application, sliceIndex)))
    return failure();

  std::string slot = getFlatSymbolAttr(storeOp, "slot");
  if (!slot.empty())
    application.initializerSliceIndices[getTileSlotKey(tileId.getInt(), slot)] = sliceIndex;

  return addL2ReadForSlice(application, sliceIndex, *sourceInfo, GeneratedGlobalKind::Initializer, 1, true);
}


LogicalResult collectInputRecv(Operation *recvOp, const llvm::StringMap<L2SourceInfo> &sources, RunCollectionState &state, MagiaApplicationDescription &application) {

  // Find the tile which the recv operation belongs to
  Operation *tileOp = findParentNamed(recvOp, "maps.tile");
  if (!tileOp)
    return failure();

  // Get the id of the parent tile
  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  if (!tileId)
    return failure();

  // Only collect receives that come from L2
  auto srcHartid = recvOp->getAttrOfType<IntegerAttr>("src_hartid");
  if (!srcHartid || srcHartid.getInt() != -1)
    return success();

  // Find the L2 source associated with this receive channel
  std::string recvChannel = getFlatSymbolAttr(recvOp, "channel");
  auto source = sources.find(recvChannel);
  if (source == sources.end())
    return failure();

  if (recvOp->getNumResults() == 0)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(recvOp->getResult(0).getType());
  if (!tensorType)
    return failure();

  // Initialize the input slice descriptor using the source tensor metadata
  GeneratedSlice slice;
  slice.globalName = source->second.globalName;
  slice.globalKind = GeneratedGlobalKind::Input;
  slice.ownerHartid = tileId.getInt();
  slice.fullOffset = source->second.fullOffset;
  if (slice.fullOffset.empty())
    slice.fullOffset.assign(tensorType.getRank(), 0);
  slice.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
  slice.elementType = tensorType.getElementType();
  slice.elemBytes = getElementByteWidth(slice.elementType);
  if (slice.elemBytes == 0)
    return failure();

  // Runtime input slices are token-buffered
  slice.numSlots = application.numTokenSlots;
  slice.stridesBytes =
      computeContiguousStridesBytes(slice.shape, slice.elemBytes);

  // Add the input slice to the application and allocate it in tile L1
  unsigned sliceIndex = addGeneratedSlice(application, std::move(slice));
  if (failed(allocateSliceInTileL1(application, sliceIndex)))
    return failure();

  // The recv result value refers to the whole input slice
  unsigned subsliceIndex = addWholeSubsliceForSlice(application, sliceIndex);
  state.valueSubsliceIndices[recvOp->getResult(0)] = subsliceIndex;

  // Add the runtime L2 read descriptor for this input
  return addL2ReadForSlice(application, sliceIndex, source->second,
                           GeneratedGlobalKind::Input, application.numTokens,
                           false);
}


LogicalResult collectLoad(Operation *loadOp, RunCollectionState &state, MagiaApplicationDescription &application) {
  // Find the tile which the load operation belongs to
  Operation *tileOp = findParentNamed(loadOp, "maps.tile");
  if (!tileOp)
    return failure();

  // Get the id of the parent tile
  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  if (!tileId)
    return failure();

  // Find the initializer slot being loaded
  std::string slot = getFlatSymbolAttr(loadOp, "slot");
  if (slot.empty() || loadOp->getNumResults() == 0)
    return failure();

  // Look up the slice that was created for this tile-local initializer slot
  auto slice = application.initializerSliceIndices.find(
      getTileSlotKey(tileId.getInt(), slot));
  if (slice == application.initializerSliceIndices.end())
    return failure();

  // The load result value refers to the whole initializer slice
  unsigned subsliceIndex = addWholeSubsliceForSlice(application, slice->second);
  state.valueSubsliceIndices[loadOp->getResult(0)] = subsliceIndex;
  return success();
}


LogicalResult collectExtractSliceAlias(Operation *op, RunCollectionState &state, MagiaApplicationDescription &application) {

  // Ignore malformed extract_slice ops
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return success();

  // Find the subslice associated with the source value
  auto source = state.valueSubsliceIndices.find(op->getOperand(0));
  if (source == state.valueSubsliceIndices.end())
    return success();

  const GeneratedSubslice &sourceSubslice = application.subslices[source->second];
  const GeneratedSlice &sourceSlice = application.slices[sourceSubslice.sliceIndex];

  // Create an alias subslice inside the same generated slice
  GeneratedSubslice alias;
  alias.sliceIndex = sourceSubslice.sliceIndex;
  alias.offset = getDenseI64Array(op, "static_offsets");
  alias.shape = getDenseI64Array(op, "static_sizes");
  alias.elementType = sourceSubslice.elementType;
  alias.elemBytes = sourceSubslice.elemBytes;
  alias.stridesBytes = sourceSlice.stridesBytes;

  // Associate the extract_slice result with the alias subslice
  unsigned aliasIndex = addGeneratedSubslice(application, std::move(alias));
  state.valueSubsliceIndices[op->getResult(0)] = aliasIndex;
  return success();
}


LogicalResult collectInsertSliceAlias(Operation *op, RunCollectionState &state, MagiaApplicationDescription &application) {
  // Ignore malformed insert_slice ops
  if (op->getNumOperands() < 2 || op->getNumResults() == 0)
    return success();

  // Find the generated subslice associated with either insert operand
  auto dest = state.valueSubsliceIndices.find(op->getOperand(1));
  auto source = state.valueSubsliceIndices.find(op->getOperand(0));
  if (dest == state.valueSubsliceIndices.end() &&
      source == state.valueSubsliceIndices.end())
    return success();

  // The insert_slice result represents the whole slice after insertion
  unsigned existingSubsliceIndex =
      dest != state.valueSubsliceIndices.end() ? dest->second : source->second;
  const GeneratedSubslice &existingSubslice =
      application.subslices[existingSubsliceIndex];
  unsigned wholeSubsliceIndex =
      addWholeSubsliceForSlice(application, existingSubslice.sliceIndex);
  state.valueSubsliceIndices[op->getResult(0)] = wholeSubsliceIndex;
  return success();
}

} // namespace mlir::maps
