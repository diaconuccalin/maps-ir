#include "TransferCollector.h"

#include "L2Collector.h"
#include "SliceCollector.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/Twine.h"

#include <optional>

namespace mlir::maps {

static LogicalResult collectTileRecv(Operation *recvOp, RunCollectionState &state, MagiaHeaderModel &model) {
  /*
  Collect tile-to-tile receives before walking sends.
  A send descriptor needs the destination tile's L1 address, so each recv
  creates or reuses the destination aggregate slice and records the channel to
  destination subslice mapping.
  */

  Operation *tileOp = findParentNamed(recvOp, "maps.tile");

  // Safeguard, every recv operation in the run region should be inside a maps.tile module.
  if (!tileOp)
    return failure();

  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  auto srcHartid = recvOp->getAttrOfType<IntegerAttr>("src_hartid");
  auto dstHartid = recvOp->getAttrOfType<IntegerAttr>("dst_hartid");

  // Host/L2 inputs are handled separately; this collector only tracks tile-to-tile receives.
  if (srcHartid.getInt() < 0)
    return success();

  // Record channel name.
  std::string channel = getFlatSymbolAttr(recvOp, "channel");

  if (recvOp->getNumResults() == 0)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(recvOp->getResult(0).getType());
  if (!tensorType)
    return failure();

  SmallVector<int64_t> dstParentSlice = getDenseI64Array(recvOp, "dst_parent_slice");

  // Receives for the same logical destination region share one aggregate L1 slice.
  std::string baseName = getChannelBaseName(channel);
  std::string aggregateKey = getAggregateSliceKey(tileId.getInt(), baseName, dstParentSlice);

  // Check if another sublice for the same slice was already recorded.
  unsigned dstSliceIndex = 0;
  // returns the key if it exists, or state.aggregateSliceIndices.end() if it doesn't.
  auto existingAggregate = state.aggregateSliceIndices.find(aggregateKey);

  // Reuse the aggregate destination slice if another recv already created it.
  if (existingAggregate != state.aggregateSliceIndices.end()) {
    dstSliceIndex = existingAggregate->second; // assign the value of the key to var
  }

  // Create the destination tile's aggregate slice the first time this region is seen.
  else {
    GeneratedSlice slice;
    slice.globalName = baseName;

    // A receive inside a run region is always an intermediate
    slice.globalKind = GeneratedGlobalKind::Intermediate;
    slice.ownerHartid = tileId.getInt();

    // dst_parent_slice attribute should always be present for tile-to-tile
    if (!dstParentSlice.empty()) {
      slice.fullOffset = getPairwiseOffsets(dstParentSlice);
      slice.shape = getPairwiseLengths(dstParentSlice);
    } 
    // for future proofing
    else {
      slice.fullOffset.assign(tensorType.getRank(), 0);
      slice.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
    }

    slice.elementType = tensorType.getElementType();
    slice.elemBytes = getElementByteWidth(slice.elementType);

    // Safeguard for 0 byte transfers
    if (slice.elemBytes == 0)
      return failure();

    // Allocate the aggregate destination slice in the receiving tile's L1 and
    // remember it so later recvs for the same region reuse the same buffer.
    slice.numSlots = model.numTokenSlots;
    slice.stridesBytes = computeContiguousStridesBytes(slice.shape, slice.elemBytes);

    dstSliceIndex = addGeneratedSlice(model, std::move(slice));

    // Check for enough l1 space
    if (failed(allocateSliceInTileL1(model, dstSliceIndex)))
      return failure();

    state.aggregateSliceIndices[aggregateKey] = dstSliceIndex;
  }

  GeneratedSlice &dstSlice = model.slices[dstSliceIndex];

  // The recv writes into this subslice of the aggregate destination slice.
  SmallVector<int64_t> dstSubslice = getDenseI64Array(recvOp, "dst_subslice");
  unsigned dstSubsliceIndex = addGeneratedSubslice(model, makeSubsliceFromAttr(dstSlice, dstSubslice));

  // Populate
  GeneratedRecv recv;
  recv.transitionId = getOrCreateTransitionId(model, channel);
  recv.srcHartid = srcHartid.getInt();
  recv.dstHartid = dstHartid.getInt();
  recv.dstSubsliceIndex = dstSubsliceIndex;
  recv.readyFlagId = 0;

  unsigned recvIndex = addGeneratedRecv(model, recv);
  noteTile(model, recv.dstHartid);
  model.tilePlans[recv.dstHartid].recvIndices.push_back(recvIndex);

  // Remember the destination endpoint for the matching send and for later users
  // of the recv result value.
  state.tileRecvs[channel] = TileRecvInfo{dstSliceIndex, dstSubsliceIndex};
  state.valueSubsliceIndices[recvOp->getResult(0)] = dstSubsliceIndex;
  return success();

}


LogicalResult collectTileRecvs(Operation *runOp, RunCollectionState &state, MagiaHeaderModel &model) {
  /*
    here we are trying to collect the following pattern types:
    %21 = maps.recv @transition_matmul_0_to_log_0_matmul_out_from_tile49_to_tile21 {dst_hartid = 21 : i64, dst_parent_slice = array<i64: 64, 64, 39, 13>, 
    dst_subslice = array<i64: 0, 32, 0, 5>, src_hartid = 49 : i64, src_parent_slice = array<i64: 64, 32, 22, 22>, 
    src_subslice = array<i64: 0, 32, 17, 5>} : tensor<32x5xf16>
  */

  WalkResult result = runOp->walk([&](Operation *op) -> WalkResult {
    if (op->getName().getStringRef() != "maps.recv")
      return WalkResult::advance();

    if (failed(collectTileRecv(op, state, model)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

static std::optional<ProducedSliceSource>
getProducedSliceSource(Operation *sendOp) {
  if (sendOp->getNumOperands() == 0)
    return std::nullopt;

  Value sentValue = sendOp->getOperand(0);
  ProducedSliceSource source;

  if (Operation *definingOp = sentValue.getDefiningOp()) {
    if (definingOp->getName().getStringRef() == "tensor.extract_slice" &&
        definingOp->getNumOperands() > 0) {
      source.rootValue = definingOp->getOperand(0);
      source.rootType = dyn_cast<RankedTensorType>(source.rootValue.getType());
      source.fullOffset = getDenseI64Array(definingOp, "static_offsets");
    }
  }

  if (!source.rootValue) {
    source.rootValue = sentValue;
    source.rootType = dyn_cast<RankedTensorType>(sentValue.getType());
  }

  if (!source.rootType)
    return std::nullopt;

  SmallVector<int64_t> parentSlice =
      getDenseI64Array(sendOp, "src_parent_slice");
  if (!parentSlice.empty())
    source.fullOffset = getPairwiseOffsets(parentSlice);

  if (source.fullOffset.empty())
    source.fullOffset.assign(source.rootType.getRank(), 0);

  return source;
}

LogicalResult collectProducedSend(Operation *sendOp, RunCollectionState &state,
                                  MagiaHeaderModel &model) {
  auto srcHartid = sendOp->getAttrOfType<IntegerAttr>("src_hartid");
  auto dstHartid = sendOp->getAttrOfType<IntegerAttr>("dst_hartid");
  if (!srcHartid || !dstHartid)
    return failure();

  if (srcHartid.getInt() < 0)
    return success();

  std::string channel = getFlatSymbolAttr(sendOp, "channel");
  if (channel.empty())
    return failure();

  std::optional<ProducedSliceSource> source = getProducedSliceSource(sendOp);
  if (!source)
    return failure();

  GeneratedGlobalKind kind = dstHartid.getInt() == -1
                                 ? GeneratedGlobalKind::Output
                                 : GeneratedGlobalKind::Intermediate;
  std::string globalName = getChannelBaseName(channel);
  std::string uniqueKey =
      (Twine(static_cast<int>(kind)) + ":" + Twine(srcHartid.getInt()) + ":" +
       globalName + ":" + getProducedValueKey(source->rootValue))
          .str();
  unsigned srcSliceIndex = 0;
  SmallVector<int64_t> outputFullOffset = source->fullOffset;
  RankedTensorType outputGlobalType = source->rootType;
  auto outputInsert = state.outputInserts.find(channel);
  if (kind == GeneratedGlobalKind::Output &&
      outputInsert != state.outputInserts.end()) {
    outputFullOffset = outputInsert->second.fullOffset;
    outputGlobalType = outputInsert->second.outputType;
  }

  auto existingSubslice = state.valueSubsliceIndices.find(source->rootValue);
  if (existingSubslice != state.valueSubsliceIndices.end()) {
    srcSliceIndex = model.subslices[existingSubslice->second].sliceIndex;
    state.producedSliceIndices[uniqueKey] = srcSliceIndex;
  } else {
    srcSliceIndex = getOrCreateProducedSlice(
        model, state.producedSliceIndices, uniqueKey, globalName, kind,
        srcHartid.getInt(), source->rootType, outputFullOffset);
    if (srcSliceIndex == std::numeric_limits<unsigned>::max())
      return failure();
  }

  GeneratedSlice &srcSlice = model.slices[srcSliceIndex];
  SmallVector<int64_t> srcSubslice = getDenseI64Array(sendOp, "src_subslice");
  unsigned srcSubsliceIndex =
      addGeneratedSubslice(model, makeSubsliceFromAttr(srcSlice, srcSubslice));

  if (dstHartid.getInt() == -1)
    return addOutputL2Write(model, globalName, outputGlobalType,
                            outputFullOffset, srcSubsliceIndex,
                            srcHartid.getInt());

  auto recvInfo = state.tileRecvs.find(channel);
  if (recvInfo == state.tileRecvs.end())
    return failure();

  GeneratedSlice &dstSlice = model.slices[recvInfo->second.dstSliceIndex];
  GeneratedSend send;
  send.transitionId = getOrCreateTransitionId(model, channel);
  send.srcHartid = srcHartid.getInt();
  send.dstHartid = dstHartid.getInt();
  send.srcSubsliceIndex = srcSubsliceIndex;
  send.dstSubsliceIndex = recvInfo->second.dstSubsliceIndex;
  send.dstL1DataBase = kGeneratedL1DataBaseOffset;
  send.dstSliceL1OffsetBytes = dstSlice.l1OffsetBytes;
  send.dstSliceSlotBytes = dstSlice.slotBytes;
  send.readyFlagId = 0;

  unsigned sendIndex = addGeneratedSend(model, send);
  noteTile(model, send.srcHartid);
  model.tilePlans[send.srcHartid].sendIndices.push_back(sendIndex);
  return success();
}

} // namespace mlir::maps
