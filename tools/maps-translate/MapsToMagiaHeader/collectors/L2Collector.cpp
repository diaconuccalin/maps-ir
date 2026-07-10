#include "L2Collector.h"

#include "SliceCollector.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::maps {


void collectL2Sources(Operation *regionOp, llvm::StringMap<L2SourceInfo> &sources) {
/*
Here we are trying to find the following pattern:
maps.send %extracted_slice, @init_matmul_w_from_L2_to_tile8 {dst_hartid = 8 : i64, src_hartid = -1 : i64} : tensor<1024x128xf32>
A send operation in the init region that starts from src = -1. This is the encoding for L2.
*/


  regionOp->walk([&](Operation *op) {

    // Return if it's not the right operation
    if (op->getName().getStringRef() != "maps.send")
      return;

    // Ensure the source id is -1. Only those should be here but better safe than sorry
    auto srcHartid = op->getAttrOfType<IntegerAttr>("src_hartid");
    if (!srcHartid || srcHartid.getInt() != -1)
      return;
    
    // Get channel name
    std::string channel = getFlatSymbolAttr(op, "channel");

    // Get tensor being sent
    Value sentValue = op->getOperand(0);

    // Get operation that defines this tensor
    Operation *definingOp = sentValue.getDefiningOp();

    // Ensure the operation that generates this tensor is a fragmentation operation; should always be this but hey
    if (!definingOp || definingOp->getName().getStringRef() != "tensor.extract_slice")
      return;

    // populate L2SourceInfo
    L2SourceInfo info;
    Value globalValue = definingOp->getOperand(0); // MLIR value of global tensor
    info.globalName = getGlobalNameForValue(globalValue);
    info.globalType = dyn_cast<RankedTensorType>(globalValue.getType());
    info.fullOffset = getDenseI64Array(definingOp, "static_offsets");
    
    /* debug
    llvm::errs() << "name: " << info.globalName << "\n";

    llvm::errs() << "definingOp: ";
    definingOp->dump();

    llvm::errs() << "globalValue: ";
    globalValue.dump();

    llvm::errs() << "globalValue type: ";
    globalValue.getType().dump();
    */

    // Move the contents of info into the sources has map and attach them to the communication channel
    sources[channel] = std::move(info); 
    

  });
}


LogicalResult addL2ReadForSlice(MagiaHeaderModel &model, unsigned sliceIndex,
                                const L2SourceInfo &source,
                                GeneratedGlobalKind globalKind,
                                unsigned numTokens, bool isInitializerRead) {
  /*
  Create the L2 read descriptor for a slice
  */
  GeneratedSlice &slice = model.slices[sliceIndex];

  unsigned l2GlobalIndex = 0;

  if (failed(getOrCreateL2Global(model, source.globalName, globalKind,
                                 source.globalType, numTokens,
                                 l2GlobalIndex)))
    return failure();
  
  // create the subslice associated with the transition
  GeneratedSubslice dst = makeWholeSubslice(slice);
  unsigned dstSubsliceIndex = addGeneratedSubslice(model, std::move(dst));

  const GeneratedL2Global &global = model.l2Globals[l2GlobalIndex];
  
  // populate read descriptor
  GeneratedL2Read read;
  read.l2GlobalIndex = l2GlobalIndex;
  read.dstSubsliceIndex = dstSubsliceIndex;
  read.srcL2Address = global.l2Address + computeOffsetBytes(slice.fullOffset, global.stridesBytes);
  read.srcL2TokenStrideBytes = globalKind == GeneratedGlobalKind::Input ? global.tokenStrideBytes : 0;
  
  // add descriptor to plan
  unsigned readIndex = addGeneratedL2Read(model, read);
  noteTile(model, slice.ownerHartid);
  GeneratedTilePlan &plan = model.tilePlans[slice.ownerHartid];
  
  // check if it's an initializer or input read
  if (isInitializerRead)
    plan.initL2ReadIndices.push_back(readIndex);
  else
    plan.l2ReadIndices.push_back(readIndex);

  return success();

}


void collectOutputInsertOffsets(Operation *ContainerOp, llvm::StringMap<OutputInsertInfo> &outputInserts) {
  /*
    Record the final output tensor placement for each host-bound receive channel.
    Tile-produced outputs are later written directly to L2, so the writer needs
    the tensor.insert_slice offsets and full output type to compute the final
    destination address
  */

  ContainerOp->walk([&](Operation *op) {

    if (op->getName().getStringRef() != "tensor.insert_slice")
      return;
    
    // Get defining op of the tensor.insert_slice op
    Operation *sourceOp = op->getOperand(0).getDefiningOp();

    // check if it's a receive
    if (!sourceOp || sourceOp->getName().getStringRef() != "maps.recv")
      return;

    // check if the receive is going to L2
    auto dstHartid = sourceOp->getAttrOfType<IntegerAttr>("dst_hartid");
    if (!dstHartid || dstHartid.getInt() != -1)
      return;

    // get channel name
    std::string channel = getFlatSymbolAttr(sourceOp, "channel");

    // Use the full result tensor type so the generated L2 output allocation matches
    // the final host-visible output.
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType)
      return;

    // Remember this channel's final insertion offset and output type for the later
    // L2 write descriptor.
    OutputInsertInfo info;
    info.fullOffset = getDenseI64Array(op, "static_offsets");
    info.outputType = outputType;
    outputInserts[channel] = std::move(info);

  });
}

LogicalResult addOutputL2Write(MagiaHeaderModel &model, StringRef globalName,
                               RankedTensorType outputGlobalType,
                               ArrayRef<int64_t> outputFullOffset,
                               unsigned srcSubsliceIndex, int64_t srcHartid) {
  unsigned l2GlobalIndex = 0;
  if (failed(getOrCreateL2Global(model, globalName,
                                 GeneratedGlobalKind::Output, outputGlobalType,
                                 model.numTokens, l2GlobalIndex)))
    return failure();

  const GeneratedL2Global &global = model.l2Globals[l2GlobalIndex];
  GeneratedL2Write write;
  write.l2GlobalIndex = l2GlobalIndex;
  write.srcSubsliceIndex = srcSubsliceIndex;
  write.dstL2Address =
      global.l2Address + computeOffsetBytes(outputFullOffset,
                                            global.stridesBytes);
  write.dstL2TokenStrideBytes = global.tokenStrideBytes;

  unsigned writeIndex = addGeneratedL2Write(model, write);
  noteTile(model, srcHartid);
  model.tilePlans[srcHartid].l2WriteIndices.push_back(writeIndex);
  return success();
}

} // namespace mlir::maps
