#include "IRCollector.h"

#include "collectors/CollectorSupport.h"
#include "collectors/L2Collector.h"
#include "collectors/OpCollector.h"
#include "collectors/SliceCollector.h"
#include "collectors/TransferCollector.h"

#include "mlir/IR/Operation.h"

#include <algorithm>
#include <limits>


namespace mlir::maps {

static LogicalResult collectExecutionContract(ModuleOp module,
                                              MagiaApplicationDescription &model) {
  auto numTokenSlots =
      module->getAttrOfType<IntegerAttr>("maps.num_token_slots");
  if (!numTokenSlots || numTokenSlots.getInt() <= 0 ||
      static_cast<uint64_t>(numTokenSlots.getInt()) >
          std::numeric_limits<unsigned>::max())
    return failure();
  model.numTokenSlots = static_cast<unsigned>(numTokenSlots.getInt());
  return success();
}

static LogicalResult collectDeploymentIdentity(ModuleOp module,
                                               MagiaApplicationDescription &model) {
  if (auto name = module->getAttrOfType<StringAttr>("maps.name"))
    model.sourceModelName = name.getValue().str();

  auto meshWidth = module->getAttrOfType<IntegerAttr>("maps.mesh_width");
  auto meshHeight = module->getAttrOfType<IntegerAttr>("maps.mesh_height");
  if (meshWidth && meshHeight && meshWidth.getInt() > 0 &&
      meshHeight.getInt() > 0 &&
      static_cast<uint64_t>(meshWidth.getInt()) <=
          std::numeric_limits<unsigned>::max() &&
      static_cast<uint64_t>(meshHeight.getInt()) <=
          std::numeric_limits<unsigned>::max()) {
    model.meshWidth = static_cast<unsigned>(meshWidth.getInt());
    model.meshHeight = static_cast<unsigned>(meshHeight.getInt());
  } else if (meshWidth || meshHeight) {
    return failure();
  }
  return success();
}

static LogicalResult collectBundleMetadata(ModuleOp module,
                                           MagiaApplicationDescription &model) {
  auto bundle = module->getAttrOfType<DictionaryAttr>("maps.bundle");
  if (!bundle)
    return success();

  auto schemaVersion = bundle.getAs<IntegerAttr>("schema_version");
  auto weightsFile = bundle.getAs<StringAttr>("weights_file");
  auto weightsSize = bundle.getAs<IntegerAttr>("weights_size");
  auto weightsSha256 = bundle.getAs<StringAttr>("weights_sha256");
  auto alignment = bundle.getAs<IntegerAttr>("alignment");
  auto endianness = bundle.getAs<StringAttr>("endianness");
  if (!schemaVersion || schemaVersion.getInt() != 1 || !weightsFile ||
      !weightsSize || weightsSize.getInt() < 0 || !weightsSha256 ||
      !alignment || alignment.getInt() <= 0 ||
      (alignment.getInt() & (alignment.getInt() - 1)) != 0 ||
      !endianness || endianness.getValue() != "little")
    return failure();
  if (weightsFile.getValue().contains('"') ||
      weightsFile.getValue().contains('\n') ||
      weightsSha256.getValue().size() != 64)
    return failure();

  model.hasPackedWeights = true;
  model.weightsFile = weightsFile.getValue().str();
  model.weightsSize = weightsSize.getInt();
  model.weightsSha256 = weightsSha256.getValue().str();
  model.weightsAlignment = alignment.getInt();
  if (model.l2Base % model.weightsAlignment != 0)
    return failure();

  if (auto capacity = module->getAttrOfType<IntegerAttr>(
          "maps.l2_capacity_bytes")) {
    if (capacity.getInt() <= 0)
      return failure();
    uint64_t configuredEnd = model.l2Base + capacity.getInt();
    model.l2End = std::min(model.l2End, configuredEnd);
  }
  if (model.weightsSize > model.l2End - model.l2Base)
    return failure();
  model.nextL2Address = alignTo(model.l2Base + model.weightsSize,
                                kGeneratedL2AlignmentBytes);
  return success();
}


static LogicalResult collectInitRegion(Operation *initOp, MagiaApplicationDescription &model) {

  llvm::StringMap<L2SourceInfo> sources; // llvm hash map optimized for llvm strings
  collectL2Sources(initOp, sources);

  WalkResult result = initOp->walk([&](Operation *op) -> WalkResult {

    // search for maps.store operations
    if (op->getName().getStringRef() != "maps.store")
      return WalkResult::advance();

    if (failed(collectInitializerStore(op, sources, model)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();

}


static LogicalResult collectRunRegion(Operation *runOp, MagiaApplicationDescription &model) {
  
  llvm::StringMap<L2SourceInfo> sources;
  collectL2Sources(runOp, sources);

  RunCollectionState state;
  if (failed(collectTileRecvs(runOp, state, model)))
    return failure();
  
  // Collect the host-side insertion offsets of outputs. The operation is the
  // parent of the run.op as the collection happens after running.
  collectOutputInsertOffsets(runOp->getParentOp(), state.outputInserts);

  WalkResult result = runOp->walk([&](Operation *op) -> WalkResult {
    StringRef name = op->getName().getStringRef();

    if (name == "maps.recv") {
      if (failed(collectInputRecv(op, sources, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "maps.load") {
      if (failed(collectLoad(op, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "tensor.extract_slice") {
      if (failed(collectExtractSliceAlias(op, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "tensor.insert_slice") {
      if (failed(collectInsertSliceAlias(op, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    /* TODO: create a maps.execute dialect so that we can directly
    branch with that instead of linalg operations.
    This would become something like:
    
    if (name == "maps.execute"){
      if (failed(collectComputeOp(op, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();}

    This can also be helpful to account for ops with 
    intra-stage copies like softmax
    */

    if (name == "linalg.matmul" || name == "linalg.generic" ||
        name == "maps.conv2d" || name == "maps.kernel") {
      if (failed(collectComputeOp(op, state, model)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name != "maps.send")
      return WalkResult::advance();

    if (failed(collectProducedSend(op, state, model)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}


// Top level collector function
LogicalResult collectMagiaApplicationDescription(ModuleOp module, MagiaApplicationDescription &model) {
  if (failed(collectDeploymentIdentity(module, model)) ||
      failed(collectExecutionContract(module, model)) ||
      failed(collectBundleMetadata(module, model)))
    return failure();
                                        
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    StringRef name = op->getName().getStringRef();

    // init region collection
    if (name == "maps.init") {
      if (failed(collectInitRegion(op, model)))
        return WalkResult::interrupt();
    }

    // run region collector
    if (name == "maps.run") {
      if (failed(collectRunRegion(op, model)))
        return WalkResult::interrupt();
    }
  
    return WalkResult::advance();
  });

  // return failure if any collection phase encountered a bad pattern
  if (result.wasInterrupted())
    return failure();

  return success();
}


} // namespace mlir::maps
