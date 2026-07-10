#include "IRCollector.h"

#include "collectors/CollectorSupport.h"
#include "collectors/L2Collector.h"
#include "collectors/OpCollector.h"
#include "collectors/SliceCollector.h"
#include "collectors/TransferCollector.h"

#include "mlir/IR/Operation.h"


namespace mlir::maps {


static LogicalResult collectInitRegion(Operation *initOp, MagiaHeaderModel &model) {

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


static LogicalResult collectRunRegion(Operation *runOp, MagiaHeaderModel &model) {
  
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

    if (name == "linalg.matmul" || name == "linalg.generic") {
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
LogicalResult collectMagiaHeaderModel(ModuleOp module, MagiaHeaderModel &model) {
                                        
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
