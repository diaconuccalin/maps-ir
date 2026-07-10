#include "Emitter.h"
#include "IRCollector.h"

#include "maps/Dialect/Maps/IR/Maps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::maps {

static llvm::cl::opt<unsigned> magiaNumTokens(
    "maps-magia-num-tokens",
    llvm::cl::desc("number of runtime tokens described by generated MAGIA L2 "
                   "input/output buffers"),
    llvm::cl::init(kDefaultNumTokens));

static llvm::cl::opt<unsigned> magiaNumTokenSlots(
    "maps-magia-num-token-slots",
    llvm::cl::desc("number of ping-pong L1 token slots used by generated "
                   "MAGIA tile plans"),
    llvm::cl::init(kDefaultNumTokenSlots));

static void configureGeneratedRuntime(MagiaHeaderModel &model) {
  model.numTokens = magiaNumTokens;
  model.numTokenSlots = magiaNumTokenSlots;
}

// Main translation hook registered with mlir-translate.  It only coordinates
// the three phases: verify the root op, collect a header model, then emit it.
static LogicalResult translateMapsToMagiaHeader(Operation *op,
                                                llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  MagiaHeaderModel model;
  configureGeneratedRuntime(model);
  if (failed(collectMagiaHeaderModel(module, model)))
    return failure();

  emitDebugHeader(model, os);
  return success();
}

static LogicalResult translateMapsToMagiaData(Operation *op,
                                              llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  MagiaHeaderModel model;
  configureGeneratedRuntime(model);
  if (failed(collectMagiaHeaderModel(module, model)))
    return failure();

  emitDataFile(model, os);
  return success();
}

void registerMapsToMagiaHeaderTranslation() {
  TranslateFromMLIRRegistration reg(
      "maps-to-magia-header",
      "emit MAGIA SDK test header from MAPS MLIR",
      translateMapsToMagiaHeader,
      [](DialectRegistry &registry) {
        registry.insert<maps::MapsDialect, func::FuncDialect,
                        tensor::TensorDialect, linalg::LinalgDialect,
                        math::MathDialect, scf::SCFDialect,
                        arith::ArithDialect>();
      });

  TranslateFromMLIRRegistration dataReg(
      "maps-to-magia-data",
      "emit MAGIA SDK L2 tensor data companion source from MAPS MLIR",
      translateMapsToMagiaData,
      [](DialectRegistry &registry) {
        registry.insert<maps::MapsDialect, func::FuncDialect,
                        tensor::TensorDialect, linalg::LinalgDialect,
                        math::MathDialect, scf::SCFDialect,
                        arith::ArithDialect>();
      });
}

} // namespace mlir::maps
