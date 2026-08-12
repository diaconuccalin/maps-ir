#include "MagiaApplicationCodegen/CodeGeneration.h"

#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputPath(llvm::cl::Positional,
                                      llvm::cl::desc("<maps-mlir>"),
                                      llvm::cl::Required);
  llvm::cl::opt<std::string> outputDirectory(
      "o", llvm::cl::desc("target application output directory"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> target(
      "target", llvm::cl::desc("target code generator"),
      llvm::cl::init("magia-v2"));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                   "Generate a Target application from maps MLIR\n");
  if (target != "magia-v2") {
    llvm::errs() << "unsupported code-generation target '" << target << "'\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::maps::MapsDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputPath, &context);
  if (!module || failed(mlir::verify(*module)))
    return 1;
  return failed(mlir::maps::generateMagiaApplication(
      *module, outputDirectory)) ? 1 : 0;
}
