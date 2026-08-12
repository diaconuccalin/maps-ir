#include "ExecutionPlanImporter.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputPath(llvm::cl::Positional,
                                      llvm::cl::desc("<execution-plan>"),
                                      llvm::cl::Required);
  llvm::cl::opt<std::string> outputPath(
      "o", llvm::cl::desc("output maps MLIR file"), llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                   "Import an Execution Plan into maps MLIR\n");

  auto input = llvm::MemoryBuffer::getFileOrSTDIN(inputPath);
  if (!input) {
    llvm::errs() << "failed to read Execution Plan '" << inputPath << "': "
                 << input.getError().message() << "\n";
    return 1;
  }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::Operation *> imported =
      mlir::maps::importExecutionPlan((*input)->getBuffer(), &context);
  if (!imported)
    return 1;

  std::error_code error;
  llvm::raw_fd_ostream output(outputPath, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open '" << outputPath
                 << "': " << error.message() << "\n";
    return 1;
  }
  imported.get()->print(output);
  output << "\n";
  return output.has_error() ? 1 : 0;
}
