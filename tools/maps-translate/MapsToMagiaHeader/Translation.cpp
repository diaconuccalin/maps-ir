#include "CodeGeneration.h"
#include "Emitter.h"
#include "IRCollector.h"
#include "../ExecutionPlanImporter.h"

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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::maps {

static llvm::cl::opt<unsigned> magiaNumTokens(
    "maps-magia-num-tokens",
    llvm::cl::desc("number of runtime tokens described by generated MAGIA L2 "
                   "input/output buffers"),
    llvm::cl::init(kDefaultNumTokens));

static llvm::cl::opt<std::string> magiaOutputStem(
    "maps-magia-output-stem",
    llvm::cl::desc("C/assembly symbol stem for generated deployment artifacts"),
    llvm::cl::init("model"));

static llvm::cl::opt<uint64_t> magiaL2Base(
    "maps-magia-l2-base",
    llvm::cl::desc("first target L2 address available to generated artifacts"),
    llvm::cl::init(kMagiaL2Base));

static llvm::cl::opt<uint64_t> magiaL2End(
    "maps-magia-l2-end",
    llvm::cl::desc("exclusive target L2 address limit"),
    llvm::cl::init(kMagiaL2End));

static llvm::cl::opt<bool> magiaRandomizeInputs(
    "maps-magia-test-randomize-inputs",
    llvm::cl::desc("test-only pseudo-random initialization of synthetic inputs"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> magiaWeightsFile(
    "maps-magia-weights-file",
    llvm::cl::desc("assembly .incbin path overriding bundle weights_file"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> magiaPackageDir(
    "maps-magia-package-dir",
    llvm::cl::desc("directory written by --json-to-magia-package"),
    llvm::cl::init(""));

static void configureGeneratedRuntime(MagiaHeaderModel &model) {
  model.numTokens = magiaNumTokens;
  model.outputStem = magiaOutputStem;
  model.l2Base = magiaL2Base;
  model.l2End = magiaL2End;
  model.nextL2Address = model.l2Base;
  model.randomizeInputs = magiaRandomizeInputs;
}

static bool isValidSymbolStem(StringRef stem) {
  if (stem.empty() || (!llvm::isAlpha(stem.front()) && stem.front() != '_'))
    return false;
  return llvm::all_of(stem.drop_front(), [](char value) {
    return llvm::isAlnum(value) || value == '_';
  });
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
  if (!isValidSymbolStem(model.outputStem) || model.l2End <= model.l2Base)
    return failure();
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
  if (!isValidSymbolStem(model.outputStem) || model.l2End <= model.l2Base)
    return failure();
  if (failed(collectMagiaHeaderModel(module, model)))
    return failure();

  emitDataFile(model, os);
  return success();
}

static LogicalResult translateMapsToMagiaWeightsAssembly(
    Operation *op, llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  MagiaHeaderModel model;
  configureGeneratedRuntime(model);
  if (!isValidSymbolStem(model.outputStem) || model.l2End <= model.l2Base ||
      failed(collectMagiaHeaderModel(module, model)) ||
      !model.hasPackedWeights)
    return failure();

  if (!magiaWeightsFile.empty())
    model.weightsFile = magiaWeightsFile;

  emitWeightsAssembly(model, os);
  return success();
}

static LogicalResult writeTextArtifact(
    StringRef path,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open '" << path << "': " << error.message()
                 << "\n";
    return failure();
  }
  emit(stream);
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << "failed to write '" << path << "'\n";
    return failure();
  }
  return success();
}

static LogicalResult writeBinaryArtifact(StringRef path, StringRef contents) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open '" << path << "': " << error.message()
                 << "\n";
    return failure();
  }
  stream.write(contents.data(), contents.size());
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << "failed to write '" << path << "'\n";
    return failure();
  }
  return success();
}

static std::string packageArtifactPath(StringRef outputDirectory,
                                       StringRef filename) {
  llvm::SmallString<256> path(outputDirectory);
  llvm::sys::path::append(path, filename);
  return path.str().str();
}

static std::string sha256(StringRef contents) {
  llvm::SHA256 digest;
  digest.update(contents);
  return llvm::toHex(digest.final(), /*LowerCase=*/true);
}

static std::optional<std::string> manifestDtype(Type type) {
  if (type.isF16())
    return "float16";
  if (type.isF32())
    return "float32";
  if (type.isF64())
    return "float64";
  if (type.isBF16())
    return "bfloat16";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    if (integer.getWidth() == 8 && integer.isUnsigned())
      return "uint8";
    if (integer.getWidth() == 32)
      return "int32";
    if (integer.getWidth() == 64)
      return "int64";
  }
  return std::nullopt;
}

static FailureOr<llvm::json::Object>
tensorManifestEntry(const GeneratedL2Global &global) {
  std::optional<std::string> dtype = manifestDtype(global.elementType);
  if (!dtype)
    return failure();

  llvm::json::Array shape;
  for (int64_t dimension : global.shape)
    shape.push_back(dimension);

  llvm::json::Object tensor;
  tensor["id"] = global.globalId;
  tensor["name"] = global.globalName;
  tensor["dtype"] = *dtype;
  tensor["shape"] = std::move(shape);
  tensor["encoding"] = "raw";
  tensor["endianness"] = "little";
  tensor["tensor_bytes"] = static_cast<int64_t>(global.tensorBytes);
  return tensor;
}

struct PackageArtifact {
  std::string path;
  std::string role;
};

static LogicalResult writeDeploymentManifest(
    const MagiaHeaderModel &model, ArrayRef<PackageArtifact> artifacts,
    StringRef outputDirectory) {
  if (model.sourceModelName.empty() || model.meshWidth == 0 ||
      model.meshHeight == 0)
    return failure();

  llvm::json::Array inputs;
  llvm::json::Array outputs;
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input &&
        global.globalKind != GeneratedGlobalKind::Output)
      continue;
    FailureOr<llvm::json::Object> entry = tensorManifestEntry(global);
    if (failed(entry))
      return failure();
    if (global.globalKind == GeneratedGlobalKind::Input)
      inputs.push_back(std::move(*entry));
    else
      outputs.push_back(std::move(*entry));
  }

  uint64_t maxTileDataBytes = 0;
  for (const auto &tilePlan : model.tilePlans)
    maxTileDataBytes =
        std::max(maxTileDataBytes, tilePlan.second.l1DataBytes);

  llvm::json::Array artifactEntries;
  for (const PackageArtifact &artifact : artifacts) {
    auto contentsOrError = llvm::MemoryBuffer::getFile(
        packageArtifactPath(outputDirectory, artifact.path), /*IsText=*/false,
        /*RequiresNullTerminator=*/false);
    if (!contentsOrError)
      return failure();
    StringRef contents = (*contentsOrError)->getBuffer();
    artifactEntries.push_back(llvm::json::Object{
        {"path", artifact.path},
        {"role", artifact.role},
        {"byte_size", static_cast<int64_t>(contents.size())},
        {"sha256", sha256(contents)},
    });
  }

  llvm::json::Object manifest{
      {"schema_version", 1},
      {"source_model",
       llvm::json::Object{
           {"name", model.sourceModelName},
           {"inputs", std::move(inputs)},
           {"outputs", std::move(outputs)},
       }},
      {"target",
       llvm::json::Object{
           {"architecture", "magia-v2"},
           {"mesh",
            llvm::json::Object{
                {"width", model.meshWidth},
                {"height", model.meshHeight},
            }},
       }},
      {"execution",
       llvm::json::Object{
           {"num_token_slots", model.numTokenSlots},
           {"pipeline_token_capacity", model.numTokens},
       }},
      {"abi",
       llvm::json::Object{
           {"operation_version", kGeneratedOperationAbiVersion},
           {"descriptor_version", kGeneratedDescriptorAbiVersion},
           {"descriptor_max_dimensions", kGeneratedDescriptorMaxDims},
       }},
      {"entry_points",
       llvm::json::Object{
           {"init_l2_data", "maps_generated_init_l2_data"},
           {"init_tensors", "maps_generated_init_tensors"},
           {"fill_plan", "maps_generated_fill_plan"},
           {"check_output", "maps_generated_check_output"},
       }},
      {"memory",
       llvm::json::Object{
           {"ready_flags_offset",
            static_cast<int64_t>(kGeneratedReadyFlagsOffset)},
           {"max_tile_data_bytes", static_cast<int64_t>(maxTileDataBytes)},
           {"weights_size", static_cast<int64_t>(model.weightsSize)},
           {"required_l2_bytes",
            static_cast<int64_t>(model.nextL2Address - model.l2Base)},
       }},
      {"artifacts", std::move(artifactEntries)},
  };

  std::string manifestText =
      llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
  return writeTextArtifact(
      packageArtifactPath(outputDirectory, "manifest.json"),
      [&](llvm::raw_ostream &os) { os << manifestText; });
}

static LogicalResult generateLegacyMagiaPackage(ModuleOp module,
                                                StringRef outputDirectory);

static OwningOpRef<Operation *>
translateJsonToMagiaPackage(StringRef input, MLIRContext *context) {
  if (magiaPackageDir.empty()) {
    llvm::errs() << "--maps-magia-package-dir is required\n";
    return {};
  }

  OwningOpRef<Operation *> imported = importExecutionPlan(input, context);
  if (!imported)
    return {};
  auto module = dyn_cast<ModuleOp>(imported.get());
  if (!module)
    return {};

  if (failed(generateLegacyMagiaPackage(module, magiaPackageDir)))
    return {};
  return imported;
}

static LogicalResult generateLegacyMagiaPackage(ModuleOp module,
                                                StringRef outputDirectory) {
  MagiaHeaderModel model;
  configureGeneratedRuntime(model);
  if (!isValidSymbolStem(model.outputStem) || model.l2End <= model.l2Base ||
      failed(collectMagiaHeaderModel(module, model)) ||
      !model.hasPackedWeights)
    return failure();

  StringRef sourceWeights =
      magiaWeightsFile.empty() ? StringRef(model.weightsFile)
                               : StringRef(magiaWeightsFile);
  auto weightsOrError = llvm::MemoryBuffer::getFile(
      sourceWeights, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!weightsOrError) {
    llvm::errs() << "failed to read packed weights '" << sourceWeights
                 << "': " << weightsOrError.getError().message() << "\n";
    return failure();
  }
  StringRef weights = (*weightsOrError)->getBuffer();
  if (weights.size() != model.weightsSize) {
    llvm::errs() << "packed weights size mismatch: expected "
                 << model.weightsSize << " bytes, found " << weights.size()
                 << "\n";
    return failure();
  }
  if (sha256(weights) != model.weightsSha256) {
    llvm::errs() << "packed weights checksum mismatch\n";
    return failure();
  }

  std::error_code directoryError =
      llvm::sys::fs::create_directories(outputDirectory);
  if (directoryError) {
    llvm::errs() << "failed to create package directory '" << outputDirectory
                 << "': " << directoryError.message() << "\n";
    return failure();
  }

  std::string headerName = model.outputStem + ".h";
  std::string dataName = model.outputStem + "_data.c";
  std::string assemblyName = model.outputStem + "_weights.S";
  std::string weightsName = model.outputStem + ".weights.bin";
  model.weightsFile = weightsName;

  if (failed(writeTextArtifact(
          packageArtifactPath(outputDirectory, headerName),
          [&](llvm::raw_ostream &os) { emitDebugHeader(model, os); })) ||
      failed(writeTextArtifact(
          packageArtifactPath(outputDirectory, dataName),
          [&](llvm::raw_ostream &os) { emitDataFile(model, os); })) ||
      failed(writeTextArtifact(
          packageArtifactPath(outputDirectory, assemblyName),
          [&](llvm::raw_ostream &os) { emitWeightsAssembly(model, os); })))
    return failure();

  {
    if (failed(writeBinaryArtifact(
            packageArtifactPath(outputDirectory, weightsName), weights)))
      return failure();
  }

  SmallVector<PackageArtifact> artifacts{
      {headerName, "header"},
      {dataName, "data_source"},
      {assemblyName, "weights_assembly"},
      {weightsName, "weights_image"},
  };
  if (failed(writeDeploymentManifest(model, artifacts, outputDirectory))) {
    llvm::errs() << "failed to write deployment manifest\n";
    return failure();
  }

  return success();
}

static FailureOr<llvm::json::Object>
applicationTensorEntry(const GeneratedL2Global &global) {
  std::optional<std::string> dtype = manifestDtype(global.elementType);
  if (!dtype)
    return failure();
  llvm::json::Array shape;
  for (int64_t dimension : global.shape)
    shape.push_back(dimension);
  return llvm::json::Object{
      {"id", global.globalId},
      {"original_name", global.globalName},
      {"normalized_name", normalizeSnakeCase(global.globalName)},
      {"dtype", *dtype},
      {"shape", std::move(shape)},
      {"byte_size", static_cast<int64_t>(global.tensorBytes)},
  };
}

static LogicalResult writeApplicationManifest(const MagiaHeaderModel &model,
                                              StringRef outputDirectory) {
  llvm::json::Array inputs;
  llvm::json::Array outputs;
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input &&
        global.globalKind != GeneratedGlobalKind::Output)
      continue;
    FailureOr<llvm::json::Object> entry = applicationTensorEntry(global);
    if (failed(entry))
      return failure();
    if (global.globalKind == GeneratedGlobalKind::Input)
      inputs.push_back(std::move(*entry));
    else
      outputs.push_back(std::move(*entry));
  }

  llvm::json::Array activeTiles;
  uint64_t maxTileDataBytes = 0;
  for (const auto &tilePlan : model.tilePlans) {
    activeTiles.push_back(tilePlan.first);
    maxTileDataBytes = std::max(maxTileDataBytes, tilePlan.second.l1DataBytes);
  }

  std::string name = model.outputStem;
  llvm::json::Array generatedFiles{
      "CMakeLists.txt",
      "README.md",
      "manifest.json",
      "include/" + name + ".h",
      "src/" + name + ".c",
      "src/" + name + "_runner.c",
      "src/" + name + "_initializers.S.in",
      "data/" + name + ".initializers.bin",
  };
  llvm::json::Array userOwnedFiles{"src/application.c"};

  llvm::json::Object manifest{
      {"schema_version", 1},
      {"application",
       llvm::json::Object{{"name", name}, {"target", "magia-v2"}}},
      {"source_model", model.sourceModelName},
      {"planned_mesh", llvm::json::Object{{"width", model.meshWidth},
                                          {"height", model.meshHeight}}},
      {"abi",
       llvm::json::Object{
           {"operation", kGeneratedOperationAbiVersion},
           {"descriptor", kGeneratedDescriptorAbiVersion},
       }},
      {"execution",
       llvm::json::Object{{"tokens", 1}, {"token_slots", model.numTokenSlots}}},
      {"active_physical_tiles", std::move(activeTiles)},
      {"tensors", llvm::json::Object{{"inputs", std::move(inputs)},
                                     {"outputs", std::move(outputs)}}},
      {"entry_points",
       llvm::json::Object{
           {"run", name + "_run"},
           {"handle_input", name + "_handle_input"},
           {"handle_output", name + "_handle_output"},
       }},
      {"memory",
       llvm::json::Object{
           {"initializers_bytes", static_cast<int64_t>(model.weightsSize)},
           {"required_l2_bytes",
            static_cast<int64_t>(model.nextL2Address - model.l2Base)},
           {"max_tile_l1_bytes", static_cast<int64_t>(maxTileDataBytes)},
       }},
      {"files", llvm::json::Object{{"generated", std::move(generatedFiles)},
                                   {"user_owned", std::move(userOwnedFiles)}}},
  };
  std::string manifestText =
      llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
  return writeTextArtifact(
      packageArtifactPath(outputDirectory, "manifest.json"),
      [&](llvm::raw_ostream &os) { os << manifestText; });
}

LogicalResult generateMagiaApplication(ModuleOp module,
                                       StringRef outputDirectory) {
  MagiaHeaderModel model;
  configureGeneratedRuntime(model);
  model.outputStem = normalizeSnakeCase(model.outputStem);
  model.numTokens = 1;
  if (model.l2End <= model.l2Base ||
      failed(collectMagiaHeaderModel(module, model)) ||
      model.sourceModelName.empty() || model.meshWidth == 0 ||
      model.meshHeight == 0 || !model.hasPackedWeights)
    return failure();

  StringRef sourceInitializers = magiaWeightsFile.empty()
                                     ? StringRef(model.weightsFile)
                                     : StringRef(magiaWeightsFile);
  auto initializersOrError = llvm::MemoryBuffer::getFile(
      sourceInitializers, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!initializersOrError) {
    llvm::errs() << "failed to read packed Initializers '" << sourceInitializers
                 << "': " << initializersOrError.getError().message() << "\n";
    return failure();
  }
  StringRef initializers = (*initializersOrError)->getBuffer();
  if (initializers.size() != model.weightsSize) {
    llvm::errs() << "packed Initializers size mismatch: expected "
                 << model.weightsSize << " bytes, found " << initializers.size()
                 << "\n";
    return failure();
  }
  if (sha256(initializers) != model.weightsSha256) {
    llvm::errs() << "packed Initializers checksum mismatch\n";
    return failure();
  }

  for (StringRef directory : {StringRef(), StringRef("include"),
                              StringRef("src"), StringRef("data")}) {
    std::string path = directory.empty()
                           ? outputDirectory.str()
                           : packageArtifactPath(outputDirectory, directory);
    std::error_code error = llvm::sys::fs::create_directories(path);
    if (error) {
      llvm::errs() << "failed to create application directory '" << path
                   << "': " << error.message() << "\n";
      return failure();
    }
  }

  std::string name = model.outputStem;
  auto writeApplicationText =
      [&](StringRef path, llvm::function_ref<void(llvm::raw_ostream &)> emit) {
        return writeTextArtifact(packageArtifactPath(outputDirectory, path),
                                 emit);
      };
  if (failed(writeApplicationText(
          "CMakeLists.txt",
          [&](llvm::raw_ostream &os) { emitApplicationCMake(model, os); })) ||
      failed(writeApplicationText(
          "README.md",
          [&](llvm::raw_ostream &os) { emitApplicationReadme(model, os); })) ||
      failed(writeApplicationText(
          "include/" + name + ".h",
          [&](llvm::raw_ostream &os) { emitApplicationHeader(model, os); })) ||
      failed(writeApplicationText("src/" + name + ".c",
                                  [&](llvm::raw_ostream &os) {
                                    emitApplicationModelSource(model, os);
                                  })) ||
      failed(writeApplicationText(
          "src/" + name + "_runner.c",
          [&](llvm::raw_ostream &os) { emitApplicationRunner(model, os); })) ||
      failed(writeApplicationText("src/application.c",
                                  [&](llvm::raw_ostream &os) {
                                    emitInitialApplicationSource(model, os);
                                  })) ||
      failed(writeApplicationText(
          "src/" + name + "_initializers.S.in", [&](llvm::raw_ostream &os) {
            emitApplicationInitializersAssembly(model, os);
          })))
    return failure();
  if (failed(writeBinaryArtifact(
          packageArtifactPath(outputDirectory,
                              "data/" + name + ".initializers.bin"),
          initializers)) ||
      failed(writeApplicationManifest(model, outputDirectory)))
    return failure();
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

  TranslateFromMLIRRegistration weightsReg(
      "maps-to-magia-weights-asm",
      "emit an assembly wrapper for the canonical packed weights image",
      translateMapsToMagiaWeightsAssembly,
      [](DialectRegistry &registry) {
        registry.insert<maps::MapsDialect, func::FuncDialect,
                        tensor::TensorDialect, linalg::LinalgDialect,
                        math::MathDialect, scf::SCFDialect,
                        arith::ArithDialect>();
      });

  TranslateToMLIRRegistration packageReg(
      "json-to-magia-package",
      "import a MAPS deployment bundle and write all MAGIA runtime artifacts",
      translateJsonToMagiaPackage,
      [](DialectRegistry &registry) {
        registry.insert<maps::MapsDialect, func::FuncDialect,
                        tensor::TensorDialect, linalg::LinalgDialect,
                        math::MathDialect, scf::SCFDialect,
                        arith::ArithDialect>();
      });
}

} // namespace mlir::maps
