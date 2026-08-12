#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_EMITTER_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAPSTOMAGIAHEADER_EMITTER_H

#include "Model.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::maps {

// Emits the currently debug-oriented MAGIA SDK header from the collected model.
void emitDebugHeader(const MagiaHeaderModel &model, llvm::raw_ostream &os);
void emitDataFile(const MagiaHeaderModel &model, llvm::raw_ostream &os);
void emitWeightsAssembly(const MagiaHeaderModel &model, llvm::raw_ostream &os);
void emitApplicationHeader(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os);
void emitApplicationModelSource(const MagiaHeaderModel &model,
                                llvm::raw_ostream &os);
void emitApplicationTileSource(const MagiaHeaderModel &model,
                               const GeneratedTilePlan &plan,
                               llvm::raw_ostream &os);
void emitApplicationRunner(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os);
void emitInitialApplicationSource(const MagiaHeaderModel &model,
                                  llvm::raw_ostream &os);
void emitApplicationCMake(const MagiaHeaderModel &model, llvm::raw_ostream &os);
void emitApplicationReadme(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os);
void emitApplicationInitializersAssembly(const MagiaHeaderModel &model,
                                         llvm::raw_ostream &os);
std::string getApplicationTileFilename(const MagiaHeaderModel &model,
                                       int64_t hartid);

} // namespace mlir::maps

#endif
