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

} // namespace mlir::maps

#endif
