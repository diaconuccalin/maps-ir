#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_EMITTER_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_EMITTER_H

#include "ApplicationDescription.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::maps {

void emitApplicationHeader(const MagiaApplicationDescription &model,
                           llvm::raw_ostream &os);
void emitApplicationDataSource(const MagiaApplicationDescription &model,
                                llvm::raw_ostream &os);
void emitApplicationTileSource(const MagiaApplicationDescription &model,
                               const GeneratedTilePlan &plan,
                               llvm::raw_ostream &os);
void emitApplicationRunner(const MagiaApplicationDescription &model,
                           llvm::raw_ostream &os);
void emitInitialApplicationSource(const MagiaApplicationDescription &model,
                                  llvm::raw_ostream &os);
void emitApplicationCMake(const MagiaApplicationDescription &model,
                          llvm::raw_ostream &os);
void emitApplicationReadme(const MagiaApplicationDescription &model,
                           llvm::raw_ostream &os);
void emitApplicationInitializersAssembly(const MagiaApplicationDescription &model,
                                         llvm::raw_ostream &os);
std::string getApplicationTileFilename(const MagiaApplicationDescription &model,
                                       int64_t hartid);

} // namespace mlir::maps

#endif
