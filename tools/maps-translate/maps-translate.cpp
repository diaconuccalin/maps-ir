#include "JsonToMaps.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir::maps {
void registerMapsToMagiaHeaderTranslation();
}

int main(int argc, char **argv) {
  mlir::maps::registerJsonToMapsTranslation();
  mlir::maps::registerMapsToMagiaHeaderTranslation();
  
  return failed(mlir::mlirTranslateMain(argc, argv, "MAPS translation-driver"));
}
