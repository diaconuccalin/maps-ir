set(MAPS_MAGIA_CODEGEN_SOURCES
  MagiaApplicationCodegen/collectors/CollectorSupport.cpp
  MagiaApplicationCodegen/Emitter.cpp
  MagiaApplicationCodegen/IRCollector.cpp
  MagiaApplicationCodegen/collectors/L2Collector.cpp
  MagiaApplicationCodegen/ApplicationDescription.cpp
  MagiaApplicationCodegen/collectors/OpCollector.cpp
  MagiaApplicationCodegen/collectors/SliceCollector.cpp
  MagiaApplicationCodegen/CodeGeneration.cpp
  MagiaApplicationCodegen/collectors/TransferCollector.cpp
)

add_llvm_executable(maps-plan-import
  maps-plan-import.cpp
  ExecutionPlanImporter.cpp
  PARTIAL_SOURCES_INTENDED
)

add_llvm_executable(maps-codegen
  maps-codegen.cpp
  ${MAPS_MAGIA_CODEGEN_SOURCES}
  PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(maps-plan-import maps-codegen)

set(MAPS_TRANSLATION_LIBRARIES
  PRIVATE
  MLIRMapsDialect
  MLIRTranslateLib
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRArithDialect
)

target_link_libraries(maps-plan-import ${MAPS_TRANSLATION_LIBRARIES})
target_link_libraries(maps-codegen ${MAPS_TRANSLATION_LIBRARIES} MLIRParser)
