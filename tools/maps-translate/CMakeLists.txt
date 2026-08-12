set(MAPS_MAGIA_CODEGEN_SOURCES
  MapsToMagiaHeader/collectors/CollectorSupport.cpp
  MapsToMagiaHeader/Emitter.cpp
  MapsToMagiaHeader/IRCollector.cpp
  MapsToMagiaHeader/collectors/L2Collector.cpp
  MapsToMagiaHeader/Model.cpp
  MapsToMagiaHeader/collectors/OpCollector.cpp
  MapsToMagiaHeader/collectors/SliceCollector.cpp
  MapsToMagiaHeader/Translation.cpp
  MapsToMagiaHeader/collectors/TransferCollector.cpp
)

add_llvm_executable(maps-translate
  maps-translate.cpp
  JsonToMaps.cpp
  ${MAPS_MAGIA_CODEGEN_SOURCES}
  PARTIAL_SOURCES_INTENDED
)

add_llvm_executable(maps-plan-import
  maps-plan-import.cpp
  JsonToMaps.cpp
  PARTIAL_SOURCES_INTENDED
)

add_llvm_executable(maps-codegen
  maps-codegen.cpp
  JsonToMaps.cpp
  ${MAPS_MAGIA_CODEGEN_SOURCES}
  PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(maps-translate maps-plan-import maps-codegen)

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

target_link_libraries(maps-translate ${MAPS_TRANSLATION_LIBRARIES})
target_link_libraries(maps-plan-import ${MAPS_TRANSLATION_LIBRARIES})
target_link_libraries(maps-codegen ${MAPS_TRANSLATION_LIBRARIES} MLIRParser)
