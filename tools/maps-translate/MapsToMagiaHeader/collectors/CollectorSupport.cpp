#include "CollectorSupport.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/Twine.h"


namespace mlir::maps {


Operation *findParentNamed(Operation *op, StringRef name) {
  /*
  Go up the mlir tree until finding a parent operation that matches the string ref name
  e.g.

  module {
    maps.program {
      maps.run {
        maps.tile @tile24 {
          %x = maps.recv ...
        }
      }
    }
  }

  findParentNamed(recvOp, "maps.tile")

  will walk up until reaching the maps.tile op

  */
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp()) {

    if (parent->getName().getStringRef() == name)
      return parent;

  }
  return nullptr;
}


std::string getFlatSymbolAttr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<FlatSymbolRefAttr>(name))
    return attr.getValue().str();
  return "";
}

SmallVector<int64_t> getDenseI64Array(Operation *op, StringRef name) {
  SmallVector<int64_t> values;

  if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name)) {
    values.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
    return values;
  }

  if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return {};
      values.push_back(intAttr.getInt());
    }
  }

  return values;
}

std::string getGlobalNameForValue(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return ("arg" + Twine(blockArg.getArgNumber())).str();

  return "";
}

std::string getChannelBaseName(StringRef channel) {
  size_t fromTile = channel.find("_from_tile");
  if (fromTile != StringRef::npos)
    return channel.take_front(fromTile).str();

  return channel.str();
}

std::string getTileSlotKey(int64_t hartid, StringRef slot) {
  return (Twine(hartid) + ":" + slot).str();
}

std::string getAggregateSliceKey(int64_t hartid, StringRef baseName,
                                 ArrayRef<int64_t> parentSlice) {
  std::string key = (Twine(hartid) + ":" + baseName).str();
  for (int64_t value : parentSlice)
    key += (Twine(":") + Twine(value)).str();
  return key;
}

SmallVector<int64_t> getPairwiseOffsets(ArrayRef<int64_t> values) {
  SmallVector<int64_t> offsets;
  for (size_t i = 0; i < values.size(); i += 2)
    offsets.push_back(values[i]);
  return offsets;
}

SmallVector<int64_t> getPairwiseLengths(ArrayRef<int64_t> values) {
  SmallVector<int64_t> lengths;
  for (size_t i = 1; i < values.size(); i += 2)
    lengths.push_back(values[i]);
  return lengths;
}

std::string getProducedValueKey(Value value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  value.printAsOperand(os, OpPrintingFlags());
  return storage;
}

} // namespace mlir::maps
