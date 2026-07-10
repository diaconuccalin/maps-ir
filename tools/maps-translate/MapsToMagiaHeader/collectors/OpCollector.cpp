#include "OpCollector.h"

#include "SliceCollector.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/Twine.h"

#include <optional>

namespace mlir::maps {

static std::optional<GeneratedOpKind> getGenericOpKind(Operation *genericOp) {
  std::optional<GeneratedOpKind> kind;

  genericOp->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "arith.addf")
      kind = GeneratedOpKind::Add;
    if (name == "math.log")
      kind = GeneratedOpKind::Log;
    if (name == "math.exp")
      kind = GeneratedOpKind::Exp;
  });

  return kind;
}

static void fillElementwiseParams(GeneratedOp &generatedOp,
                                  RankedTensorType outputType) {
  ArrayRef<int64_t> shape = outputType.getShape();
  generatedOp.params[0] = shape.empty() ? 1u : static_cast<uint32_t>(shape[0]);
  generatedOp.params[1] =
      shape.size() < 2 ? 1u : static_cast<uint32_t>(shape[1]);
  generatedOp.params[2] = static_cast<uint32_t>(product(shape));
}

LogicalResult collectComputeOp(Operation *op, RunCollectionState &state,
                               MagiaHeaderModel &model) {
  StringRef name = op->getName().getStringRef();
  bool isMatmul = name == "linalg.matmul";
  bool isGeneric = name == "linalg.generic";
  if (!isMatmul && !isGeneric)
    return success();

  Operation *tileOp = findParentNamed(op, "maps.tile");
  if (!tileOp)
    return failure();

  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  if (!tileId)
    return failure();

  if (op->getNumResults() == 0)
    return failure();

  auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType)
    return failure();

  GeneratedOp generatedOp;
  if (isMatmul) {
    generatedOp.kind = GeneratedOpKind::Matmul;
    if (op->getNumOperands() < 2)
      return failure();
    auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!lhsType || !rhsType || lhsType.getRank() < 2 || rhsType.getRank() < 2)
      return failure();
    generatedOp.params[0] = static_cast<uint32_t>(lhsType.getShape()[0]);
    generatedOp.params[1] = static_cast<uint32_t>(lhsType.getShape()[1]);
    generatedOp.params[2] = static_cast<uint32_t>(rhsType.getShape()[1]);
  } else {
    std::optional<GeneratedOpKind> kind = getGenericOpKind(op);
    if (!kind)
      return success();
    generatedOp.kind = *kind;
    fillElementwiseParams(generatedOp, outputType);
  }

  unsigned numOutputs = op->getNumResults();
  unsigned numInputs = op->getNumOperands() - numOutputs;
  for (unsigned i = 0; i < numInputs; ++i) {
    auto input = state.valueSubsliceIndices.find(op->getOperand(i));
    if (input == state.valueSubsliceIndices.end())
      return failure();
    generatedOp.inputSubsliceIndices.push_back(input->second);
  }

  std::string valueKey = getProducedValueKey(op->getResult(0));
  std::string uniqueKey =
      (Twine("op:") + Twine(tileId.getInt()) + ":" + valueKey).str();
  unsigned outputSliceIndex = getOrCreateProducedSlice(
      model, state.producedSliceIndices, uniqueKey, valueKey,
      GeneratedGlobalKind::Intermediate, tileId.getInt(), outputType, {});
  if (outputSliceIndex == std::numeric_limits<unsigned>::max())
    return failure();

  unsigned outputSubsliceIndex =
      addWholeSubsliceForSlice(model, outputSliceIndex);
  generatedOp.outputSubsliceIndices.push_back(outputSubsliceIndex);
  state.valueSubsliceIndices[op->getResult(0)] = outputSubsliceIndex;

  unsigned opIndex = addGeneratedOp(model, generatedOp);
  noteTile(model, tileId.getInt());
  model.tilePlans[tileId.getInt()].opIndices.push_back(opIndex);
  return success();
}

} // namespace mlir::maps
