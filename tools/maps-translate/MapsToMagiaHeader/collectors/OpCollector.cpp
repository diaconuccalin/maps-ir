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

static std::optional<GeneratedOpKind> getKernelOpKind(StringRef kind) {
  if (kind == "MATMUL")
    return GeneratedOpKind::Matmul;
  if (kind == "DEPTHWISE_CONV")
    return GeneratedOpKind::DepthwiseConv;
  if (kind == "ABS")
    return GeneratedOpKind::Abs;
  if (kind == "ADD")
    return GeneratedOpKind::Add;
  if (kind == "DIV")
    return GeneratedOpKind::Div;
  if (kind == "EXP")
    return GeneratedOpKind::Exp;
  if (kind == "GROUP_NORMALIZE")
    return GeneratedOpKind::GroupNormalize;
  if (kind == "GROUP_REDUCE")
    return GeneratedOpKind::GroupReduce;
  if (kind == "LOG")
    return GeneratedOpKind::Log;
  if (kind == "MUL")
    return GeneratedOpKind::Mul;
  if (kind == "NEG")
    return GeneratedOpKind::Neg;
  if (kind == "POW")
    return GeneratedOpKind::Pow;
  if (kind == "REDUCE_MAX")
    return GeneratedOpKind::ReduceMax;
  if (kind == "REDUCE_SUM")
    return GeneratedOpKind::ReduceSum;
  if (kind == "RELU")
    return GeneratedOpKind::Relu;
  if (kind == "RESHAPE")
    return GeneratedOpKind::Reshape;
  if (kind == "SIGMOID")
    return GeneratedOpKind::Sigmoid;
  if (kind == "SLICE")
    return GeneratedOpKind::Slice;
  if (kind == "SQRT")
    return GeneratedOpKind::Sqrt;
  if (kind == "SUB")
    return GeneratedOpKind::Sub;
  if (kind == "TRANSPOSE")
    return GeneratedOpKind::Transpose;
  if (kind == "AllReduceMax")
    return GeneratedOpKind::AllReduceMax;
  if (kind == "AllReduceSum")
    return GeneratedOpKind::AllReduceSum;
  return std::nullopt;
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
  bool isConv2D = name == "maps.conv2d";
  bool isKernel = name == "maps.kernel";
  if (!isMatmul && !isGeneric && !isConv2D && !isKernel)
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
  if (isKernel) {
    auto kind = op->getAttrOfType<StringAttr>("kind");
    auto params = op->getAttrOfType<DenseI64ArrayAttr>("params");
    auto generatedKind = kind ? getKernelOpKind(kind.getValue()) : std::nullopt;
    if (!generatedKind || !params || params.size() > 8)
      return failure();
    generatedOp.kind = *generatedKind;
    for (auto indexed : llvm::enumerate(params.asArrayRef()))
      generatedOp.params[indexed.index()] =
          static_cast<uint32_t>(indexed.value());
  } else if (isConv2D) {
    generatedOp.kind = GeneratedOpKind::Conv2D;
    if (op->getNumOperands() < 2 || op->getNumOperands() > 3)
      return failure();
    auto strides = getDenseI64Array(op, "strides");
    auto dilations = getDenseI64Array(op, "dilations");
    auto padding = getDenseI64Array(op, "padding");
    if (strides.size() != 2 || dilations.size() != 2 ||
        padding.size() != 4)
      return failure();
    generatedOp.params[0] = static_cast<uint32_t>(strides[0]);
    generatedOp.params[1] = static_cast<uint32_t>(strides[1]);
    generatedOp.params[2] = static_cast<uint32_t>(dilations[0]);
    generatedOp.params[3] = static_cast<uint32_t>(dilations[1]);
    for (unsigned i = 0; i < 4; ++i)
      generatedOp.params[i + 4] = static_cast<uint32_t>(padding[i]);
  } else if (isMatmul) {
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
  unsigned numInputs =
      (isConv2D || isKernel) ? op->getNumOperands()
                            : op->getNumOperands() - numOutputs;
  for (unsigned i = 0; i < numInputs; ++i) {
    auto input = state.valueSubsliceIndices.find(op->getOperand(i));
    if (input == state.valueSubsliceIndices.end())
      return failure();
    generatedOp.inputSubsliceIndices.push_back(input->second);
  }

  std::string valueKey = getProducedValueKey(state, op->getResult(0));
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
