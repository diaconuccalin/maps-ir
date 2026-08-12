#include "Emitter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace mlir::maps {

static std::string getPaddedPhysicalTileId(const MagiaHeaderModel &model,
                                           int64_t hartid) {
  std::string id = std::to_string(hartid);
  uint64_t physicalTileCount =
      static_cast<uint64_t>(model.meshWidth) * model.meshHeight;
  size_t width = std::max<size_t>(
      2, std::to_string(physicalTileCount - 1).size());
  if (id.size() < width)
    id.insert(0, width - id.size(), '0');
  return id;
}

std::string getApplicationTileFilename(const MagiaHeaderModel &model,
                                       int64_t hartid) {
  return "tile_" + getPaddedPhysicalTileId(model, hartid) + ".c";
}

static void emitGeneratedPrelude(const MagiaHeaderModel &model,
                                 llvm::raw_ostream &os) {
  os << "#include <stdint.h>\n\n";
  os << "#include \"tile.h\"\n";
  os << "#include \"utils/maps_operations.h\"\n";
  os << "#include \"utils/maps_utils_v2.h\"\n\n";
  os << "#define MAPS_GENERATED_NUM_TOKENS "
     << model.numTokens << "u\n";
  os << "#define MAPS_GENERATED_NUM_TOKEN_SLOTS "
     << model.numTokenSlots << "u\n";
  os << "#define MAPS_GENERATED_L1_DATA_OFFSET "
     << kGeneratedL1DataBaseOffset << "u\n";
  os << "#define MAPS_GENERATED_READY_FLAGS_OFFSET "
     << kGeneratedReadyFlagsOffset << "u\n";
  os << "#define MAPS_GENERATED_READY_FLAGS_WORDS "
     << kGeneratedReadyFlagsWords << "u\n\n";
  os << "#ifndef MAPS_NUM_TOKENS\n";
  os << "#define MAPS_NUM_TOKENS MAPS_GENERATED_NUM_TOKENS\n";
  os << "#endif\n\n";
  os << "#ifdef MAPS_NUM_TOKEN_SLOTS\n";
  os << "#if MAPS_NUM_TOKEN_SLOTS != MAPS_GENERATED_NUM_TOKEN_SLOTS\n";
  os << "#error \"MAPS_NUM_TOKEN_SLOTS disagrees with the planned execution contract\"\n";
  os << "#endif\n";
  os << "#else\n";
  os << "#define MAPS_NUM_TOKEN_SLOTS MAPS_GENERATED_NUM_TOKEN_SLOTS\n";
  os << "#endif\n\n";
}

static std::string getL2GlobalSymbol(const GeneratedL2Global &global) {
  return (Twine("maps_generated_l2_global_") + Twine(global.modelIndex)).str();
}

static std::string getWeightsSymbol(const MagiaHeaderModel &model,
                                    StringRef suffix) {
  return (Twine(model.outputStem) + "_weights_" + suffix).str();
}

static std::string getRuntimeInputSymbol(const MagiaHeaderModel &model,
                                         const GeneratedL2Global &global,
                                         StringRef suffix) {
  return (Twine(model.outputStem) + "_input_" +
          normalizeSnakeCase(global.globalName) + "_" + suffix)
      .str();
}

static void emitL2GlobalAddress(const MagiaHeaderModel &model,
                                const GeneratedL2Global &global,
                                uint64_t offset, llvm::raw_ostream &os) {
  if (global.isConstant) {
    os << getWeightsSymbol(model, "start");
    uint64_t packedOffset = global.packedImageOffset + offset;
    if (packedOffset != 0)
      os << " + " << packedOffset << "u";
    return;
  }
  os << getL2GlobalSymbol(global);
  if (offset != 0)
    os << " + " << offset << "u";
}

static uint64_t getL2GlobalAllocationBytes(const GeneratedL2Global &global) {
  return global.tokenStrideBytes * global.numTokens;
}

static void emitL2GlobalDeclarations(const MagiaHeaderModel &model,
                                     llvm::raw_ostream &os) {
  if (model.hasPackedWeights) {
    os << "extern const uint8_t " << getWeightsSymbol(model, "start")
       << "[];\n";
    os << "extern const uint8_t " << getWeightsSymbol(model, "end")
       << "[];\n";
    os << "extern const uint8_t " << getWeightsSymbol(model, "size")
       << "[];\n";
  }
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.isConstant)
      continue;
    os << "extern uint8_t " << getL2GlobalSymbol(global) << "["
       << getL2GlobalAllocationBytes(global) << "];\n";
  }
  os << "void maps_generated_init_l2_data(uint32_t seed);\n\n";
  os << "static inline void maps_generated_init_tensors(uint32_t hartid) {\n";
  os << "  if (hartid == 0u) {\n";
  if (model.hasPackedWeights) {
    os << "    if ((((uintptr_t)" << getWeightsSymbol(model, "start")
       << ") & " << (model.weightsAlignment - 1) << "u) != 0u ||\n";
    os << "        (uintptr_t)" << getWeightsSymbol(model, "end")
       << " - (uintptr_t)" << getWeightsSymbol(model, "start") << " != "
       << model.weightsSize << "u)\n";
    os << "      maps_trap();\n";
  }
  os << "    maps_generated_init_l2_data(0x1234u);\n";
  os << "  }\n";
  os << "}\n\n";
}

static StringRef getGlobalKindName(GeneratedGlobalKind kind) {
  switch (kind) {
  case GeneratedGlobalKind::Input:
    return "GLOBAL_INPUT";
  case GeneratedGlobalKind::Output:
    return "GLOBAL_OUTPUT";
  case GeneratedGlobalKind::Initializer:
    return "GLOBAL_INITIALIZER";
  case GeneratedGlobalKind::Intermediate:
    return "GLOBAL_INTERMEDIATE";
  }
  llvm_unreachable("unknown global kind");
}

static StringRef getElemTypeName(Type type) {
  if (type.isF32())
    return "ELEM_F32";
  if (type.isF16())
    return "ELEM_F16";
  if (type.isInteger(32))
    return "ELEM_I32";
  if (type.isInteger(8))
    return "ELEM_U8";
  return "ELEM_F32";
}

static StringRef getOpKindName(GeneratedOpKind kind) {
  switch (kind) {
  case GeneratedOpKind::Matmul:
    return "OP_MATMUL";
  case GeneratedOpKind::Conv2D:
    return "OP_CONV2D";
  case GeneratedOpKind::DepthwiseConv:
    return "OP_DEPTHWISE_CONV";
  case GeneratedOpKind::Abs:
    return "OP_ABS";
  case GeneratedOpKind::Add:
    return "OP_ADD";
  case GeneratedOpKind::Div:
    return "OP_DIV";
  case GeneratedOpKind::Log:
    return "OP_LOG";
  case GeneratedOpKind::Exp:
    return "OP_EXP";
  case GeneratedOpKind::GroupNormalize:
    return "OP_GROUP_NORMALIZE";
  case GeneratedOpKind::GroupReduce:
    return "OP_GROUP_REDUCE";
  case GeneratedOpKind::Mul:
    return "OP_MUL";
  case GeneratedOpKind::Neg:
    return "OP_NEG";
  case GeneratedOpKind::Pow:
    return "OP_POW";
  case GeneratedOpKind::ReduceMax:
    return "OP_REDUCE_MAX";
  case GeneratedOpKind::ReduceSum:
    return "OP_REDUCE_SUM";
  case GeneratedOpKind::Relu:
    return "OP_RELU";
  case GeneratedOpKind::Reshape:
    return "OP_RESHAPE";
  case GeneratedOpKind::Sigmoid:
    return "OP_SIGMOID";
  case GeneratedOpKind::Slice:
    return "OP_SLICE";
  case GeneratedOpKind::Sqrt:
    return "OP_SQRT";
  case GeneratedOpKind::Sub:
    return "OP_SUB";
  case GeneratedOpKind::Transpose:
    return "OP_TRANSPOSE";
  case GeneratedOpKind::AllReduceMax:
    return "OP_ALL_REDUCE_MAX";
  case GeneratedOpKind::AllReduceSum:
    return "OP_ALL_REDUCE_SUM";
  case GeneratedOpKind::Copy:
    return "OP_COPY";
  }
  llvm_unreachable("unknown op kind");
}

// Keep descriptor array padding consistent with the MAPS tensor-rank limit.
static void emitPaddedI64Array(ArrayRef<int64_t> values,
                               llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < kGeneratedDescriptorMaxDims; ++i) {
    if (i != 0)
      os << ", ";
    os << (i < values.size() ? values[i] : 0);
  }
  os << "}";
}

static void emitPaddedU64Array(ArrayRef<uint64_t> values,
                               llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < kGeneratedDescriptorMaxDims; ++i) {
    if (i != 0)
      os << ", ";
    os << (i < values.size() ? values[i] : 0);
  }
  os << "}";
}

static void emitSliceDesc(const GeneratedSlice &slice,
                          llvm::raw_ostream &os) {
  os << "  {\n";
  os << "    .slice_id = " << slice.sliceId << ",\n";
  os << "    .global_id = " << slice.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(slice.globalKind) << ",\n";
  os << "    .owner_hartid = " << slice.ownerHartid << ",\n";
  os << "    .full_offset = ";
  emitPaddedI64Array(slice.fullOffset, os);
  os << ",\n";
  os << "    .rank = " << slice.shape.size() << ",\n";
  os << "    .shape = ";
  emitPaddedI64Array(slice.shape, os);
  os << ",\n";
  os << "    .elem_type = " << getElemTypeName(slice.elementType) << ",\n";
  os << "    .elem_bytes = " << slice.elemBytes << ",\n";
  os << "    .l1_offset_bytes = " << slice.l1OffsetBytes << ",\n";
  os << "    .slot_bytes = " << slice.slotBytes << ",\n";
  os << "    .num_slots = " << slice.numSlots << ",\n";
  os << "    .strides_bytes = ";
  emitPaddedU64Array(slice.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitSubsliceDesc(const MagiaHeaderModel &model,
                             const GeneratedSubslice &subslice,
                             llvm::raw_ostream &os) {
  const GeneratedSlice &slice = model.slices[subslice.sliceIndex];

  os << "{\n";
  os << "      .slice_id = " << slice.sliceId << ",\n";
  os << "      .offset = ";
  emitPaddedI64Array(subslice.offset, os);
  os << ",\n";
  os << "      .rank = " << subslice.shape.size() << ",\n";
  os << "      .shape = ";
  emitPaddedI64Array(subslice.shape, os);
  os << ",\n";
  os << "      .elem_type = " << getElemTypeName(subslice.elementType) << ",\n";
  os << "      .elem_bytes = " << subslice.elemBytes << ",\n";
  os << "      .strides_bytes = ";
  emitPaddedU64Array(subslice.stridesBytes, os);
  os << ",\n";
  os << "    }";
}

static void emitTensorSubSlice(ArrayRef<int64_t> shape,
                               ArrayRef<uint64_t> stridesBytes,
                               llvm::raw_ostream &os) {
  os << "{\n";
  os << "      .rank = " << shape.size() << "u,\n";
  os << "      .num_elems = " << product(shape) << "u,\n";
  os << "      .dims = {";
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << "{0u, " << shape[i] << "u, " << stridesBytes[i] << "u}";
  }
  os << "},\n";
  os << "    }";
}

static void emitL2ReadDesc(const MagiaHeaderModel &model,
                           const GeneratedL2Read &read,
                           llvm::raw_ostream &os) {
  const GeneratedL2Global &global = model.l2Globals[read.l2GlobalIndex];
  const GeneratedSubslice &dst = model.subslices[read.dstSubsliceIndex];

  os << "  {\n";
  os << "    .global_id = " << global.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(global.globalKind) << ",\n";
  os << "    .src_l2_addr = (uint32_t)";
  uint64_t srcOffset = read.srcL2Address - global.l2Address;
  emitL2GlobalAddress(model, global, srcOffset, os);
  os << ",\n";
  os << "    .src_l2_token_stride_bytes = "
     << read.srcL2TokenStrideBytes << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(model, dst, os);
  os << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(dst.shape, global.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(dst.shape, dst.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitRecvDesc(const MagiaHeaderModel &model,
                         const GeneratedRecv &recv,
                         llvm::raw_ostream &os) {
  const GeneratedSubslice &dst = model.subslices[recv.dstSubsliceIndex];

  os << "  {\n";
  os << "    .transition_id = " << recv.transitionId << ",\n";
  os << "    .src_hartid = " << recv.srcHartid << ",\n";
  os << "    .dst_hartid = " << recv.dstHartid << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(model, dst, os);
  os << ",\n";
  os << "    .producer_idx = " << recv.producerIdx << ",\n";
  os << "  }";
}

static void emitSendDesc(const MagiaHeaderModel &model,
                         const GeneratedSend &send,
                         llvm::raw_ostream &os) {
  const GeneratedSubslice &src = model.subslices[send.srcSubsliceIndex];
  const GeneratedSubslice &dst = model.subslices[send.dstSubsliceIndex];

  os << "  {\n";
  os << "    .transition_id = " << send.transitionId << ",\n";
  os << "    .src_hartid = " << send.srcHartid << ",\n";
  os << "    .dst_hartid = " << send.dstHartid << ",\n";
  os << "    .src = ";
  emitSubsliceDesc(model, src, os);
  os << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(model, dst, os);
  os << ",\n";
  os << "    .dst_l1_data_base = " << send.dstL1DataBase << ",\n";
  os << "    .dst_slice_l1_offset_bytes = "
     << send.dstSliceL1OffsetBytes << ",\n";
  os << "    .dst_slice_slot_bytes = " << send.dstSliceSlotBytes << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(src.shape, src.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(dst.shape, dst.stridesBytes, os);
  os << ",\n";
  os << "    .producer_idx = " << send.producerIdx << ",\n";
  os << "  }";
}

static void emitSubsliceArray(const MagiaHeaderModel &model,
                              ArrayRef<unsigned> subsliceIndices,
                              unsigned paddedSize,
                              llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < paddedSize; ++i) {
    if (i != 0)
      os << ", ";
    if (i < subsliceIndices.size())
      emitSubsliceDesc(model, model.subslices[subsliceIndices[i]], os);
    else
      os << "{0}";
  }
  os << "}";
}

using CollectiveKey = std::tuple<unsigned, unsigned, unsigned>;

static CollectiveKey getCollectiveKey(const GeneratedOp &op) {
  return {op.collectiveIdentity.stageId, op.collectiveIdentity.layerIndex,
          op.collectiveIdentity.groupId};
}

static unsigned getCollectiveFlagBase(const MagiaHeaderModel &model,
                                      const GeneratedOp &target) {
  std::map<CollectiveKey, const GeneratedOp *> collectives;
  for (const GeneratedOp &op : model.ops)
    if (op.isCollective)
      collectives.try_emplace(getCollectiveKey(op), &op);

  unsigned flagBase = 0;
  for (const auto &[key, op] : collectives) {
    if (key == getCollectiveKey(target))
      return flagBase;
    flagBase += 2 * model.numTokenSlots * op->collectiveParticipants.size();
  }
  llvm_unreachable("collective operation has no stable flag allocation");
}

static const GeneratedOp &getParticipantCollectiveOp(
    const MagiaHeaderModel &model, const GeneratedOp &target,
    unsigned participantHartid) {
  for (const GeneratedOp &op : model.ops)
    if (op.isCollective && op.ownerHartid == participantHartid &&
        getCollectiveKey(op) == getCollectiveKey(target))
      return op;
  llvm_unreachable("collective participant has no operation descriptor");
}

static uint64_t getSubsliceL1Offset(const MagiaHeaderModel &model,
                                    unsigned subsliceIndex) {
  const GeneratedSubslice &subslice = model.subslices[subsliceIndex];
  const GeneratedSlice &slice = model.slices[subslice.sliceIndex];
  return slice.l1OffsetBytes +
         computeOffsetBytes(subslice.offset, subslice.stridesBytes);
}

static uint64_t getFifoSlotDataSize(const MagiaHeaderModel &model,
                                    const GeneratedTilePlan &plan);

static void emitL1DataBaseOffset(const MagiaHeaderModel &model,
                                 int64_t hartid, llvm::raw_ostream &os) {
  const GeneratedTilePlan &plan = model.tilePlans.find(hartid)->second;
  const uint64_t slotDataSize = getFifoSlotDataSize(model, plan);
  os << "FIFO_HEADER_SIZE + " << plan.recvIndices.size()
     << "u * FIFO_RING_STATE_SIZE + " << plan.recvIndices.size()
     << "u * MAPS_NUM_TOKEN_SLOTS * (FIFO_SLOT_META_SIZE + (("
     << slotDataSize << "u + 3u) & ~3u))";
}

static void emitOpDesc(const MagiaHeaderModel &model,
                       const GeneratedOp &op,
                       llvm::raw_ostream &os) {
  os << "  {\n";
  os << "    .kind = " << getOpKindName(op.kind) << ",\n";
  os << "    .num_inputs = " << op.inputSubsliceIndices.size() << "u,\n";
  os << "    .inputs = ";
  emitSubsliceArray(model, op.inputSubsliceIndices, 4, os);
  os << ",\n";
  os << "    .num_outputs = " << op.outputSubsliceIndices.size() << "u,\n";
  os << "    .outputs = ";
  emitSubsliceArray(model, op.outputSubsliceIndices, 2, os);
  os << ",\n";
  if (op.isCollective) {
    os << "    .collective = {.stage_id = " << op.collectiveIdentity.stageId
       << "u, .layer_index = " << op.collectiveIdentity.layerIndex
       << "u, .group_id = " << op.collectiveIdentity.groupId
       << "u, .flag_base = " << getCollectiveFlagBase(model, op)
       << "u, .num_token_slots = " << model.numTokenSlots
       << "u, .num_participants = " << op.collectiveParticipants.size()
       << "u, .participants = {";
    for (auto participant : llvm::enumerate(op.collectiveParticipants)) {
      if (participant.index() != 0)
        os << ", ";
      const GeneratedOp &participantOp = getParticipantCollectiveOp(
          model, op, participant.value());
      const GeneratedSubslice &input =
          model.subslices[participantOp.inputSubsliceIndices.front()];
      const GeneratedSubslice &output =
          model.subslices[participantOp.outputSubsliceIndices.front()];
      const GeneratedSlice &inputSlice = model.slices[input.sliceIndex];
      const GeneratedSlice &outputSlice = model.slices[output.sliceIndex];
      os << "{.hartid = " << participant.value()
         << "u, .l1_data_base_offset = ";
      emitL1DataBaseOffset(model, participant.value(), os);
      os << ", .input_l1_offset_bytes = "
         << getSubsliceL1Offset(model,
                                participantOp.inputSubsliceIndices.front())
         << "u, .input_slot_bytes = " << inputSlice.slotBytes
         << "u, .output_l1_offset_bytes = "
         << getSubsliceL1Offset(model,
                                participantOp.outputSubsliceIndices.front())
         << "u, .output_slot_bytes = " << outputSlice.slotBytes << "u}";
    }
    os << "}},\n";
  }
  os << "    .params = {";
  for (unsigned i = 0; i < 8; ++i) {
    if (i != 0)
      os << ", ";
    os << op.params[i] << "u";
  }
  os << "},\n";
  os << "  }";
}

static void emitL2WriteDesc(const MagiaHeaderModel &model,
                            const GeneratedL2Write &write,
                            llvm::raw_ostream &os) {
  const GeneratedL2Global &global = model.l2Globals[write.l2GlobalIndex];
  const GeneratedSubslice &src = model.subslices[write.srcSubsliceIndex];

  os << "  {\n";
  os << "    .global_id = " << global.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(global.globalKind) << ",\n";
  os << "    .src = ";
  emitSubsliceDesc(model, src, os);
  os << ",\n";
  os << "    .dst_l2_addr = (uint32_t)" << getL2GlobalSymbol(global);
  uint64_t dstOffset = write.dstL2Address - global.l2Address;
  if (dstOffset != 0)
    os << " + " << dstOffset << "u";
  os << ",\n";
  os << "    .dst_l2_token_stride_bytes = "
     << write.dstL2TokenStrideBytes << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(src.shape, src.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(src.shape, global.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitTileSliceDescriptors(const MagiaHeaderModel &model,
                                     const GeneratedTilePlan &plan,
                                     llvm::raw_ostream &os) {
  os << "static const slice_desc_t maps_generated_tile_"
     << plan.hartid << "_slices[] = {\n";
  for (unsigned sliceIndex : plan.sliceIndices) {
    emitSliceDesc(model.slices[sliceIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileL2ReadDescriptors(const MagiaHeaderModel &model,
                                      const GeneratedTilePlan &plan,
                                      llvm::raw_ostream &os) {
  if (!plan.initL2ReadIndices.empty()) {
    os << "static const l2_read_desc_t maps_generated_tile_"
       << plan.hartid << "_init_l2_reads[] = {\n";
    for (unsigned readIndex : plan.initL2ReadIndices) {
      emitL2ReadDesc(model, model.l2Reads[readIndex], os);
      os << ",\n";
    }
    os << "};\n\n";
  }

  if (!plan.l2ReadIndices.empty()) {
    os << "static const l2_read_desc_t maps_generated_tile_"
       << plan.hartid << "_l2_reads[] = {\n";
    for (unsigned readIndex : plan.l2ReadIndices) {
      emitL2ReadDesc(model, model.l2Reads[readIndex], os);
      os << ",\n";
    }
    os << "};\n\n";
  }
}

static void emitTileRecvDescriptors(const MagiaHeaderModel &model,
                                    const GeneratedTilePlan &plan,
                                    llvm::raw_ostream &os) {
  if (plan.recvIndices.empty())
    return;

  os << "static const fifo_recv_desc_t maps_generated_tile_"
     << plan.hartid << "_recvs[] = {\n";
  for (unsigned recvIndex : plan.recvIndices) {
    emitRecvDesc(model, model.recvs[recvIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileSendDescriptors(const MagiaHeaderModel &model,
                                    const GeneratedTilePlan &plan,
                                    llvm::raw_ostream &os) {
  if (plan.sendIndices.empty())
    return;

  os << "static const fifo_send_desc_t maps_generated_tile_"
     << plan.hartid << "_sends[] = {\n";
  for (unsigned sendIndex : plan.sendIndices) {
    emitSendDesc(model, model.sends[sendIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileOpDescriptors(const MagiaHeaderModel &model,
                                  const GeneratedTilePlan &plan,
                                  llvm::raw_ostream &os) {
  if (plan.opIndices.empty())
    return;

  os << "static const op_desc_t maps_generated_tile_"
     << plan.hartid << "_ops[] = {\n";
  for (unsigned opIndex : plan.opIndices) {
    emitOpDesc(model, model.ops[opIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileL2WriteDescriptors(const MagiaHeaderModel &model,
                                       const GeneratedTilePlan &plan,
                                       llvm::raw_ostream &os) {
  if (plan.l2WriteIndices.empty())
    return;

  os << "static const l2_write_desc_t maps_generated_tile_"
     << plan.hartid << "_l2_writes[] = {\n";
  for (unsigned writeIndex : plan.l2WriteIndices) {
    emitL2WriteDesc(model, model.l2Writes[writeIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static uint64_t getFifoSlotDataSize(const MagiaHeaderModel &model,
                                    const GeneratedTilePlan &plan) {
  uint64_t maxBytes = 0;
  for (unsigned recvIndex : plan.recvIndices) {
    const GeneratedRecv &recv = model.recvs[recvIndex];
    const GeneratedSubslice &dst = model.subslices[recv.dstSubsliceIndex];
    maxBytes = std::max(maxBytes, product(dst.shape) * dst.elemBytes);
  }
  return maxBytes;
}

static void emitFifoStorageHelper(llvm::raw_ostream &os) {
  os << "static inline uint32_t maps_generated_fifo_bytes(\n";
  os << "    uint32_t num_producers, uint32_t num_slots,\n";
  os << "    uint32_t slot_data_size) {\n";
  os << "  uint32_t slot_stride = FIFO_SLOT_META_SIZE + "
        "((slot_data_size + 3u) & ~3u);\n";
  os << "  return FIFO_HEADER_SIZE + num_producers * FIFO_RING_STATE_SIZE +\n";
  os << "         num_producers * num_slots * slot_stride;\n";
  os << "}\n\n";
}

static void emitPlanArrayAssignment(StringRef fieldName, StringRef arraySuffix,
                                    unsigned count, int64_t hartid,
                                    llvm::raw_ostream &os) {
  if (count == 0)
    return;

  os << "    plan->num_" << fieldName << " = " << count << "u;\n";
  os << "    plan->" << fieldName << " = maps_generated_tile_"
     << hartid << "_" << arraySuffix << ";\n";
}

static void emitFillPlan(const MagiaHeaderModel &model,
                         llvm::raw_ostream &os) {
  os << "static inline uint32_t maps_generated_fill_plan(\n";
  os << "    fifo_tile_plan_t *plan, uint32_t hartid,\n";
  os << "    maps_operation_runtime_t *runtime) {\n";
  os << "  *plan = (fifo_tile_plan_t){\n";
  os << "    .hartid = hartid,\n";
  os << "    .l1_data_base = get_l1_base(hartid),\n";
  os << "    .ready_flags_base = MAPS_GENERATED_READY_FLAGS_OFFSET,\n";
  os << "    .ready_flags_count = MAPS_GENERATED_READY_FLAGS_WORDS,\n";
  os << "    .num_token_slots = MAPS_NUM_TOKEN_SLOTS,\n";
  os << "    .operation_runtime = runtime,\n";
  os << "  };\n\n";
  os << "  switch (hartid) {\n";

  for (const auto &entry : model.tilePlans) {
    const GeneratedTilePlan &plan = entry.second;
    uint64_t fifoSlotDataSize = getFifoSlotDataSize(model, plan);
    os << "  case " << plan.hartid << "u:\n";
    os << "    plan->fifo = (fifo_desc_t){\n";
    os << "      .l1_offset_bytes = 0u,\n";
    os << "      .num_producers = " << plan.recvIndices.size() << "u,\n";
    os << "      .num_slots = MAPS_NUM_TOKEN_SLOTS,\n";
    os << "      .slot_data_size = " << fifoSlotDataSize << "u,\n";
    os << "    };\n";
    os << "    plan->l1_data_base += maps_generated_fifo_bytes(\n";
    os << "        plan->fifo.num_producers, plan->fifo.num_slots,\n";
    os << "        plan->fifo.slot_data_size);\n";
    emitPlanArrayAssignment("slices", "slices", plan.sliceIndices.size(),
                            plan.hartid, os);
    emitPlanArrayAssignment("init_l2_reads", "init_l2_reads",
                            plan.initL2ReadIndices.size(), plan.hartid, os);
    emitPlanArrayAssignment("l2_reads", "l2_reads",
                            plan.l2ReadIndices.size(), plan.hartid, os);
    emitPlanArrayAssignment("recvs", "recvs", plan.recvIndices.size(),
                            plan.hartid, os);
    emitPlanArrayAssignment("ops", "ops", plan.opIndices.size(),
                            plan.hartid, os);
    emitPlanArrayAssignment("sends", "sends", plan.sendIndices.size(),
                            plan.hartid, os);
    emitPlanArrayAssignment("l2_writes", "l2_writes",
                            plan.l2WriteIndices.size(), plan.hartid, os);
    os << "    return 1u;\n";
  }

  os << "  default:\n";
  os << "    return 0u;\n";
  os << "  }\n";
  os << "}\n\n";
}

static void emitFirstOutputChecker(const MagiaHeaderModel &model,
                                   llvm::raw_ostream &os) {
  const GeneratedL2Global *firstOutput = nullptr;
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind == GeneratedGlobalKind::Output) {
      firstOutput = &global;
      break;
    }
  }

  if (!firstOutput) {
    os << "static inline uint32_t maps_generated_check_output(void) {\n";
    os << "  return 0u;\n";
    os << "}\n\n";
    return;
  }

  std::string symbol = getL2GlobalSymbol(*firstOutput);
  os << "#define MAPS_GENERATED_FIRST_OUTPUT_TOKEN_BYTES "
     << firstOutput->tensorBytes << "u\n";
  os << "#define MAPS_GENERATED_FIRST_OUTPUT_TOKEN_STRIDE_BYTES "
     << firstOutput->tokenStrideBytes << "u\n";
  os << "#ifndef MAPS_GENERATED_FIRST_OUTPUT_TOKEN0_CHECKSUM\n";
  os << "#define MAPS_GENERATED_FIRST_OUTPUT_TOKEN0_CHECKSUM 0u\n";
  os << "#endif\n\n";

  os << "static inline uint32_t "
        "maps_generated_first_output_token0_checksum(void) {\n";
  os << "  uint32_t checksum = 2166136261u;\n";
  os << "  volatile const uint8_t *out = "
     << "(volatile const uint8_t *)" << symbol << ";\n\n";
  os << "  for (uint32_t i = 0u; "
        "i < MAPS_GENERATED_FIRST_OUTPUT_TOKEN_BYTES; ++i) {\n";
  os << "    checksum ^= out[i];\n";
  os << "    checksum *= 16777619u;\n";
  os << "  }\n";
  os << "  return checksum;\n";
  os << "}\n\n";

  os << "static inline uint32_t maps_generated_check_output(void) {\n";
  os << "#if MAPS_GENERATED_FIRST_OUTPUT_TOKEN0_CHECKSUM != 0u\n";
  os << "  return maps_generated_first_output_token0_checksum() !=\n";
  os << "         MAPS_GENERATED_FIRST_OUTPUT_TOKEN0_CHECKSUM;\n";
  os << "#else\n";
  os << "  uint32_t seenNonzero = 0u;\n";
  os << "  volatile const uint8_t *out = "
     << "(volatile const uint8_t *)" << symbol << ";\n\n";
  os << "  for (uint32_t i = 0u; "
        "i < MAPS_GENERATED_FIRST_OUTPUT_TOKEN_BYTES; ++i)\n";
  os << "    seenNonzero |= out[i];\n";
  os << "  if (seenNonzero == 0u)\n";
  os << "    return 1u;\n\n";
  os << "  return 0u;\n";
  os << "#endif\n";
  os << "}\n\n";
}

[[maybe_unused]] static void emitTileSliceDebug(const MagiaHeaderModel &model,
                                                llvm::raw_ostream &os) {

  os << "/* Debug MAPS -> Magia header model */\n";
  os << "/* tiles: " << model.tilePlans.size() << " */\n";
  os << "/* l2 globals: " << model.l2Globals.size()
     << ", l2 bytes used: " << (model.nextL2Address - kMagiaL2Base)
     << " */\n";
  for (const auto &entry : model.tilePlans) {
    const GeneratedTilePlan &plan = entry.second;
    os << "/* tile " << plan.hartid << ": " << plan.sliceIndices.size()
       << " slices, " << plan.initL2ReadIndices.size()
       << " init_l2_reads, " << plan.l2ReadIndices.size()
       << " l2_reads, " << plan.recvIndices.size()
       << " recvs, " << plan.opIndices.size()
       << " ops, " << plan.sendIndices.size()
       << " sends, " << plan.l2WriteIndices.size()
       << " l2_writes, l1_data_bytes = " << plan.l1DataBytes << " */\n";

    for (unsigned sliceIndex : plan.sliceIndices) {
      const GeneratedSlice &slice = model.slices[sliceIndex];
      os << "/*   slice model_index = " << slice.modelIndex
         << ", slice_id = " << slice.sliceId
         << ", global_id = " << slice.globalId
         << ", kind = " << getGlobalKindName(slice.globalKind)
         << ", l1_offset_bytes = " << slice.l1OffsetBytes << " */\n";
    }
  }
}

void emitDebugHeader(const MagiaHeaderModel &model, llvm::raw_ostream &os) {
  os << "#ifndef MAPS_GENERATED_TEST_H\n";
  os << "#define MAPS_GENERATED_TEST_H\n\n";

  emitGeneratedPrelude(model, os);
  emitL2GlobalDeclarations(model, os);
  for (const GeneratedTilePlan *plan : getSortedTilePlans(model)) {
    emitTileSliceDescriptors(model, *plan, os);
    emitTileL2ReadDescriptors(model, *plan, os);
    emitTileRecvDescriptors(model, *plan, os);
    emitTileOpDescriptors(model, *plan, os);
    emitTileSendDescriptors(model, *plan, os);
    emitTileL2WriteDescriptors(model, *plan, os);
  }
  emitFifoStorageHelper(os);
  emitFillPlan(model, os);
  emitFirstOutputChecker(model, os);

  //emitTileSliceDebug(model, os);

  os << "\n#endif /* MAPS_GENERATED_TEST_H */\n";
}

void emitDataFile(const MagiaHeaderModel &model, llvm::raw_ostream &os) {
  os << "#include <stdint.h>\n";
  os << "#include <stddef.h>\n\n";

  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.isConstant)
      continue;
    os << "uint8_t " << getL2GlobalSymbol(global) << "["
       << getL2GlobalAllocationBytes(global)
       << "] __attribute__((aligned(16)));\n";
  }

  if (model.randomizeInputs) {
    os << "\nstatic uint32_t maps_generated_lcg(uint32_t *state) {\n";
    os << "  *state = *state * 1664525u + 1013904223u;\n";
    os << "  return *state;\n";
    os << "}\n\n";
  }

  os << "static void maps_generated_zero_bytes(uint8_t *dst, size_t bytes) {\n";
  os << "  for (size_t i = 0; i < bytes; ++i)\n";
  os << "    dst[i] = 0u;\n";
  os << "}\n\n";

  if (model.randomizeInputs) {
    os << "static void maps_generated_fill_f16(uint8_t *dst, size_t bytes,\n";
    os << "                                    uint32_t *state) {\n";
    os << "  static const uint16_t values[] = {\n";
    os << "      0x3c00u, 0x4000u, 0x4200u, 0x4400u,\n";
    os << "      0x4500u, 0x4600u, 0x4700u, 0x4800u,\n";
    os << "  };\n";
    os << "  uint16_t *half = (uint16_t *)dst;\n";
    os << "  size_t elems = bytes / sizeof(uint16_t);\n";
    os << "  for (size_t i = 0; i < elems; ++i)\n";
    os << "    half[i] = values[(maps_generated_lcg(state) >> 24) & 7u];\n";
    os << "}\n\n";

    os << "static void maps_generated_fill_bytes(uint8_t *dst, size_t bytes,\n";
    os << "                                      uint32_t *state) {\n";
    os << "  for (size_t i = 0; i < bytes; ++i)\n";
    os << "    dst[i] = (uint8_t)(maps_generated_lcg(state) >> 24);\n";
    os << "}\n\n";
  }

  os << "void maps_generated_init_l2_data(uint32_t seed) {\n";
  if (model.randomizeInputs)
    os << "  uint32_t state = seed;\n";
  else
    os << "  (void)seed;\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.isConstant)
      continue;
    os << "\n";
    if (model.randomizeInputs &&
        (global.globalKind == GeneratedGlobalKind::Input ||
         global.globalKind == GeneratedGlobalKind::Initializer)) {
      if (global.elemBytes == 2) {
        os << "  maps_generated_fill_f16(" << getL2GlobalSymbol(global)
           << ", sizeof(" << getL2GlobalSymbol(global) << "), &state);\n";
      } else {
        os << "  maps_generated_fill_bytes(" << getL2GlobalSymbol(global)
           << ", sizeof(" << getL2GlobalSymbol(global) << "), &state);\n";
      }
    } else {
      os << "  maps_generated_zero_bytes(" << getL2GlobalSymbol(global)
         << ", sizeof(" << getL2GlobalSymbol(global) << "));\n";
    }
  }
  os << "}\n";
}

void emitWeightsAssembly(const MagiaHeaderModel &model,
                         llvm::raw_ostream &os) {
  if (!model.hasPackedWeights)
    return;

  std::string start = getWeightsSymbol(model, "start");
  std::string end = getWeightsSymbol(model, "end");
  std::string size = getWeightsSymbol(model, "size");
  os << ".section .maps_weights,\"a\",@progbits\n";
  os << ".balign " << model.weightsAlignment << "\n";
  os << ".global " << start << "\n";
  os << ".global " << end << "\n";
  os << ".global " << size << "\n";
  os << start << ":\n";
  os << ".incbin \"" << model.weightsFile << "\"\n";
  os << end << ":\n";
  os << ".set " << size << ", " << end << " - " << start << "\n";
}

static std::string uppercase(StringRef value) {
  std::string result = value.str();
  for (char &character : result)
    character = llvm::toUpper(character);
  return result;
}

static std::string escapePrintfText(StringRef value) {
  std::string result;
  for (unsigned char character : value.bytes()) {
    switch (character) {
    case '%':
      result += "%%";
      break;
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character >= 0x20 && character <= 0x7e) {
        result += static_cast<char>(character);
      } else {
        result += '\\';
        result += static_cast<char>('0' + ((character >> 6) & 7));
        result += static_cast<char>('0' + ((character >> 3) & 7));
        result += static_cast<char>('0' + (character & 7));
      }
    }
  }
  return result;
}

static void replaceAll(std::string &text, StringRef from, StringRef to) {
  size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
}

static std::string
emitWithApplicationSymbols(const MagiaHeaderModel &model,
                           llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  emit(stream);
  stream.flush();
  replaceAll(text, "MAPS_GENERATED", uppercase(model.outputStem));
  replaceAll(text, "maps_generated", model.outputStem);
  replaceAll(text, model.outputStem + "_weights_",
             model.outputStem + "_initializers_");
  return text;
}

void emitApplicationHeader(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os) {
  std::string guard = uppercase(model.outputStem) + "_H";
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#include <stdint.h>\n\n";
  os << "#define " << uppercase(model.outputStem) << "_EXECUTION_TOKENS "
     << model.numTokens << "u\n";
  os << "#define " << uppercase(model.outputStem) << "_TOKEN_SLOTS "
     << model.numTokenSlots << "u\n\n";
  os << "int " << model.outputStem << "_run(void);\n";
  os << "void " << model.outputStem << "_handle_input(uint32_t token);\n";
  os << "void " << model.outputStem << "_handle_output(uint32_t token);\n\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind == GeneratedGlobalKind::Input)
      os << "uint8_t *" << model.outputStem << "_input_"
         << normalizeSnakeCase(global.globalName) << "(uint32_t token);\n";
    if (global.globalKind == GeneratedGlobalKind::Output)
      os << "const uint8_t *" << model.outputStem << "_output_"
         << normalizeSnakeCase(global.globalName) << "(uint32_t token);\n";
  }
  os << "\n#endif /* " << guard << " */\n";
}

void emitApplicationModelSource(const MagiaHeaderModel &model,
                                llvm::raw_ostream &os) {
  os << emitWithApplicationSymbols(model, [&](llvm::raw_ostream &stream) {
    emitGeneratedPrelude(model, stream);
    emitL2GlobalDeclarations(model, stream);
  });
  os << "\n";
  os << emitWithApplicationSymbols(
      model, [&](llvm::raw_ostream &stream) { emitDataFile(model, stream); });
  os << "\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(model)) {
    os << "extern uint32_t " << model.outputStem << "_build_tile_"
       << getPaddedPhysicalTileId(model, plan->hartid) << "_plan(\n";
    os << "    fifo_tile_plan_t *plan,\n";
    os << "    maps_operation_runtime_t *runtime);\n";
  }
  os << "\nvoid " << model.outputStem
     << "_initialize_data(uint32_t hartid) {\n";
  os << "  " << model.outputStem << "_init_tensors(hartid);\n";
  os << "}\n\n";
  os << "uint32_t " << model.outputStem << "_build_plan(\n";
  os << "    fifo_tile_plan_t *plan, uint32_t hartid,\n";
  os << "    maps_operation_runtime_t *runtime) {\n";
  os << "  switch (hartid) {\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(model)) {
    std::string tileId = getPaddedPhysicalTileId(model, plan->hartid);
    os << "  case " << plan->hartid << "u:\n";
    os << "    return " << model.outputStem << "_build_tile_" << tileId
       << "_plan(plan, runtime);\n";
  }
  os << "  default:\n";
  os << "    return 0u;\n";
  os << "  }\n";
  os << "}\n\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input &&
        global.globalKind != GeneratedGlobalKind::Output)
      continue;
    StringRef direction =
        global.globalKind == GeneratedGlobalKind::Input ? "input" : "output";
    if (global.globalKind == GeneratedGlobalKind::Output)
      os << "const ";
    os << "uint8_t *" << model.outputStem << "_" << direction << "_"
       << normalizeSnakeCase(global.globalName) << "(uint32_t token) {\n";
    os << "  return " << model.outputStem << "_l2_global_" << global.modelIndex
       << " + token * " << global.tokenStrideBytes << "u;\n";
    os << "}\n\n";
  }
}

static void emitApplicationTilePlanConstructor(
    const MagiaHeaderModel &model, const GeneratedTilePlan &plan,
    llvm::raw_ostream &os) {
  uint64_t fifoSlotDataSize = getFifoSlotDataSize(model, plan);
  os << "uint32_t maps_generated_build_tile_"
     << getPaddedPhysicalTileId(model, plan.hartid) << "_plan(\n";
  os << "    fifo_tile_plan_t *plan,\n";
  os << "    maps_operation_runtime_t *runtime) {\n";
  os << "  *plan = (fifo_tile_plan_t){\n";
  os << "    .hartid = " << plan.hartid << "u,\n";
  os << "    .l1_data_base = get_l1_base(" << plan.hartid << "u),\n";
  os << "    .ready_flags_base = MAPS_GENERATED_READY_FLAGS_OFFSET,\n";
  os << "    .ready_flags_count = MAPS_GENERATED_READY_FLAGS_WORDS,\n";
  os << "    .num_token_slots = MAPS_GENERATED_NUM_TOKEN_SLOTS,\n";
  os << "    .operation_runtime = runtime,\n";
  os << "    .fifo = {\n";
  os << "      .l1_offset_bytes = 0u,\n";
  os << "      .num_producers = " << plan.recvIndices.size() << "u,\n";
  os << "      .num_slots = MAPS_GENERATED_NUM_TOKEN_SLOTS,\n";
  os << "      .slot_data_size = " << fifoSlotDataSize << "u,\n";
  os << "    },\n";
  os << "  };\n";
  os << "  plan->l1_data_base += maps_generated_fifo_bytes(\n";
  os << "      plan->fifo.num_producers, plan->fifo.num_slots,\n";
  os << "      plan->fifo.slot_data_size);\n";
  emitPlanArrayAssignment("slices", "slices", plan.sliceIndices.size(),
                          plan.hartid, os);
  emitPlanArrayAssignment("init_l2_reads", "init_l2_reads",
                          plan.initL2ReadIndices.size(), plan.hartid, os);
  emitPlanArrayAssignment("l2_reads", "l2_reads",
                          plan.l2ReadIndices.size(), plan.hartid, os);
  emitPlanArrayAssignment("recvs", "recvs", plan.recvIndices.size(),
                          plan.hartid, os);
  emitPlanArrayAssignment("ops", "ops", plan.opIndices.size(), plan.hartid,
                          os);
  emitPlanArrayAssignment("sends", "sends", plan.sendIndices.size(),
                          plan.hartid, os);
  emitPlanArrayAssignment("l2_writes", "l2_writes",
                          plan.l2WriteIndices.size(), plan.hartid, os);
  os << "  return 1u;\n";
  os << "}\n";
}

void emitApplicationTileSource(const MagiaHeaderModel &model,
                               const GeneratedTilePlan &plan,
                               llvm::raw_ostream &os) {
  os << emitWithApplicationSymbols(model, [&](llvm::raw_ostream &stream) {
    emitGeneratedPrelude(model, stream);
    emitL2GlobalDeclarations(model, stream);
    emitTileSliceDescriptors(model, plan, stream);
    emitTileL2ReadDescriptors(model, plan, stream);
    emitTileRecvDescriptors(model, plan, stream);
    emitTileOpDescriptors(model, plan, stream);
    emitTileSendDescriptors(model, plan, stream);
    emitTileL2WriteDescriptors(model, plan, stream);
    emitFifoStorageHelper(stream);
    emitApplicationTilePlanConstructor(model, plan, stream);
  });
}

void emitApplicationRunner(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os) {
  os << "#include \"" << model.outputStem << ".h\"\n\n";
  os << "#include \"fsync.h\"\n";
  os << "#include \"redmule.h\"\n";
  os << "#include \"tile.h\"\n";
  os << "#include \"utils/maps_operations.h\"\n";
  os << "#include \"utils/maps_utils_v2.h\"\n\n";
  os << "#define " << uppercase(model.outputStem) << "_WAIT_MODE WFE\n\n";
  os << "extern void " << model.outputStem
     << "_initialize_data(uint32_t hartid);\n";
  os << "extern uint32_t " << model.outputStem << "_build_plan(\n";
  os << "    fifo_tile_plan_t *plan, uint32_t hartid,\n";
  os << "    maps_operation_runtime_t *runtime);\n\n";
  os << "int " << model.outputStem << "_run(void) {\n";
  os << "  const uint32_t hartid = get_hartid();\n";
  os << "  idma_config_t idma_config = {.hartid = hartid};\n";
  os << "  idma_controller_t idma = {\n";
  os << "      .base = 0, .cfg = &idma_config, .api = &idma_api};\n";
  os << "  redmule_config_t redmule_config = {.hartid = hartid};\n";
  os << "  redmule_controller_t redmule = {\n";
  os << "      .base = 0, .cfg = &redmule_config, .api = &redmule_api};\n";
  os << "  fsync_config_t fsync_config = {.hartid = hartid};\n";
  os << "  fsync_controller_t fsync = {\n";
  os << "      .base = 0, .cfg = &fsync_config, .api = &fsync_api};\n";
  os << "  eu_config_t eu_config = {.hartid = hartid};\n";
  os << "  eu_controller_t event_unit = {\n";
  os << "      .base = 0, .cfg = &eu_config, .api = &eu_api};\n\n";
  os << "  idma_init(&idma);\n";
  os << "  redmule_init(&redmule);\n";
  os << "  fsync_init(&fsync);\n";
  os << "  eu_init(&event_unit);\n";
  os << "  eu_clear_events(0xffffffffu);\n";
  os << "  eu_fsync_init(&event_unit, 0u);\n";
  os << "  eu_idma_init(&event_unit, 0u);\n";
  os << "  eu_redmule_init(&event_unit, 0u);\n\n";
  os << "  maps_operation_runtime_t runtime = {\n";
  os << "      .redmule_ctrl = &redmule, .eu_ctrl = &event_unit};\n";
  os << "  fifo_tile_plan_t plan;\n";
  os << "  const uint32_t active = " << model.outputStem
     << "_build_plan(&plan, hartid, &runtime);\n";
  os << "  if (active)\n";
  os << "    maps_fifo_init(&plan);\n\n";
  os << "  fsync_sync_level(&fsync, MAX_SYNC_LVL - 1, 0);\n";
  os << "  eu_fsync_wait(&event_unit, " << uppercase(model.outputStem)
     << "_WAIT_MODE);\n";
  os << "  " << model.outputStem << "_initialize_data(hartid);\n";
  os << "  if (hartid == 0u) {\n";
  os << "    for (uint32_t token = 0u; token < "
     << uppercase(model.outputStem) << "_EXECUTION_TOKENS; ++token)\n";
  os << "      " << model.outputStem << "_handle_input(token);\n";
  os << "  }\n";
  os << "  fsync_sync_global(&fsync);\n";
  os << "  eu_fsync_wait(&event_unit, " << uppercase(model.outputStem)
     << "_WAIT_MODE);\n";
  os << "  if (active)\n";
  os << "    maps_fifo_init_tile(&plan, &idma, &event_unit);\n";
  os << "  fsync_sync_global(&fsync);\n";
  os << "  eu_fsync_wait(&event_unit, " << uppercase(model.outputStem)
     << "_WAIT_MODE);\n";
  os << "  if (active)\n";
  os << "    maps_fifo_run_tile_tokens(\n";
  os << "        &plan, " << uppercase(model.outputStem)
     << "_EXECUTION_TOKENS, &idma, &event_unit);\n";
  os << "  fsync_sync_global(&fsync);\n";
  os << "  eu_fsync_wait(&event_unit, " << uppercase(model.outputStem)
     << "_WAIT_MODE);\n";
  os << "  if (hartid == 0u) {\n";
  os << "    for (uint32_t token = 0u; token < "
     << uppercase(model.outputStem) << "_EXECUTION_TOKENS; ++token)\n";
  os << "      " << model.outputStem << "_handle_output(token);\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) { return " << model.outputStem << "_run(); }\n";
}

void emitInitialApplicationSource(const MagiaHeaderModel &model,
                                  llvm::raw_ostream &os) {
  os << "/* Only customize this source file. MAPS owns every other emitted "
        "file. */\n";
  os << "#include \"" << model.outputStem << ".h\"\n\n";
  os << "#include <stddef.h>\n";
  os << "#include \"utils/printf.h\"\n\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.runtimeInputFile.empty())
      continue;
    os << "extern const uint8_t "
       << getRuntimeInputSymbol(model, global, "start") << "[];\n";
  }
  os << "\n";
  os << "void " << model.outputStem << "_handle_input(uint32_t token) {\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input)
      continue;
    std::string name = normalizeSnakeCase(global.globalName);
    os << "  uint8_t *" << name << " = " << model.outputStem << "_input_"
       << name << "(token);\n";
    if (!global.runtimeInputFile.empty()) {
      os << "  const uint8_t *" << name << "_data = "
         << getRuntimeInputSymbol(model, global, "start") << ";\n";
      os << "  for (size_t index = 0; index < " << global.tensorBytes
         << "u; ++index)\n";
      os << "    " << name << "[index] = " << name
         << "_data[index + token * " << global.tensorBytes << "u];\n";
    } else {
      os << "  for (size_t index = 0; index < " << global.tensorBytes
         << "u; ++index)\n";
      os << "    " << name << "[index] = (uint8_t)(index + token * 17u + "
         << global.globalId << "u * 31u);\n";
    }
  }
  os << "}\n\n";
  os << "void " << model.outputStem << "_handle_output(uint32_t token) {\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Output)
      continue;
    std::string name = normalizeSnakeCase(global.globalName);
    os << "  const uint8_t *" << name << " = " << model.outputStem << "_output_"
       << name << "(token);\n";
    os << "  uint32_t " << name << "_checksum = 2166136261u;\n";
    os << "  for (size_t index = 0; index < " << global.tensorBytes
       << "u; ++index)\n";
    os << "    " << name << "_checksum = (" << name << "_checksum ^ " << name
       << "[index]) * 16777619u;\n";
    os << "  printf(\"" << model.outputStem << " token %u output "
       << escapePrintfText(global.globalName)
       << " checksum: %08x\\n\", token, " << name
       << "_checksum);\n";
  }
  os << "}\n";
}

void emitApplicationCMake(const MagiaHeaderModel &model,
                          llvm::raw_ostream &os) {
  os << "set(MAPS_APPLICATION " << model.outputStem << ")\n\n";
  os << "add_executable(" << model.outputStem << "\n";
  os << "  src/application.c\n";
  os << "  src/" << model.outputStem << ".c\n";
  os << "  src/" << model.outputStem << "_runner.c\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(model))
    os << "  src/tiles/" << getApplicationTileFilename(model, plan->hartid)
       << "\n";
  os << ")\n";
  os << "configure_file(\n";
  os << "  src/" << model.outputStem << "_initializers.S.in\n";
  os << "  ${CMAKE_CURRENT_BINARY_DIR}/" << model.outputStem
     << "_initializers.S\n";
  os << "  @ONLY\n";
  os << ")\n";
  os << "target_sources(" << model.outputStem
     << " PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/" << model.outputStem
     << "_initializers.S)\n";
  os << "target_include_directories(" << model.outputStem
     << " PUBLIC include)\n";
  os << "target_compile_options(" << model.outputStem << " PRIVATE -O2)\n";
  os << "target_link_libraries(" << model.outputStem
     << " PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib "
        "kernels_redmule kernels_generic)\n";
}

void emitApplicationReadme(const MagiaHeaderModel &model,
                           llvm::raw_ostream &os) {
  os << "# " << model.outputStem << "\n\n";
  os << "Generated from `" << model.sourceModelName
     << "` for the `magia-v2` Target and a " << model.meshWidth << " x "
     << model.meshHeight << " planned Mesh.\n\n";
  os << "Copy this directory into the MAGIA SDK application/test tree. Add "
        "the following line to its parent `CMakeLists.txt`:\n\n";
  os << "```cmake\nadd_subdirectory(" << model.outputStem << ")\n```\n\n";
  os << "Configure the SDK normally, then build this application with:\n\n";
  os << "```sh\ncmake --build build --target " << model.outputStem
     << "\n```\n\n";
  os << "Only `src/application.c` is developer-owned. Edit that file to "
        "replace the generated input and output hooks; all other files are "
        "generated by MAPS.\n";
}

void emitApplicationInitializersAssembly(const MagiaHeaderModel &model,
                                         llvm::raw_ostream &os) {
  std::string start = model.outputStem + "_initializers_start";
  std::string end = model.outputStem + "_initializers_end";
  std::string size = model.outputStem + "_initializers_size";
  os << ".section .maps_initializers,\"a\",@progbits\n";
  os << ".balign " << model.weightsAlignment << "\n";
  os << ".global " << start << "\n";
  os << ".global " << end << "\n";
  os << ".global " << size << "\n";
  os << start << ":\n";
  os << ".incbin \"@CMAKE_CURRENT_SOURCE_DIR@/data/" << model.outputStem
     << ".initializers.bin\"\n";
  os << end << ":\n";
  os << ".set " << size << ", " << end << " - " << start << "\n";
  for (const GeneratedL2Global &global : model.l2Globals) {
    if (global.runtimeInputFile.empty())
      continue;
    start = getRuntimeInputSymbol(model, global, "start");
    end = getRuntimeInputSymbol(model, global, "end");
    size = getRuntimeInputSymbol(model, global, "size");
    os << "\n.section .maps_runtime_inputs,\"a\",@progbits\n";
    os << ".balign " << kGeneratedL2AlignmentBytes << "\n";
    os << ".global " << start << "\n";
    os << ".global " << end << "\n";
    os << ".global " << size << "\n";
    os << start << ":\n";
    os << ".incbin \"@CMAKE_CURRENT_SOURCE_DIR@/data/"
       << global.runtimeInputAsset << "\"\n";
    os << end << ":\n";
    os << ".set " << size << ", " << end << " - " << start << "\n";
  }
}

} // namespace mlir::maps
