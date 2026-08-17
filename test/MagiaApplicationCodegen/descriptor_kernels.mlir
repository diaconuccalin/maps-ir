module attributes {
  maps.bundle = {
    alignment = 16 : i64,
    endianness = "little",
    schema_version = 1 : i64,
    weights_file = "descriptors.weights.bin",
    weights_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
    weights_size = 1440 : i64
  },
  maps.l2_capacity_bytes = 2048 : i64,
  maps.num_token_slots = 2 : i64,
  maps.name = "descriptors"
} {
  func.func @main(
      %input: tensor<1x2x3x4x5x6xf16> {
        maps.global_name = "input",
        maps.initializer = {
          byte_size = 1440 : i64,
          dtype = "float16",
          offset = 0 : i64,
          sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
          shape = [1, 2, 3, 4, 5, 6]
        }
      }) {
    maps.init {
      %slice = tensor.extract_slice %input[0, 0, 0, 0, 0, 0] [1, 2, 3, 4, 5, 6] [1, 1, 1, 1, 1, 1]
        : tensor<1x2x3x4x5x6xf16> to tensor<1x2x3x4x5x6xf16>
      maps.send %slice, @input_to_tile0 {
        dst_hartid = 0 : i64,
        src_hartid = -1 : i64
      } : tensor<1x2x3x4x5x6xf16>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %received = maps.recv @input_to_tile0 {
          dst_hartid = 0 : i64,
          src_hartid = -1 : i64
        } : tensor<1x2x3x4x5x6xf16>
        maps.store %received, @input_tile0 : tensor<1x2x3x4x5x6xf16>
      }
    }
    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %0 = maps.load @input_tile0 : tensor<1x2x3x4x5x6xf16>
          %1 = "maps.kernel"(%0) <{kind = "DEPTHWISE_CONV", params = array<i64: 1, 1, 1, 1, 0, 0, 0, 0>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %2 = "maps.kernel"(%1) <{kind = "ABS", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %3 = "maps.kernel"(%2, %2) <{kind = "DIV", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %4 = "maps.kernel"(%3) <{kind = "GROUP_NORMALIZE", params = array<i64: 2, 925353388, 360, 1>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %5 = "maps.kernel"(%4) <{kind = "GROUP_REDUCE", params = array<i64: 2, 360, 1>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %6 = "maps.kernel"(%5, %5) <{kind = "MUL", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %7 = "maps.kernel"(%6) <{kind = "NEG", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %8 = "maps.kernel"(%7, %7) <{kind = "POW", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %9 = "maps.kernel"(%8) <{kind = "REDUCE_MAX", params = array<i64: 1>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %10 = "maps.kernel"(%9) <{kind = "REDUCE_SUM", params = array<i64: 1>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %11 = "maps.kernel"(%10) <{kind = "RELU", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %12 = "maps.kernel"(%11) <{kind = "RESHAPE", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %13 = "maps.kernel"(%12) <{kind = "SIGMOID", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %14 = "maps.kernel"(%13) <{kind = "SLICE", params = array<i64: 0, 0>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %15 = "maps.kernel"(%14) <{kind = "SQRT", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %16 = "maps.kernel"(%15, %15) <{kind = "SUB", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %17 = "maps.kernel"(%16) <{kind = "TRANSPOSE", params = array<i64: 1, 0>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %18 = "maps.kernel"(%17) <{collective_group_id = 0 : i64, collective_layer_index = 18 : i64, collective_participants = array<i64: 0>, collective_stage_id = 0 : i64, kind = "AllReduceMax", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %19 = "maps.kernel"(%18) <{collective_group_id = 0 : i64, collective_layer_index = 19 : i64, collective_participants = array<i64: 0>, collective_stage_id = 0 : i64, kind = "AllReduceSum", params = array<i64>}> : (tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
          %20 = "maps.kernel"(%19, %19) <{kind = "MATMUL", params = array<i64: 2, 2, 2, 1>}> : (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x4x5x6xf16>
        }
      }
    }
    return
  }
}
