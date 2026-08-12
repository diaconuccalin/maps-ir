module attributes {
  maps.bundle = {
    alignment = 16 : i64,
    endianness = "little",
    schema_version = 1 : i64,
    weights_file = "source.weights.bin",
    weights_sha256 = "3cadcc694b8014a8d342ffef5050d775a0385dea6ee03fb8753ca85404af2903",
    weights_size = 4 : i64
  },
  maps.l2_capacity_bytes = 1048576 : i64,
  maps.mesh_height = 1 : i64,
  maps.mesh_width = 2 : i64,
  maps.name = "two_tile_transitions",
  maps.num_token_slots = 2 : i64
} {
  func.func @main(%arg0: tensor<2x2xf16> {maps.global_name = "x"}) {
    %output = tensor.empty() : tensor<2x2xf16>
    maps.init {
    }
    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        %input_slice = tensor.extract_slice %arg0[0, 0] [2, 2] [1, 1]
          : tensor<2x2xf16> to tensor<2x2xf16>
        maps.send %input_slice, @transition_0_destination_0 {
          dst_hartid = 0 : i64,
          src_hartid = -1 : i64
        } : tensor<2x2xf16>
        maps.tile @tile0 attributes {
          coords = array<i64: 0, 0>,
          tile_id = 0 : i64
        } {
          %input = maps.recv @transition_0_destination_0 {
            dst_hartid = 0 : i64,
            src_hartid = -1 : i64
          } : tensor<2x2xf16>
          %produced = "maps.kernel"(%input) <{
            kind = "SIGMOID",
            params = array<i64>
          }> : (tensor<2x2xf16>) -> tensor<2x2xf16>
          %produced_slice = tensor.extract_slice %produced[0, 0] [2, 2] [1, 1]
            : tensor<2x2xf16> to tensor<2x2xf16>
          maps.send %produced_slice, @transition_1_transfer_0 {
            dst_hartid = 1 : i64,
            dst_parent_slice = array<i64: 0, 2, 0, 2>,
            dst_subslice = array<i64: 0, 2, 0, 2>,
            src_hartid = 0 : i64,
            src_parent_slice = array<i64: 0, 2, 0, 2>,
            src_subslice = array<i64: 0, 2, 0, 2>
          } : tensor<2x2xf16>
        }
      }
      maps.stage @stage1 attributes {stage_id = 1 : i64} {
        maps.tile @tile1 attributes {
          coords = array<i64: 1, 0>,
          tile_id = 1 : i64
        } {
          %received = maps.recv @transition_1_transfer_0 {
            dst_hartid = 1 : i64,
            dst_parent_slice = array<i64: 0, 2, 0, 2>,
            dst_subslice = array<i64: 0, 2, 0, 2>,
            src_hartid = 0 : i64,
            src_parent_slice = array<i64: 0, 2, 0, 2>,
            src_subslice = array<i64: 0, 2, 0, 2>
          } : tensor<2x2xf16>
          %consumed = "maps.kernel"(%received) <{
            kind = "EXP",
            params = array<i64>
          }> : (tensor<2x2xf16>) -> tensor<2x2xf16>
          %output_slice = tensor.extract_slice %consumed[0, 0] [2, 2] [1, 1]
            : tensor<2x2xf16> to tensor<2x2xf16>
          maps.send %output_slice, @transition_2_source_0 {
            dst_hartid = -1 : i64,
            src_hartid = 1 : i64
          } : tensor<2x2xf16>
        }
      }
    }
    %result = maps.recv @transition_2_source_0 {
      dst_hartid = -1 : i64,
      src_hartid = 1 : i64
    } : tensor<2x2xf16>
    %complete = tensor.insert_slice %result into %output[0, 0] [2, 2] [1, 1]
      : tensor<2x2xf16> into tensor<2x2xf16>
    return
  }
}
