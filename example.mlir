#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {maps.l2_capacity_bytes = 1048576 : i64, maps.mesh_height = 1 : i64, maps.mesh_width = 2 : i64, maps.name = "execution_plan", maps.num_token_slots = 2 : i64} {
  func.func @main(%arg0: tensor<2x2xf16> {maps.global_name = "x"}, %arg1: tensor<2x2xf16> {maps.global_name = "w"}) {
    %0 = tensor.empty() : tensor<2x2xf16>
    maps.init {
      %extracted_slice = tensor.extract_slice %arg1[0, 0] [2, 2] [1, 1] : tensor<2x2xf16> to tensor<2x2xf16>
      maps.send %extracted_slice, @initializer_stage_0_layer_0_input_1_destination_0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<2x2xf16>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %2 = maps.recv @initializer_stage_0_layer_0_input_1_destination_0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<2x2xf16>
        maps.store %2, @initializer_stage_0_layer_0_input_1_destination_0_slot : tensor<2x2xf16>
      }
    }
    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        %extracted_slice = tensor.extract_slice %arg0[0, 0] [2, 2] [1, 1] : tensor<2x2xf16> to tensor<2x2xf16>
        maps.send %extracted_slice, @transition_0_destination_0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<2x2xf16>
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %2 = maps.recv @transition_0_destination_0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<2x2xf16>
          %3 = maps.load @initializer_stage_0_layer_0_input_1_destination_0_slot : tensor<2x2xf16>
          %4 = tensor.empty() : tensor<2x2xf16>
          %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<2x2xf16>, tensor<2x2xf16>) outs(%4 : tensor<2x2xf16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %6 = arith.addf %in, %in_1 : f16
            linalg.yield %6 : f16
          } -> tensor<2x2xf16>
          %extracted_slice_0 = tensor.extract_slice %5[0, 0] [2, 2] [1, 1] : tensor<2x2xf16> to tensor<2x2xf16>
          maps.send %extracted_slice_0, @transition_1_transfer_0 {dst_hartid = 0 : i64, dst_parent_slice = array<i64: 0, 2, 0, 2>, dst_subslice = array<i64: 0, 2, 0, 2>, src_hartid = 0 : i64, src_parent_slice = array<i64: 0, 2, 0, 2>, src_subslice = array<i64: 0, 2, 0, 2>} : tensor<2x2xf16>
        }
      }
      maps.stage @stage1 attributes {stage_id = 1 : i64} {
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %2 = maps.recv @transition_1_transfer_0 {dst_hartid = 0 : i64, dst_parent_slice = array<i64: 0, 2, 0, 2>, dst_subslice = array<i64: 0, 2, 0, 2>, src_hartid = 0 : i64, src_parent_slice = array<i64: 0, 2, 0, 2>, src_subslice = array<i64: 0, 2, 0, 2>} : tensor<2x2xf16>
          %3 = "maps.kernel"(%2) <{kind = "SIGMOID", params = array<i64>}> {output_full_offset_0 = array<i64: 0, 0>} : (tensor<2x2xf16>) -> tensor<2x2xf16>
          %extracted_slice = tensor.extract_slice %3[0, 0] [2, 2] [1, 1] : tensor<2x2xf16> to tensor<2x2xf16>
          maps.send %extracted_slice, @transition_2_source_0 {dst_hartid = -1 : i64, maps.global_name = "z", src_hartid = 0 : i64} : tensor<2x2xf16>
        }
      }
    }
    %1 = maps.recv @transition_2_source_0 {dst_hartid = -1 : i64, src_hartid = 0 : i64} : tensor<2x2xf16>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0] [2, 2] [1, 1] : tensor<2x2xf16> into tensor<2x2xf16>
    return
  }
}
