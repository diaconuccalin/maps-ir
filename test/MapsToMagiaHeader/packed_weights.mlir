module attributes {
  maps.bundle = {
    alignment = 16 : i64,
    endianness = "little",
    schema_version = 1 : i64,
    weights_file = "tiny.weights.bin",
    weights_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
    weights_size = 32 : i64
  },
  maps.l2_capacity_bytes = 64 : i64,
  maps.num_token_slots = 2 : i64,
  maps.name = "tiny"
} {
  func.func @main(
      %weight: tensor<2x2xf16> {
        maps.global_name = "weight",
        maps.initializer = {
          byte_size = 8 : i64,
          dtype = "float16",
          offset = 16 : i64,
          sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
          shape = [2, 2]
        }
      }) {
    maps.init {
      %slice = tensor.extract_slice %weight[0, 0] [2, 2] [1, 1]
        : tensor<2x2xf16> to tensor<2x2xf16>
      maps.send %slice, @weight_to_tile0 {
        dst_hartid = 0 : i64,
        src_hartid = -1 : i64
      } : tensor<2x2xf16>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %received = maps.recv @weight_to_tile0 {
          dst_hartid = 0 : i64,
          src_hartid = -1 : i64
        } : tensor<2x2xf16>
        maps.store %received, @weight_tile0 : tensor<2x2xf16>
      }
    }
    return
  }
}
