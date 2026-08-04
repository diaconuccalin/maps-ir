#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
translator="${repo_root}/build/tools/maps-translate/maps-translate"
fixture="${repo_root}/test/MapsToMagiaHeader/packed_weights.mlir"
conv2d_fixture="${repo_root}/test/MapsToMagiaHeader/conv2d.json"
descriptor_fixture="${repo_root}/test/MapsToMagiaHeader/descriptor_kernels.mlir"
execution_plan_fixture="${repo_root}/test/MapsToMagiaHeader/execution_plan.json"
granular_sharding_writer="${repo_root}/test/MapsToMagiaHeader/write_granular_sharding_fixture.py"
planner_execution_plan_fixture="${repo_root}/examples/magia_example.execution-plan.json"
sigmoid_fixture="${repo_root}/test/MapsToMagiaHeader/sigmoid.json"
test_dir="$(mktemp -d)"
trap 'rm -rf "${test_dir}"' EXIT
shard_mode=1
replicate_mode=3

expect_json_translation_failure() {
  local input_path="$1"
  local diagnostic="$2"
  if "${translator}" --json-to-maps "${input_path}" -o /dev/null 2>/dev/null; then
    echo "${diagnostic}" >&2
    exit 1
  fi
}

cmake --build "${repo_root}/build" --target maps-translate -j2 >/dev/null

collective_plan="${test_dir}/collective.execution-plan.json"
"${repo_root}/../.venv/bin/python" -c \
  'from pathlib import Path; import sys; from examples.magia_collective_example import write_collective_bundle; write_collective_bundle(Path(sys.argv[1]))' \
  "${collective_plan}"
"${translator}" --json-to-maps "${collective_plan}" \
  -o "${test_dir}/collective.mlir"
"${translator}" --maps-to-magia-header --maps-magia-output-stem=collective \
  "${test_dir}/collective.mlir" -o "${test_dir}/collective.h"
grep -q 'collective_participants = array<i64: 0, 1>' \
  "${test_dir}/collective.mlir"
grep -q 'collective_participants = array<i64: 2, 3>' \
  "${test_dir}/collective.mlir"
test "$(grep -c '.num_participants = 2u' "${test_dir}/collective.h")" -eq 8
grep -q '.group_id = 0u' "${test_dir}/collective.h"
grep -q '.group_id = 1u' "${test_dir}/collective.h"
! grep -q 'collective_axis' "${test_dir}/collective.mlir"
grep -q '#include "utils/maps_operations.h"' "${test_dir}/collective.h"
! grep -q 'maps_generated_execute_op' "${test_dir}/collective.h"
grep -q '.operation_runtime = runtime' "${test_dir}/collective.h"

mkdir "${test_dir}/second"
second_collective_plan="${test_dir}/second/collective.execution-plan.json"
"${repo_root}/../.venv/bin/python" -c \
  'from pathlib import Path; import sys; from examples.magia_collective_example import write_collective_bundle; write_collective_bundle(Path(sys.argv[1]))' \
  "${second_collective_plan}"
cmp "${collective_plan}" "${second_collective_plan}"
"${translator}" --json-to-maps "${second_collective_plan}" \
  -o "${test_dir}/second/collective.mlir"
"${translator}" --maps-to-magia-header --maps-magia-output-stem=collective \
  "${test_dir}/second/collective.mlir" \
  -o "${test_dir}/second/collective.h"
cmp "${test_dir}/collective.h" "${test_dir}/second/collective.h"

"${translator}" --maps-to-magia-header --maps-magia-output-stem=tiny \
  "${fixture}" -o "${test_dir}/tiny_generated.h"
"${translator}" --maps-to-magia-data --maps-magia-output-stem=tiny \
  "${fixture}" -o "${test_dir}/tiny_generated_data.c"
"${translator}" --maps-to-magia-weights-asm --maps-magia-output-stem=tiny \
  "${fixture}" -o "${test_dir}/tiny_weights.S"

grep -q 'tiny_weights_start + 16u' "${test_dir}/tiny_generated.h"
grep -q '.incbin "tiny.weights.bin"' "${test_dir}/tiny_weights.S"
grep -q '.balign 16' "${test_dir}/tiny_weights.S"
! grep -q 'maps_generated_lcg\|maps_generated_fill' "${test_dir}/tiny_generated_data.c"
! grep -q 'maps_generated_l2_global_0' "${test_dir}/tiny_generated_data.c"

cmp "${test_dir}/tiny_generated.h" <(
  "${translator}" --maps-to-magia-header --maps-magia-output-stem=tiny "${fixture}"
)

if "${translator}" --maps-to-magia-header --maps-magia-output-stem=tiny \
    --maps-magia-l2-end=3422617615 "${fixture}" -o /dev/null 2>/dev/null; then
  echo "expected L2-capacity failure" >&2
  exit 1
fi

"${translator}" --json-to-maps "${conv2d_fixture}" \
  -o "${test_dir}/conv2d.mlir"
sed 's/"elem_bytes": 2/"elem_bytes": 4/g' "${conv2d_fixture}" \
  > "${test_dir}/conv2d_f32.json"
"${translator}" --json-to-maps "${test_dir}/conv2d_f32.json" \
  -o "${test_dir}/conv2d_f32.mlir"
"${translator}" --maps-to-magia-header --maps-magia-output-stem=conv2d \
  --maps-magia-test-randomize-inputs \
  "${test_dir}/conv2d.mlir" -o "${test_dir}/conv2d_generated.h"

grep -q 'maps.conv2d' "${test_dir}/conv2d.mlir"
grep -q 'maps.num_token_slots = 3' "${test_dir}/conv2d.mlir"
grep -q 'maps.conv2d' "${test_dir}/conv2d_f32.mlir"
grep -q 'MAPS_GENERATED_NUM_TOKEN_SLOTS 3u' "${test_dir}/conv2d_generated.h"
grep -q 'MAPS_NUM_TOKEN_SLOTS disagrees with the planned execution contract' \
  "${test_dir}/conv2d_generated.h"
grep -q '.kind = OP_CONV2D' "${test_dir}/conv2d_generated.h"
! grep -q 'maps_generated_execute_op' "${test_dir}/conv2d_generated.h"
grep -q '.params = {2u, 2u, 1u, 2u, 1u, 1u, 0u, 0u}' \
  "${test_dir}/conv2d_generated.h"

sed '/"execution"/d' "${conv2d_fixture}" \
  > "${test_dir}/missing_execution.json"
if "${translator}" --json-to-maps "${test_dir}/missing_execution.json" \
    -o /dev/null 2>/dev/null; then
  echo "expected missing execution contract failure" >&2
  exit 1
fi

sed '/maps.num_token_slots/d' "${fixture}" \
  > "${test_dir}/missing_execution.mlir"
if "${translator}" --maps-to-magia-header \
    "${test_dir}/missing_execution.mlir" -o /dev/null 2>/dev/null; then
  echo "expected missing execution contract failure" >&2
  exit 1
fi

"${translator}" --maps-to-magia-header --maps-magia-output-stem=descriptors \
  "${descriptor_fixture}" -o "${test_dir}/descriptor_kernels.h"

for kind in \
  DEPTHWISE_CONV ABS DIV GROUP_NORMALIZE GROUP_REDUCE MUL NEG POW \
  REDUCE_MAX REDUCE_SUM RELU RESHAPE SIGMOID SLICE SQRT SUB TRANSPOSE \
  ALL_REDUCE_MAX ALL_REDUCE_SUM; do
  grep -q ".kind = OP_${kind}" "${test_dir}/descriptor_kernels.h"
done
grep -q '.params = {2u, 2u, 2u, 1u, 0u, 0u, 0u, 0u}' \
  "${test_dir}/descriptor_kernels.h"
grep -q '.shape = {1, 2, 3, 4, 5, 6}' \
  "${test_dir}/descriptor_kernels.h"
grep -q '.strides_bytes = {1440, 720, 240, 60, 12, 2}' \
  "${test_dir}/descriptor_kernels.h"
grep -q '.dims = {{0u, 1u, 1440u}, {0u, 2u, 720u}, {0u, 3u, 240u}, {0u, 4u, 60u}, {0u, 5u, 12u}, {0u, 6u, 2u}}' \
  "${test_dir}/descriptor_kernels.h"
grep -q '.collective = {.stage_id = 0u, .layer_index = 18u, .group_id = 0u' \
  "${test_dir}/descriptor_kernels.h"
grep -q '.collective = {.stage_id = 0u, .layer_index = 19u, .group_id = 0u' \
  "${test_dir}/descriptor_kernels.h"
test "$(grep -c '.num_participants = 1u' \
  "${test_dir}/descriptor_kernels.h")" -eq 2
grep -q '.participants = {{.hartid = 0u' \
  "${test_dir}/descriptor_kernels.h"

"${translator}" --json-to-maps "${sigmoid_fixture}" \
  -o "${test_dir}/sigmoid.mlir"
"${translator}" --maps-to-magia-header --maps-magia-output-stem=sigmoid \
  --maps-magia-test-randomize-inputs \
  "${test_dir}/sigmoid.mlir" -o "${test_dir}/sigmoid_generated.h"

grep -q 'maps.kernel.*kind = "SIGMOID"' "${test_dir}/sigmoid.mlir"
grep -q '.kind = OP_SIGMOID' "${test_dir}/sigmoid_generated.h"

granular_sharding_fixture="${test_dir}/granular_sharding.json"
"${repo_root}/../.venv/bin/python" "${granular_sharding_writer}" \
  "${granular_sharding_fixture}"
"${translator}" --json-to-maps "${granular_sharding_fixture}" \
  -o "${test_dir}/granular_sharding.mlir"
grep -Fq 'maps.recv @transition_0_destination_0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<6xf16>' \
  "${test_dir}/granular_sharding.mlir"
grep -Fq 'maps.recv @transition_0_destination_1 {dst_hartid = 1 : i64, src_hartid = -1 : i64} : tensor<3xf16>' \
  "${test_dir}/granular_sharding.mlir"
grep -Fq 'tensor.insert_slice %1 into %0[0] [6] [1] : tensor<6xf16> into tensor<9xf16>' \
  "${test_dir}/granular_sharding.mlir"
grep -Fq 'tensor.insert_slice %2 into %inserted_slice[6] [3] [1] : tensor<3xf16> into tensor<9xf16>' \
  "${test_dir}/granular_sharding.mlir"

legacy_sharding_with_defaults="${test_dir}/legacy_sharding_with_defaults.json"
"${repo_root}/../.venv/bin/python" "${granular_sharding_writer}" \
  "${legacy_sharding_with_defaults}" 1
sed '/"shard_granularity"/d' "${legacy_sharding_with_defaults}" \
  > "${test_dir}/legacy_sharding.json"
"${translator}" --json-to-maps "${test_dir}/legacy_sharding.json" \
  -o "${test_dir}/legacy_sharding.mlir"
grep -q 'tensor<5xf16>' "${test_dir}/legacy_sharding.mlir"
grep -q 'tensor<4xf16>' "${test_dir}/legacy_sharding.mlir"

sed 's/"shard_granularity": 3/"shard_granularity": 4/' \
  "${granular_sharding_fixture}" > "${test_dir}/non_divisible_sharding.json"
expect_json_translation_failure \
  "${test_dir}/non_divisible_sharding.json" \
  "expected non-divisible shard granularity failure"

sed "0,/\"mode\": ${shard_mode}/s//\"mode\": ${replicate_mode}/" \
  "${granular_sharding_fixture}" \
  > "${test_dir}/non_sharded_granularity.json"
expect_json_translation_failure \
  "${test_dir}/non_sharded_granularity.json" \
  "expected non-sharded granularity failure"

sed 's/"shard_granularity": 3/"shard_granularity": "three"/' \
  "${granular_sharding_fixture}" > "${test_dir}/non_integer_granularity.json"
expect_json_translation_failure \
  "${test_dir}/non_integer_granularity.json" \
  "expected non-integer shard granularity failure"

"${translator}" --json-to-maps "${execution_plan_fixture}" \
  -o "${test_dir}/execution_plan.mlir"
"${translator}" --maps-to-magia-header \
  --maps-magia-output-stem=execution_plan \
  --maps-magia-test-randomize-inputs \
  "${test_dir}/execution_plan.mlir" \
  -o "${test_dir}/execution_plan_generated.h"

grep -q '@transition_0_destination_0' "${test_dir}/execution_plan.mlir"
grep -q '@transition_1_transfer_0' "${test_dir}/execution_plan.mlir"
grep -q '@transition_2_source_0' "${test_dir}/execution_plan.mlir"
grep -q '@initializer_stage_0_layer_0_input_1_destination_0' \
  "${test_dir}/execution_plan.mlir"
test "$(grep -c 'maps.recv @transition_0_destination_0' \
  "${test_dir}/execution_plan.mlir")" -eq 1
test "$(grep -c 'maps.recv @transition_1_transfer_0' \
  "${test_dir}/execution_plan.mlir")" -eq 1
test "$(grep -c 'maps.send .*@transition_1_transfer_0' \
  "${test_dir}/execution_plan.mlir")" -eq 1
grep -q '.kind = OP_ADD' "${test_dir}/execution_plan_generated.h"
grep -q '.kind = OP_SIGMOID' "${test_dir}/execution_plan_generated.h"

"${translator}" --json-to-maps "${planner_execution_plan_fixture}" \
  -o "${test_dir}/planner_execution_plan.mlir"
grep -q '@transition_0_destination_0' "${test_dir}/planner_execution_plan.mlir"
grep -q '@transition_1_transfer_0' "${test_dir}/planner_execution_plan.mlir"
grep -q '@transition_3_source_0' "${test_dir}/planner_execution_plan.mlir"
grep -q '@initializer_stage_0_layer_0_input_1_destination_0' \
  "${test_dir}/planner_execution_plan.mlir"

package_dir="${test_dir}/package"
source_weights="${test_dir}/source.weights.bin"
printf 'MAPS' > "${source_weights}"
"${translator}" --json-to-magia-package \
  --maps-magia-package-dir="${package_dir}" \
  --maps-magia-output-stem=model \
  --maps-magia-weights-file="${source_weights}" \
  "${sigmoid_fixture}" -o /dev/null

test -f "${package_dir}/model.h"
test -f "${package_dir}/model_data.c"
test -f "${package_dir}/model_weights.S"
test -f "${package_dir}/model.weights.bin"
test -f "${package_dir}/manifest.json"
cmp "${source_weights}" "${package_dir}/model.weights.bin"
grep -q '.kind = OP_SIGMOID' "${package_dir}/model.h"
grep -q '.incbin "model.weights.bin"' "${package_dir}/model_weights.S"

python3 - "${package_dir}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

package = Path(sys.argv[1])
manifest = json.loads((package / "manifest.json").read_text())
assert manifest["schema_version"] == 1
assert manifest["source_model"] == {
    "name": "sigmoid",
    "inputs": [{
        "id": 0,
        "name": "x",
        "dtype": "float16",
        "shape": [2, 2],
        "encoding": "raw",
        "endianness": "little",
        "tensor_bytes": 8,
    }],
    "outputs": [{
        "id": 2,
        "name": "transition_1_source_0",
        "dtype": "float16",
        "shape": [2, 2],
        "encoding": "raw",
        "endianness": "little",
        "tensor_bytes": 8,
    }],
}
assert manifest["target"] == {
    "architecture": "magia-v2",
    "mesh": {"width": 1, "height": 1},
}
assert manifest["execution"] == {
    "num_token_slots": 2,
    "pipeline_token_capacity": 1,
}
assert manifest["abi"] == {
    "operation_version": 1,
    "descriptor_version": 1,
    "descriptor_max_dimensions": 6,
}
assert manifest["entry_points"] == {
    "init_l2_data": "maps_generated_init_l2_data",
    "init_tensors": "maps_generated_init_tensors",
    "fill_plan": "maps_generated_fill_plan",
    "check_output": "maps_generated_check_output",
}
assert manifest["memory"] == {
    "ready_flags_offset": 0xD0000,
    "max_tile_data_bytes": 32,
    "weights_size": 4,
    "required_l2_bytes": 48,
}
artifacts = manifest["artifacts"]
assert [item["path"] for item in artifacts] == [
    "model.h",
    "model_data.c",
    "model_weights.S",
    "model.weights.bin",
]
assert [item["role"] for item in artifacts] == [
    "header",
    "data_source",
    "weights_assembly",
    "weights_image",
]
for artifact in artifacts:
    path = Path(artifact["path"])
    assert not path.is_absolute() and ".." not in path.parts
    data = (package / path).read_bytes()
    assert artifact["byte_size"] == len(data)
    assert artifact["sha256"] == hashlib.sha256(data).hexdigest()
PY

second_package_dir="${test_dir}/second-package"
"${translator}" --json-to-magia-package \
  --maps-magia-package-dir="${second_package_dir}" \
  --maps-magia-output-stem=model \
  --maps-magia-weights-file="${source_weights}" \
  "${sigmoid_fixture}" -o /dev/null
for artifact in manifest.json model.h model_data.c model_weights.S model.weights.bin; do
  cmp "${package_dir}/${artifact}" "${second_package_dir}/${artifact}"
done

printf 'bad' > "${test_dir}/bad.weights.bin"
if "${translator}" --json-to-magia-package \
    --maps-magia-package-dir="${test_dir}/bad-package" \
    --maps-magia-weights-file="${test_dir}/bad.weights.bin" \
    "${sigmoid_fixture}" -o /dev/null 2>/dev/null; then
  echo "expected packed-weights size failure" >&2
  exit 1
fi
test ! -e "${test_dir}/bad-package"

printf 'FAIL' > "${test_dir}/wrong-checksum.weights.bin"
if "${translator}" --json-to-magia-package \
    --maps-magia-package-dir="${test_dir}/wrong-checksum-package" \
    --maps-magia-weights-file="${test_dir}/wrong-checksum.weights.bin" \
    "${sigmoid_fixture}" -o /dev/null 2>/dev/null; then
  echo "expected packed-weights checksum failure" >&2
  exit 1
fi
test ! -e "${test_dir}/wrong-checksum-package"

make -C "${repo_root}" magia-header \
  EXECUTION_PLAN_JSON="${sigmoid_fixture}" \
  EXECUTION_PLAN_MLIR="${test_dir}/sigmoid.mlir" \
  MAGIA_HEADER="${test_dir}/sigmoid_make_generated.h" \
  OUTPUT_STEM=sigmoid >/dev/null
grep -q '.kind = OP_SIGMOID' "${test_dir}/sigmoid_make_generated.h"

if make -n -C "${repo_root}/.." magia-header | grep -q 'examples/magia_example.py'; then
  echo "standalone magia-header must not replan Execution Plan JSON" >&2
  exit 1
fi

stress_fixture="${test_dir}/produced_value_stress.mlir"
{
  cat <<'EOF'
module attributes {maps.l2_capacity_bytes = 1048576 : i64, maps.name = "produced_value_stress", maps.num_token_slots = 2 : i64} {
  func.func @main(%arg0: tensor<1xf16> {maps.global_name = "input"}) {
    %empty = tensor.empty() : tensor<1xf16>
    maps.init {
    }
    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        %input_slice = tensor.extract_slice %arg0[0] [1] [1] : tensor<1xf16> to tensor<1xf16>
        maps.send %input_slice, @input_from_L2_to_tile0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<1xf16>
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %received = maps.recv @input_from_L2_to_tile0 {dst_hartid = 0 : i64, src_hartid = -1 : i64} : tensor<1xf16>
EOF
  previous="%received"
  for index in $(seq 0 9999); do
    printf '          %%value%s = "maps.kernel"(%s) <{kind = "SIGMOID", params = array<i64>}> : (tensor<1xf16>) -> tensor<1xf16>\n' \
      "${index}" "${previous}"
    previous="%value${index}"
  done
  cat <<EOF
          %output_slice = tensor.extract_slice ${previous}[0] [1] [1] : tensor<1xf16> to tensor<1xf16>
          maps.send %output_slice, @output_from_tile0_to_L2 {dst_hartid = -1 : i64, src_hartid = 0 : i64} : tensor<1xf16>
        }
      }
    }
    %output = maps.recv @output_from_tile0_to_L2 {dst_hartid = -1 : i64, src_hartid = 0 : i64} : tensor<1xf16>
    %result = tensor.insert_slice %output into %empty[0] [1] [1] : tensor<1xf16> into tensor<1xf16>
    return
  }
}
EOF
} > "${stress_fixture}"

timeout 10s "${translator}" --maps-to-magia-header \
  --maps-magia-test-randomize-inputs \
  --maps-magia-output-stem=produced_value_stress \
  "${stress_fixture}" -o "${test_dir}/produced_value_stress.h"
stress_op_count="$(grep -c '.kind = OP_SIGMOID' "${test_dir}/produced_value_stress.h")"
if [[ "${stress_op_count}" -ne 10000 ]]; then
  echo "unexpected stress operation count: ${stress_op_count}" >&2
  exit 1
fi
