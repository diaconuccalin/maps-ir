#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
plan_importer="${repo_root}/build/tools/maps-translate/maps-plan-import"
code_generator="${repo_root}/build/tools/maps-translate/maps-codegen"
fixture_root="${repo_root}/test/MagiaApplicationCodegen"
test_dir="$(mktemp -d)"
trap 'rm -rf "${test_dir}"' EXIT

cmake --build "${repo_root}/build" \
  --target maps-plan-import maps-codegen -j2 >/dev/null

execution_plan_fixture="${fixture_root}/execution_plan.json"
"${plan_importer}" "${execution_plan_fixture}" \
  -o "${test_dir}/execution_plan.mlir"
grep -q '@transition_0_destination_0' "${test_dir}/execution_plan.mlir"
grep -q '@transition_1_transfer_0' "${test_dir}/execution_plan.mlir"
grep -q '@transition_2_source_0' "${test_dir}/execution_plan.mlir"
grep -q '@initializer_stage_0_layer_0_input_1_destination_0' \
  "${test_dir}/execution_plan.mlir"

sed '/"execution"/d' "${fixture_root}/conv2d.json" \
  > "${test_dir}/missing_execution.json"
if "${plan_importer}" "${test_dir}/missing_execution.json" \
    -o /dev/null 2>/dev/null; then
  echo "expected missing Execution Plan contract failure" >&2
  exit 1
fi

granular_plan="${test_dir}/granular.plan.json"
PYTHONPATH="${repo_root}/.." "${repo_root}/../.venv/bin/python" \
  "${fixture_root}/write_granular_sharding_fixture.py" "${granular_plan}"
"${plan_importer}" "${granular_plan}" -o "${test_dir}/granular.mlir"
grep -Fq 'tensor<6xf16>' "${test_dir}/granular.mlir"
grep -Fq 'tensor<3xf16>' "${test_dir}/granular.mlir"

"${plan_importer}" "${fixture_root}/sigmoid.json" \
  -o "${test_dir}/sigmoid.mlir"
grep -q 'maps.kernel.*kind = "SIGMOID"' "${test_dir}/sigmoid.mlir"

source_initializers="${test_dir}/source.initializers.bin"
printf 'MAPS' > "${source_initializers}"
sed 's/maps.mesh_width = 1 : i64/maps.mesh_width = 2 : i64/' \
  "${test_dir}/sigmoid.mlir" > "${test_dir}/sigmoid_with_inactive_tile.mlir"
application="${test_dir}/my-model-v1"
"${code_generator}" "${test_dir}/sigmoid_with_inactive_tile.mlir" \
  --target=magia-v2 \
  --maps-magia-output-stem='My Model-v1' \
  --maps-magia-weights-file="${source_initializers}" \
  -o "${application}"

for path in \
  CMakeLists.txt README.md manifest.json \
  include/my_model_v1.h \
  src/my_model_v1.c src/my_model_v1_runner.c src/application.c \
  src/tiles/tile_00.c \
  src/my_model_v1_initializers.S.in \
  data/my_model_v1.initializers.bin; do
  test -f "${application}/${path}"
done
cmp "${source_initializers}" \
  "${application}/data/my_model_v1.initializers.bin"
! test -e "${application}/src/tiles/tile_01.c"
grep -q 'add_executable(my_model_v1' "${application}/CMakeLists.txt"
grep -q 'int my_model_v1_run(void);' \
  "${application}/include/my_model_v1.h"
grep -q 'void my_model_v1_handle_input(uint32_t token);' \
  "${application}/include/my_model_v1.h"
! grep -q 'slice_desc_t\|fifo_tile_plan_t' \
  "${application}/include/my_model_v1.h"
grep -q '.kind = OP_SIGMOID' "${application}/src/tiles/tile_00.c"
grep -q 'my_model_v1_build_tile_00_plan' \
  "${application}/src/tiles/tile_00.c"
grep -q 'my_model_v1_handle_input(token)' \
  "${application}/src/my_model_v1_runner.c"
grep -q 'my_model_v1_handle_output(token)' \
  "${application}/src/my_model_v1_runner.c"
grep -q 'checksum:' "${application}/src/application.c"
! grep -Eqi 'expected|correct(ness)?' "${application}/src/application.c"
retired_prefix="maps"_generated
! grep -Rq "${retired_prefix}" "${application}"
grep -q 'add_subdirectory(my_model_v1)' "${application}/README.md"
grep -q 'src/application.c' "${application}/README.md"

python3 - "${application}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

application = Path(sys.argv[1])
manifest = json.loads((application / "manifest.json").read_text())
assert manifest["application"] == {"name": "my_model_v1", "target": "magia-v2"}
assert manifest["planned_mesh"] == {"width": 2, "height": 1}
assert manifest["execution"] == {"tokens": 1, "token_slots": 2}
assert manifest["active_physical_tiles"] == [0]
assert manifest["tensors"]["inputs"][0]["original_name"] == "x"
assert manifest["tensors"]["outputs"][0]["original_name"] == "y"
assert manifest["files"]["user_owned"] == [{
    "path": "src/application.c",
    "role": "application_source",
}]
for record in manifest["files"]["generated"]:
    data = (application / record["path"]).read_bytes()
    assert record["byte_size"] == len(data)
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
PY

second_application="${test_dir}/second-my-model-v1"
"${code_generator}" "${test_dir}/sigmoid_with_inactive_tile.mlir" \
  --target=magia-v2 \
  --maps-magia-output-stem='My Model-v1' \
  --maps-magia-weights-file="${source_initializers}" \
  -o "${second_application}"
diff -r "${application}" "${second_application}"

transition_application="${test_dir}/transitions"
"${code_generator}" "${fixture_root}/two_tile_transitions.mlir" \
  --target=magia-v2 \
  --maps-magia-output-stem=transitions \
  --maps-magia-weights-file="${source_initializers}" \
  -o "${transition_application}"
grep -q 'static const fifo_send_desc_t' \
  "${transition_application}/src/tiles/tile_00.c"
grep -q '.dst_hartid = 1' \
  "${transition_application}/src/tiles/tile_00.c"
grep -q 'static const fifo_recv_desc_t' \
  "${transition_application}/src/tiles/tile_01.c"

collective_plan="${test_dir}/collective.plan.json"
PYTHONPATH="${repo_root}/.." "${repo_root}/../.venv/bin/python" -c \
  'from pathlib import Path; import sys; from examples.magia_collective_example import write_collective_bundle; write_collective_bundle(Path(sys.argv[1]))' \
  "${collective_plan}"
"${plan_importer}" "${collective_plan}" -o "${test_dir}/collective.mlir"
collective_application="${test_dir}/collective"
"${code_generator}" "${test_dir}/collective.mlir" \
  --target=magia-v2 \
  --maps-magia-output-stem=collective \
  --maps-magia-weights-file="${test_dir}/collective.plan.weights.bin" \
  -o "${collective_application}"
test "$(find "${collective_application}/src/tiles" -name 'tile_*.c' | wc -l)" -eq 4
grep -Rq '.num_participants = 2u' "${collective_application}/src/tiles"

if "${code_generator}" "${test_dir}/sigmoid.mlir" \
    --target=unsupported \
    --maps-magia-weights-file="${source_initializers}" \
    -o "${test_dir}/unsupported" 2>/dev/null; then
  echo "expected unsupported Target failure" >&2
  exit 1
fi
