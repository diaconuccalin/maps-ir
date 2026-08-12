"""Write the cross-language Shard Granularity acceptance fixture."""

from __future__ import annotations

from pathlib import Path
import sys

from maps.deployment.serialization import write_execution_plan
from maps.graph import Node, OpKind, Tensor, TensorDType
from maps.operations.elementwise import UnaryElementwisePayload
from maps.planning import (
    ExecutionContract,
    ExecutionPlan,
    Layer,
    LayerInput,
    LayerOutput,
    Stage,
)
from maps.planning.mapping import (
    LayoutAxis,
    LayoutAxisMode,
    Submesh,
    TensorLayout,
    tile_tensor_slice,
)
from maps.planning.transitions import (
    InputDestination,
    InputTransition,
    OutputSource,
    OutputTransition,
)
from maps.target.magia import build_mesh


def write_granular_sharding_fixture(
    output_path: Path,
    shard_granularity: int = 3,
) -> Path:
    """Serialize Python-owned Slices for runtime reconstruction."""

    mesh = build_mesh(width=2, height=1)
    submesh = Submesh(mesh, 0, frozenset((0, 1)))
    x = Tensor("x", 1, (9,), 2, dtype=TensorDType.FLOAT16)
    y = Tensor("y", 1, (9,), 2, dtype=TensorDType.FLOAT16)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(
            LayoutAxisMode.SHARD,
            tensor_axis=0,
            shard_granularity=shard_granularity,
        ),
        mesh_y=LayoutAxis(LayoutAxisMode.REPLICATE),
        logical_width=2,
        logical_height=1,
    )
    node = Node(
        "sigmoid",
        OpKind.ELEMENTWISE,
        inputs=(x,),
        outputs=(y,),
        payload=UnaryElementwisePayload("Sigmoid", x, y),
    )
    slices = tuple(tile_tensor_slice(y, layout, tile) for tile in submesh.tiles)
    plan = ExecutionPlan(
        "granular_sharding",
        mesh,
        tensors=(x, y),
        stages=(
            Stage(
                "sigmoid",
                submesh,
                layers=(
                    Layer(
                        node,
                        inputs=(LayerInput.transition_source(0, 0),),
                        outputs=(LayerOutput(tensor_id=1, layout=layout),),
                        device_name="spatz",
                    ),
                ),
                virtual_to_physical={0: 0, 1: 1},
            ),
        ),
        transitions=(
            InputTransition(
                tensor_id=0,
                destination_stage_id=0,
                destinations=tuple(
                    InputDestination(tile.tile_id, tensor_slice)
                    for tile, tensor_slice in zip(submesh.tiles, slices)
                ),
            ),
            OutputTransition(
                tensor_id=1,
                source_stage_id=0,
                sources=tuple(
                    OutputSource(tile.tile_id, tensor_slice)
                    for tile, tensor_slice in zip(submesh.tiles, slices)
                ),
            ),
        ),
        execution=ExecutionContract(num_token_slots=1),
    )
    return write_execution_plan(plan, output_path)


if __name__ == "__main__":
    granularity = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    write_granular_sharding_fixture(Path(sys.argv[1]), granularity)
