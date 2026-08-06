"""Canonical immutable model declarations for execution-boundary tests."""

from uniserve_worker.batch import WorkVariant
from uniserve_worker.spec import (
    CacheSpec,
    DeploymentOverlay,
    FlowBranchSource,
    FlowConditioningKind,
    FlowSpec,
    InputSpec,
    LatentLayout,
    MaterializationKind,
    ModelSpec,
    NoiseScaleSpec,
    OperationSpec,
    OperationStageSpec,
    PositionLayout,
    ResourcePlan,
    RoutePlacement,
    RouteRowKind,
    RouteShape,
    RouteSpec,
    WeightSpec,
)

TEST_MODEL_SPEC = ModelSpec(
    architecture="TestExecutionModel",
    routes=(
        RouteSpec(
            name="test",
            row_kinds=(
                RouteRowKind.TOKEN,
                RouteRowKind.FLOW,
                RouteRowKind.ENCODE,
                RouteRowKind.DECODE,
            ),
            mixed_combinations=((RouteRowKind.TOKEN, RouteRowKind.FLOW),),
            dtype="float32",
            placement=RoutePlacement.PRIMARY,
            topology_axes=("tensor",),
            shape=RouteShape(max_tokens_per_row=4096, token_multiple=1),
            graph_eligible=False,
        ),
    ),
    operations=(
        OperationSpec(
            WorkVariant.TOKEN_EXTEND,
            (OperationStageSpec("test", RouteRowKind.TOKEN),),
        ),
        OperationSpec(
            WorkVariant.TOKEN_DECODE,
            (OperationStageSpec("test", RouteRowKind.TOKEN),),
        ),
        OperationSpec(
            WorkVariant.TOKEN_VERIFY,
            (OperationStageSpec("test", RouteRowKind.TOKEN),),
        ),
        OperationSpec(
            WorkVariant.GEN_TRANSITION,
            (OperationStageSpec("test", RouteRowKind.FLOW),),
        ),
        OperationSpec(
            WorkVariant.GEN_FLOW,
            (OperationStageSpec("test", RouteRowKind.FLOW),),
        ),
        OperationSpec(
            WorkVariant.MATERIALIZE,
            (OperationStageSpec("test", RouteRowKind.DECODE),),
        ),
        OperationSpec(
            WorkVariant.ENCODE_LATENT,
            (OperationStageSpec("test", RouteRowKind.ENCODE),),
        ),
        OperationSpec(
            WorkVariant.ENCODE_VISION,
            (OperationStageSpec("test", RouteRowKind.ENCODE),),
        ),
    ),
    weights=WeightSpec(),
    inputs=InputSpec(),
    cache=CacheSpec(
        num_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=4,
        dtype="float32",
    ),
    flow=FlowSpec(
        latent_downsample=16,
        prediction="velocity",
        prediction_dtype="float32",
        schedule_direction="ascending",
        schedule_shift_domain="time",
        max_latent_tokens=4096,
        max_vae_grid_tokens=4096,
        commit_marker_tokens=2,
        rope_advance=2,
        max_cfg_branches=3,
        latent_layout=LatentLayout.PATCH_TOKENS,
        latent_channels=4,
        latent_patch_size=1,
        positions=PositionLayout.TEMPORAL,
        conditioning=FlowConditioningKind.NONE,
        materialization=MaterializationKind.RGB_LATENT,
        noise_scale=NoiseScaleSpec(),
        text_unconditional=FlowBranchSource.START,
        image_unconditional=FlowBranchSource.START,
    ),
)

TEST_DEPLOYMENT = DeploymentOverlay(
    device="cpu",
    model_scope="whole",
    tp_rank=0,
    tp_size=1,
    block_size=64,
    kv_token_capacity=4096,
    generation_kv_capacity_tokens=None,
    attention_backend="auto",
    model_dtype="float32",
    kv_cache_dtype=None,
    kv_memory_fraction=1.0,
    resources=ResourcePlan(),
    max_batch_operations=1024,
    generation_device=None,
)
