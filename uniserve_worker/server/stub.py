"""Deterministic raw-output model used by worker-process simulation."""

from __future__ import annotations

import torch
from torch import nn

from ..batch import WorkVariant
from ..forward import (
    EncodeKind,
    EncodeOutput,
    EncodeRow,
    FlowOutput,
    FlowRow,
    ForwardBatch,
    ForwardOutput,
    PackedAttentionPlan,
    PagedDecodePlan,
    PatchInput,
    TokenEmbeddings,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSegments,
    TokenSelection,
    TowerInput,
)
from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
from ..spec import (
    CacheSpec,
    DeploymentOverlay,
    EncoderResourcePolicy,
    FlowBranchSource,
    FlowConditioningKind,
    FlowSpec,
    ImageInputSpec,
    ImagePatchSpec,
    ImageTowerSpec,
    InputSpec,
    KvBlockResourcePolicy,
    LatentLayout,
    LatentTokens,
    MaterializationKind,
    ModelSpec,
    NoiseScaleSpec,
    OperationSpec,
    OperationStageSpec,
    PerBranch,
    PositionLayout,
    ResourcePlan,
    RoutePlacement,
    RouteRowKind,
    RouteShape,
    RouteShapeGrouping,
    RouteSpec,
    StrideResizeSpec,
    WeightSpec,
)

STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670
STUB_NUM_BLOCKS = 4096
STUB_NUM_LAYERS = 1
STUB_SCRATCH_TOKENS = 65536
STUB_MAX_LATENT_SIZE = 1024
STUB_LATENT_DOWNSAMPLE = 16
STUB_BYTES_PER_TOKEN = 4
_STUB_VOCAB_SIZE = STUB_IMG_START_TOKEN_ID + 1
_STUB_HIDDEN_SIZE = 4

__all__ = [
    "STUB_EOS_TOKEN_ID",
    "STUB_IMG_START_TOKEN_ID",
    "StubModel",
    "stub_deployment",
]


def stub_deployment(block_size: int = DEFAULT_BLOCK_SIZE) -> DeploymentOverlay:
    return DeploymentOverlay(
        device="cpu",
        model_scope="whole",
        tp_rank=0,
        tp_size=1,
        block_size=int(block_size),
        kv_token_capacity=int(block_size) * STUB_NUM_BLOCKS,
        generation_kv_capacity_tokens=None,
        attention_backend="torch_sdpa",
        model_dtype="bfloat16",
        kv_cache_dtype=None,
        kv_memory_fraction=1.0,
        resources=ResourcePlan(
            kv_block=KvBlockResourcePolicy.PER_BLOCK,
            encoder_output=EncoderResourcePolicy.PER_HANDLE,
            image_latent=LatentTokens(downsample=STUB_LATENT_DOWNSAMPLE),
            scratch=PerBranch(fixed_tokens=STUB_SCRATCH_TOKENS),
        ),
        max_batch_operations=DEFAULT_MAX_BATCH_OPS,
        generation_device=None,
    )


class StubModel(nn.Module):
    """Stateless neural test double that obeys the canonical model boundary."""

    architectures = ("UniServeStubForUnifiedGeneration",)

    def __init__(self) -> None:
        super().__init__()
        self.spec = self._build_spec()

    @staticmethod
    def _build_spec() -> ModelSpec:
        image_resize = StrideResizeSpec(
            max_size=512,
            min_size=16,
            stride=16,
            max_pixels=512 * 512,
        )
        return ModelSpec(
            architecture="UniServeStubForUnifiedGeneration",
            revision="stub-v3",
            routes=(
                RouteSpec(
                    name="stub",
                    row_kinds=(RouteRowKind.TOKEN, RouteRowKind.FLOW),
                    mixed_combinations=((RouteRowKind.TOKEN, RouteRowKind.FLOW),),
                    dtype="bfloat16",
                    placement=RoutePlacement.PRIMARY,
                    topology_axes=("tp",),
                    shape=RouteShape(max_tokens_per_row=STUB_MAX_LATENT_SIZE, token_multiple=1),
                    graph_eligible=False,
                ),
                RouteSpec(
                    name="encode",
                    row_kinds=(RouteRowKind.ENCODE,),
                    mixed_combinations=(),
                    dtype="bfloat16",
                    placement=RoutePlacement.PRIMARY,
                    topology_axes=("tp",),
                    shape=RouteShape(
                        max_tokens_per_row=STUB_MAX_LATENT_SIZE,
                        token_multiple=1,
                        grouping=RouteShapeGrouping.INPUT_SHAPE,
                    ),
                    graph_eligible=False,
                ),
            ),
            operations=(
                OperationSpec(
                    WorkVariant.TOKEN_EXTEND,
                    (OperationStageSpec("stub", RouteRowKind.TOKEN),),
                ),
                OperationSpec(
                    WorkVariant.TOKEN_DECODE,
                    (OperationStageSpec("stub", RouteRowKind.TOKEN),),
                ),
                OperationSpec(
                    WorkVariant.TOKEN_VERIFY,
                    (OperationStageSpec("stub", RouteRowKind.TOKEN),),
                ),
                OperationSpec(
                    WorkVariant.GEN_TRANSITION,
                    (OperationStageSpec("stub", RouteRowKind.FLOW),),
                ),
                OperationSpec(
                    WorkVariant.GEN_FLOW,
                    (OperationStageSpec("stub", RouteRowKind.FLOW),),
                ),
                OperationSpec(
                    WorkVariant.ENCODE_VISION,
                    (OperationStageSpec("encode", RouteRowKind.ENCODE),),
                ),
                OperationSpec(
                    WorkVariant.ENCODE_LATENT,
                    (OperationStageSpec("encode", RouteRowKind.ENCODE),),
                ),
                OperationSpec(WorkVariant.MATERIALIZE),
                OperationSpec(WorkVariant.TRANSFER_PRODUCT),
                OperationSpec(WorkVariant.TRANSFER_KV_PUBLISH),
                OperationSpec(WorkVariant.TRANSFER_KV_INSTALL),
            ),
            weights=WeightSpec(),
            inputs=InputSpec(
                images=ImageInputSpec(
                    vit=ImagePatchSpec(
                        patch_size=16,
                        downsample_ratio=1.0,
                        min_pixels=16 * 16,
                        max_pixels=512 * 512,
                        multi_image_pixel_budget=512 * 512,
                        normalization="signed_unit",
                    ),
                    vae=ImageTowerSpec(image_resize, normalization="signed_unit"),
                    staging_dtype="bfloat16",
                ),
                encoder_cache_budget=1024,
                max_vit_grid_tokens=STUB_MAX_LATENT_SIZE,
            ),
            cache=CacheSpec(
                num_layers=STUB_NUM_LAYERS,
                num_attention_heads=1,
                num_kv_heads=1,
                head_dim=1,
                dtype="bfloat16",
                store_dtype="bfloat16",
                position_layout=PositionLayout.TEMPORAL_SPATIAL,
            ),
            flow=FlowSpec(
                latent_downsample=STUB_LATENT_DOWNSAMPLE,
                prediction="velocity",
                prediction_dtype="bfloat16",
                schedule_direction="ascending",
                schedule_shift_domain="time",
                max_latent_tokens=STUB_MAX_LATENT_SIZE,
                max_vae_grid_tokens=STUB_MAX_LATENT_SIZE,
                commit_marker_tokens=2,
                rope_advance=2,
                max_cfg_branches=3,
                latent_layout=LatentLayout.IMAGE_NCHW,
                latent_channels=3,
                latent_patch_size=STUB_LATENT_DOWNSAMPLE,
                positions=PositionLayout.TEMPORAL_SPATIAL,
                conditioning=FlowConditioningKind.IMAGE_PATCHES,
                materialization=MaterializationKind.RGB_LATENT,
                noise_scale=NoiseScaleSpec(),
                text_unconditional=FlowBranchSource.START,
                image_unconditional=FlowBranchSource.START,
            ),
        )

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        if batch.route == "stub":
            return self._sequence_and_flow(batch)
        if batch.route == "encode":
            return self._encode(batch)
        raise ValueError(f"stub model received unknown route {batch.route!s}")

    def _sequence_and_flow(self, batch: ForwardBatch) -> ForwardOutput:
        rows: list[TokenRow | FlowRow] = []
        for row in batch.rows:
            if not isinstance(row, (TokenRow, FlowRow)):
                raise TypeError("stub route accepts token and flow rows")
            rows.append(row)
        query_tokens = sum(
            _row_token_count(row) if isinstance(row, TokenRow) else int(row.image_tokens)
            for row in rows
        )
        zeros = torch.zeros(
            (query_tokens, 1, 1),
            dtype=torch.bfloat16,
            device=_row_device(rows[0]),
        )
        plan = batch.context.attention
        if isinstance(plan, PagedDecodePlan):
            if any(not isinstance(row, TokenRow) or _row_token_count(row) != 1 for row in rows):
                raise TypeError("stub paged decode requires one token per row")
            batch.context.kv.append(0, zeros.unsqueeze(1), zeros.unsqueeze(1))
        elif isinstance(plan, PackedAttentionPlan):
            batch.context.kv.append_packed(
                0,
                zeros,
                zeros,
                page_ids=plan.write_page_ids,
                page_offsets=plan.write_page_offsets,
                token_indices=plan.write_token_indices,
            )
        else:
            raise TypeError("stub sequence/flow route requires a paged attention plan")
        outputs: list[TokenOutput | FlowOutput] = []
        for row in rows:
            if isinstance(row, TokenRow):
                outputs.append(self._token(row))
            else:
                outputs.append(
                    FlowOutput(
                        row_id=row.row_id,
                        output_slot=row.output_slot,
                        prediction=torch.zeros_like(row.latent, dtype=torch.bfloat16),
                    )
                )
        return ForwardOutput(tuple(outputs))

    def _token(self, row: TokenRow) -> TokenOutput:
        value: TokenHidden | TokenLogits
        if row.selection is TokenSelection.HIDDEN:
            value = TokenHidden(self._hidden(row))
        else:
            targets = self._targets(row)
            if row.selection is TokenSelection.LAST_LOGITS:
                targets = targets[-1:]
            logits = torch.full(
                (len(targets), _STUB_VOCAB_SIZE),
                -16.0,
                dtype=torch.bfloat16,
                device=row.positions.device,
            )
            for index, target in enumerate(targets):
                logits[index, target] = 16.0
            value = TokenLogits(logits)
        return TokenOutput(row.row_id, row.output_slot, value)

    @staticmethod
    def _hidden(row: TokenRow) -> torch.Tensor:
        positions = (
            row.positions.reshape(-1) if row.positions.ndim == 1 else row.positions[0].reshape(-1)
        ).to(torch.bfloat16)
        return torch.stack(
            (
                positions,
                positions.remainder(17),
                positions.remainder(31),
                torch.ones_like(positions),
            ),
            dim=-1,
        )

    @staticmethod
    def _targets(row: TokenRow) -> tuple[int, ...]:
        ids = _token_ids(row)
        return tuple(_next_token(token) for token in ids)

    @staticmethod
    def _encode(batch: ForwardBatch) -> ForwardOutput:
        rows: list[EncodeRow] = []
        for row in batch.rows:
            if not isinstance(row, EncodeRow):
                raise TypeError("stub encode route accepts encode rows")
            rows.append(row)
        outputs: list[EncodeOutput] = []
        for row in rows:
            if isinstance(row.inputs, PatchInput):
                pixels = row.inputs.pixels.to(torch.bfloat16)
                mean = pixels.mean(dim=-1, keepdim=True)
                features = mean.repeat(1, _STUB_HIDDEN_SIZE)
            elif isinstance(row.inputs, TowerInput):
                pixels = row.inputs.pixels.to(torch.bfloat16)
                if row.kind is EncodeKind.LATENT:
                    features = pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
                else:
                    features = pixels.mean().reshape(1, 1).repeat(1, _STUB_HIDDEN_SIZE)
            else:
                raise TypeError("stub encode row has an unknown input variant")
            outputs.append(EncodeOutput(row.row_id, row.output_slot, features))
        return ForwardOutput(tuple(outputs))


def _row_device(row: TokenRow | FlowRow) -> torch.device:
    return row.positions.device


def _token_ids(row: TokenRow) -> tuple[int, ...]:
    if isinstance(row.inputs, TokenIds):
        return tuple(int(value) for value in row.inputs.values.reshape(-1).tolist())
    if isinstance(row.inputs, TokenEmbeddings):
        return (1001,) * _row_token_count(row)
    if isinstance(row.inputs, TokenSegments):
        values: list[int] = []
        for segment in row.inputs.values:
            if isinstance(segment, TokenIds):
                values.extend(int(value) for value in segment.values.reshape(-1).tolist())
            else:
                values.extend([1001] * int(segment.values.shape[0]))
        return tuple(values)
    raise TypeError("stub token row has an unknown input variant")


def _next_token(token: int) -> int:
    if token == 1000:
        return 1001
    if token == 1001:
        return STUB_IMG_START_TOKEN_ID
    if token == STUB_IMG_START_TOKEN_ID:
        return 1002
    if 1002 <= token < 1007:
        return token + 1
    if token == 1007:
        return STUB_EOS_TOKEN_ID
    return 1000


def _row_token_count(row: TokenRow) -> int:
    inputs = row.inputs
    if isinstance(inputs, (TokenIds, TokenEmbeddings)):
        return int(inputs.values.shape[0])
    return sum(int(segment.values.shape[0]) for segment in inputs.values)
