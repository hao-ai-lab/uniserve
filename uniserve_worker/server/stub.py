"""Deterministic raw-output model used by worker-process simulation."""

from __future__ import annotations

import torch

from ..batch import WorkVariant
from ..execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    TokenSelection,
)
from ..foundation.sizing import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BATCH_OPS,
    DEFAULT_MAX_REQUEST_POOL_SIZE,
)
from ..models.generation import BranchSource, GenerationPipeline, LatentLayout, Materialization
from ..models.inputs import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
    StrideResize,
    TowerTransform,
)
from ..models.runtime import (
    CacheGeometry,
    ExecutionModel,
    PositionLayout,
    ResourceGeometry,
    ScratchGeometry,
    WorkerDeployment,
)
from ..nn.diffusion.cfg import CfgRecipe
from ..nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain

STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670
STUB_NUM_BLOCKS = 4096
STUB_NUM_LAYERS = 1
STUB_SCRATCH_TOKENS = 65536
STUB_MAX_LATENT_SIZE = 1024
STUB_LATENT_DOWNSAMPLE = 16
_STUB_VOCAB_SIZE = STUB_IMG_START_TOKEN_ID + 1
_STUB_HIDDEN_SIZE = 4

__all__ = [
    "STUB_EOS_TOKEN_ID",
    "STUB_IMG_START_TOKEN_ID",
    "StubModel",
    "stub_deployment",
]


def stub_deployment(
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    max_batch_tokens: int,
) -> WorkerDeployment:
    return WorkerDeployment(
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
        max_batch_operations=DEFAULT_MAX_BATCH_OPS,
        max_batch_tokens=int(max_batch_tokens),
        max_request_pool_size=DEFAULT_MAX_REQUEST_POOL_SIZE,
        generation_device=None,
    )


class StubModel(ExecutionModel):
    """Stateless neural test double for the concrete imperative model boundary."""

    architectures = ("UniServeStubForUnifiedGeneration",)

    def __init__(self) -> None:
        super().__init__()
        self.architecture = "UniServeStubForUnifiedGeneration"
        image_resize = StrideResize(
            max_size=512,
            min_size=16,
            stride=16,
            max_pixels=512 * 512,
        )
        self.image_processor = ImageProcessor(
            vit=PatchTransform(
                patch_size=16,
                downsample_ratio=1.0,
                min_pixels=16 * 16,
                max_pixels=512 * 512,
                normalization="signed_unit",
            ),
            vae=TowerTransform(image_resize),
            staging_dtype="bfloat16",
            feature_injection=FeatureInjection(
                layout=FeatureLayout.DIRECT,
                positions=PositionLayout.TEMPORAL_SPATIAL,
                end_token_id=1007,
            ),
        )
        self.cache_geometry = CacheGeometry(
            num_layers=STUB_NUM_LAYERS,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=1,
            dtype="bfloat16",
            store_dtype="bfloat16",
        )
        self.generation = GenerationPipeline(
            latent_downsample=STUB_LATENT_DOWNSAMPLE,
            prediction="velocity",
            prediction_dtype="bfloat16",
            schedule_direction=ScheduleDirection.ASCENDING,
            schedule_shift_domain=ScheduleShiftDomain.TIME,
            max_latent_tokens=STUB_MAX_LATENT_SIZE,
            max_vae_grid_tokens=STUB_MAX_LATENT_SIZE,
            commit_marker_tokens=2,
            rope_advance=2,
            max_cfg_branches=3,
            latent_layout=LatentLayout.IMAGE_NCHW,
            latent_channels=3,
            latent_patch_size=STUB_LATENT_DOWNSAMPLE,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            materialization=Materialization.RGB_LATENT,
            text_unconditional=BranchSource.START,
            image_unconditional=BranchSource.START,
            cfg_recipe=CfgRecipe.ADDITIVE_DELTAS,
        )
        self.resource_geometry = ResourceGeometry(
            encoder_cache_entries=1024,
            latent_downsample=STUB_LATENT_DOWNSAMPLE,
            scratch=ScratchGeometry(fixed_tokens=STUB_SCRATCH_TOKENS),
        )
        self.max_vit_grid_tokens = STUB_MAX_LATENT_SIZE
        self.supported_work = frozenset(WorkVariant)
        self.vocab_size = _STUB_VOCAB_SIZE
        self.hidden_size = _STUB_HIDDEN_SIZE
        self.text_max_tokens = STUB_MAX_LATENT_SIZE
        self.text_topology = ("tp",)
        self.tensorized_mixed = True

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        query_tokens = sum(forward_batch.query_lens) + sum(forward_batch.flow_image_tokens)
        device = positions.device
        if query_tokens < 1:
            raise ValueError("stub text/denoise forward requires query tokens")
        kv = torch.zeros((query_tokens, 1, 1), dtype=torch.bfloat16, device=device)
        attention = forward_batch.attention
        if isinstance(attention, PagedDecodePlan):
            forward_batch.kv.append(0, kv.unsqueeze(1), kv.unsqueeze(1))
        elif isinstance(attention, PagedVarlenPlan):
            forward_batch.kv.append_varlen(
                0,
                kv,
                kv,
                attention.query_lens_cpu,
                block_table=attention.block_table,
                cache_seqlens=attention.cache_seqlens,
                query_offsets=attention.cu_seqlens_q,
            )
        elif isinstance(attention, PackedAttentionPlan):
            forward_batch.kv.append_packed(
                0,
                kv,
                kv,
                page_ids=attention.write_page_ids,
                page_offsets=attention.write_page_offsets,
                token_indices=attention.write_token_indices,
            )
        else:
            raise TypeError("stub text/denoise forward requires paged attention")

        chunks: list[torch.Tensor | None] = [None] * forward_batch.row_count
        token_offset = 0
        for row_index, count in zip(
            forward_batch.token_row_indices,
            forward_batch.query_lens,
            strict=True,
        ):
            row_positions = positions[..., token_offset : token_offset + count]
            temporal = row_positions.reshape(-1) if row_positions.ndim == 1 else row_positions[0]
            temporal = temporal.to(torch.bfloat16)
            chunks[row_index] = torch.stack(
                (
                    temporal,
                    temporal.remainder(17),
                    temporal.remainder(31),
                    torch.ones_like(temporal),
                ),
                dim=-1,
            )
            token_offset += count
        for flow_index, row_index in enumerate(forward_batch.flow_row_indices):
            count = int(forward_batch.flow_image_tokens[flow_index])
            chunks[row_index] = torch.zeros(
                (count, _STUB_HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            )
        if any(chunk is None for chunk in chunks):
            raise RuntimeError("stub forward batch contains an unbound row")
        return torch.cat(tuple(chunk for chunk in chunks if chunk is not None), dim=0)

    def project(self, hidden: torch.Tensor, forward_batch: ForwardBatch) -> ForwardOutput:
        row_lengths = [0] * forward_batch.row_count
        for row_index, count in zip(
            forward_batch.token_row_indices,
            forward_batch.query_lens,
            strict=True,
        ):
            row_lengths[row_index] = count
        for row_index, count in zip(
            forward_batch.flow_row_indices,
            forward_batch.flow_image_tokens,
            strict=True,
        ):
            row_lengths[row_index] = count
        rows: list[torch.Tensor] = []
        offset = 0
        for count in row_lengths:
            rows.append(hidden[offset : offset + count])
            offset += count

        token_ids: dict[int, torch.Tensor] = {}
        token_offset = 0
        if forward_batch.input_ids is not None:
            for row_index, count in zip(
                forward_batch.token_row_indices,
                forward_batch.query_lens,
                strict=True,
            ):
                token_ids[row_index] = forward_batch.input_ids[token_offset : token_offset + count]
                token_offset += count
        selections = dict(
            zip(
                forward_batch.token_row_indices,
                forward_batch.token_selections,
                strict=True,
            )
        )
        flow_rows = set(forward_batch.flow_row_indices)
        outputs: list[torch.Tensor] = []
        for row_index, row_hidden in enumerate(rows):
            selection = selections.get(row_index)
            if selection is TokenSelection.HIDDEN:
                outputs.append(row_hidden)
            elif selection is not None:
                ids = token_ids[row_index]
                targets = _next_tokens(ids)
                if selection is TokenSelection.LAST_LOGITS:
                    targets = targets[-1:]
                logits = torch.full(
                    (int(targets.numel()), _STUB_VOCAB_SIZE),
                    -16.0,
                    dtype=torch.bfloat16,
                    device=ids.device,
                )
                logits.scatter_(1, targets.reshape(-1, 1), 16.0)
                outputs.append(logits)
            elif row_index in flow_rows:
                flow_index = forward_batch.flow_row_indices.index(row_index)
                outputs.append(
                    torch.zeros_like(forward_batch.flow_latents[flow_index], dtype=torch.bfloat16)
                )
            else:
                raise RuntimeError("stub output row has no concrete phase")
        return ForwardOutput(tuple(outputs))

    def encode(
        self,
        pixels: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        outputs: list[torch.Tensor] = []
        for value, grid in zip(pixels, batch.encode_grids, strict=True):
            typed = value.to(torch.bfloat16)
            if grid is not None:
                features = typed.mean(dim=-1, keepdim=True).repeat(1, _STUB_HIDDEN_SIZE)
            else:
                features = typed.mean().reshape(1, 1).repeat(1, _STUB_HIDDEN_SIZE)
            outputs.append(features)
        return ForwardOutput(tuple(outputs))

    def encode_latent(
        self,
        pixels: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        del batch
        return ForwardOutput(
            tuple(
                value.to(torch.bfloat16).unsqueeze(0)
                if value.ndim == 3
                else value.to(torch.bfloat16)
                for value in pixels
            )
        )

    def decode_latent(
        self,
        latents: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        outputs = tuple(
            torch.zeros(
                (3, height, width),
                dtype=torch.bfloat16,
                device=latent.device,
            )
            for latent, height, width in zip(
                latents,
                batch.decode_heights,
                batch.decode_widths,
                strict=True,
            )
        )
        return ForwardOutput(outputs)


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


def _next_tokens(tokens: torch.Tensor) -> torch.Tensor:
    targets = torch.full_like(tokens, 1000)
    targets = torch.where(tokens == 1000, 1001, targets)
    targets = torch.where(tokens == 1001, STUB_IMG_START_TOKEN_ID, targets)
    targets = torch.where(tokens == STUB_IMG_START_TOKEN_ID, 1002, targets)
    targets = torch.where((tokens >= 1002) & (tokens < 1007), tokens + 1, targets)
    return torch.where(tokens == 1007, STUB_EOS_TOKEN_ID, targets)
