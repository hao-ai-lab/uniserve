"""Defines a deterministic CPU model for exercising every worker execution route.

The model implements the same cache, forward, projection, vision, latent, and
diffusion boundaries as a neural worker_config while deriving outputs entirely from
request coordinates. Its fixed token cycle makes scheduler outcomes reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve_worker.modeling.geometry import CacheGeometry, MediaShape, Shape
from uniserve_worker.modeling.tensors import AttentionMode, PositionLayout, TokenSelection

from ..modeling.batch import (
    DiffusionBatch,
    EncodeBatch,
    TensorOutput,
    TextBatch,
    TextOutput,
)
from ..modeling.components import Call, CallSpec, ComponentSpec
from ..modeling.decoder import DecoderMixin
from ..modeling.diffusion import DiffusionMixin
from ..modeling.encoder import EncodeKind, EncoderMixin
from ..modeling.image_diffusion import BranchSource, ImageDiffusion, LatentLayout
from ..modeling.inputs import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
    StrideResize,
    TowerTransform,
)
from ..modeling.model import Model
from ..modeling.resources import TensorNeeds, TensorSchema
from ..modeling.tensors import TensorViews, VocabularyPartition
from ..modeling.text import TextMixin
from ..nn.attention import CacheWrite
from ..nn.diffusion.cfg import CfgRecipe
from ..nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain
from ..nn.vae.patch import RgbDecoder

STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670
STUB_NUM_LAYERS = 1
STUB_MAX_LATENT_SIZE = 1024
STUB_LATENT_DOWNSAMPLE = 16
_STUB_VOCAB_SIZE = STUB_IMG_START_TOKEN_ID + 1
_STUB_HIDDEN_SIZE = 4

__all__ = [
    "STUB_EOS_TOKEN_ID",
    "STUB_IMG_START_TOKEN_ID",
    "StubModel",
]


@dataclass(frozen=True, slots=True)
class StubConfig:
    """Logical replicas of the deterministic numerical graph."""

    components: tuple[str, ...] = ("model",)


class StubModel(TextMixin, EncoderMixin, DiffusionMixin, DecoderMixin, Model):
    """Implements every execution route with deterministic coordinate-derived output."""

    architectures = ("UniServeStubForUnifiedGeneration",)
    dtype = torch.bfloat16

    @property
    def vocabulary(self) -> VocabularyPartition:
        """Describe the complete unpadded deterministic vocabulary."""

        return VocabularyPartition(self.vocab_size, self.vocab_size, 0, (0,), None)

    @classmethod
    def components(cls, config: object) -> tuple[ComponentSpec, ...]:
        """Declare the numerical calls sharing this model graph."""

        if not isinstance(config, StubConfig):
            raise TypeError("simulation components require numerical model configuration")
        return tuple(
            ComponentSpec(
                name,
                (
                    CallSpec(Call.TEXT),
                    CallSpec(Call.DIFFUSION),
                    CallSpec(Call.ENCODE_VISION),
                    CallSpec(Call.ENCODE_LATENT),
                    CallSpec(Call.DECODE_IMAGE),
                ),
            )
            for name in config.components
        )

    def __init__(self, config: StubConfig = StubConfig()) -> None:
        """Declare fixed numerical feature, cache, and diffusion geometry."""

        super().__init__(config)
        self.architecture = "UniServeStubForUnifiedGeneration"

        # Feature transforms match the shape contracts of vision and VAE routes.
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

        # A single scalar KV head is sufficient to exercise physical cache writes.
        self.cache_geometry = CacheGeometry(
            num_layers=STUB_NUM_LAYERS,
            num_attention_heads=1,
            num_kv_heads=1,
            total_kv_heads=1,
            kv_head_offset=0,
            head_dim=1,
            dtype="bfloat16",
            store_dtype="bfloat16",
        )

        # Deterministic zero velocity keeps the diffusion route stable at every point.
        self.generation = ImageDiffusion(
            latent_downsample=STUB_LATENT_DOWNSAMPLE,
            prediction="velocity",
            prediction_dtype="bfloat16",
            schedule_direction=ScheduleDirection.ASCENDING,
            schedule_shift_domain=ScheduleShiftDomain.TIME,
            max_latent_tokens=STUB_MAX_LATENT_SIZE,
            max_vae_grid_tokens=STUB_MAX_LATENT_SIZE,
            marker_tokens=2,
            rope_advance=2,
            max_cfg_branches=3,
            latent_layout=LatentLayout.IMAGE_NCHW,
            latent_channels=3,
            latent_patch_size=STUB_LATENT_DOWNSAMPLE,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            text_unconditional=BranchSource.START,
            image_unconditional=BranchSource.START,
            cfg_recipe=CfgRecipe.ADDITIVE_DELTAS,
        )
        self.image_decoder = RgbDecoder(STUB_LATENT_DOWNSAMPLE)
        self.max_vit_grid_tokens = STUB_MAX_LATENT_SIZE
        self.vocab_size = _STUB_VOCAB_SIZE
        self.hidden_size = _STUB_HIDDEN_SIZE
        self.text_max_tokens = STUB_MAX_LATENT_SIZE
        self.text_topology = ("tp",)
        self.text_attention_mode = AttentionMode.PACKED
        self.cache_write = CacheWrite()

    @property
    def text_pipeline(self) -> None:
        return None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed integer coordinates in the simulator's four-channel feature space."""

        coordinates = input_ids.reshape(-1).to(torch.bfloat16)
        return torch.stack(
            (
                coordinates,
                coordinates.remainder(17),
                coordinates.remainder(31),
                torch.ones_like(coordinates),
            ),
            dim=-1,
        )

    @torch.inference_mode()
    def forward(
        self, batch: TextBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> torch.Tensor:
        """Encode temporal coordinates while writing deterministic KV values."""

        query_tokens = sum(batch.attention.query_lens_cpu)
        kv = torch.zeros((query_tokens, 1, 1), dtype=torch.bfloat16, device=batch.positions.device)
        self.cache_write(batch.attention.out_cache_loc, kv, kv)
        temporal = batch.positions if batch.positions.ndim == 1 else batch.positions[0]
        hidden = self.embed_input_ids(temporal[:query_tokens])
        if batch.inputs_embeds is not None:
            assert batch.embedding_mask is not None
            hidden = torch.where(
                batch.embedding_mask[:query_tokens].reshape(-1, 1),
                batch.inputs_embeds[:query_tokens].to(hidden.dtype),
                hidden,
            )
        return hidden

    def compute_logits(self, hidden: torch.Tensor, batch: TextBatch) -> TextOutput:
        """Select deterministic successor logits or the supplied hidden rows."""

        lengths = batch.attention.query_lens_cpu
        rows = hidden[: sum(lengths)].split(lengths)
        token_rows = batch.input_ids[: sum(lengths)].split(lengths)
        logit_rows = sum(
            1 if selection is TokenSelection.LAST_LOGITS else count
            for selection, count in zip(batch.selections, lengths, strict=True)
            if selection is not TokenSelection.HIDDEN
        )
        storage = torch.full(
            (logit_rows, _STUB_VOCAB_SIZE), -16.0, dtype=torch.bfloat16, device=hidden.device
        )
        offset = 0
        outputs = []
        for row, ids, selection in zip(rows, token_rows, batch.selections, strict=True):
            if selection is TokenSelection.HIDDEN:
                outputs.append(row)
                continue
            targets = _next_tokens(ids)
            if selection is TokenSelection.LAST_LOGITS:
                targets = targets[-1:]
            count = targets.numel()
            logits = storage[offset : offset + count]
            offset += count
            logits.scatter_(1, targets.reshape(-1, 1), 16.0)
            outputs.append(logits)
        return TextOutput(tuple(outputs))

    def forward_diffusion(
        self,
        batch: DiffusionBatch,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Return zero velocity for each independent latent row."""

        return TensorOutput(
            {
                "image": tuple(
                    torch.zeros_like(latent, dtype=torch.bfloat16)
                    for latent in batch.latents["image"]
                )
            }
        )

    encoder_kinds: frozenset[EncodeKind] = frozenset({"vision", "latent"})

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Declare deterministic features and uncompressed BF16 image latents."""

        if call not in {Call.ENCODE_VISION, Call.ENCODE_LATENT}:
            return super().tensor_specs(call, shape)
        if not isinstance(shape, MediaShape) or shape.frames != 1:
            raise ValueError("simulation image encoding requires single-frame geometry")
        if call is Call.ENCODE_LATENT:
            return TensorNeeds(
                outputs={"latents": TensorSchema((1, 3, shape.height, shape.width), torch.bfloat16)}
            )
        # A supplied patch grid produces per-patch features; a whole numerical
        # image without a grid produces one pooled feature within this bound.
        rows = max(1, shape.height // 16 * (shape.width // 16))
        return TensorNeeds(
            outputs={
                "features": TensorSchema(
                    (rows, self.hidden_size), torch.bfloat16, variable_axes=(0,)
                )
            }
        )

    def encode(
        self, kind: EncodeKind, batch: EncodeBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> TensorOutput:
        """Compute deterministic vision features or BF16 image latents."""

        if kind == "latent":
            return TensorOutput(
                {
                    "latents": tuple(
                        value.to(torch.bfloat16).unsqueeze(0)
                        if value.ndim == 3
                        else value.to(torch.bfloat16)
                        for value in batch.values
                    )
                }
            )
        if kind != "vision":
            raise ValueError(f"unsupported encoder kind {kind!r}")
        outputs = []
        for value, grid in zip(batch.values, batch.grids, strict=True):
            typed = value.to(torch.bfloat16)
            if grid is not None:
                features = typed.mean(dim=-1, keepdim=True).repeat(1, _STUB_HIDDEN_SIZE)
            else:
                features = typed.mean().reshape(1, 1).repeat(1, _STUB_HIDDEN_SIZE)
            outputs.append(features)
        return TensorOutput({"features": tuple(outputs)})


def _next_token(token: int) -> int:
    """Return the scalar successor in the deterministic multimodal token cycle."""

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
    """Vectorize the deterministic multimodal token cycle over a tensor."""

    targets = torch.full_like(tokens, 1000)
    targets = torch.where(tokens == 1000, 1001, targets)
    targets = torch.where(tokens == 1001, STUB_IMG_START_TOKEN_ID, targets)
    targets = torch.where(tokens == STUB_IMG_START_TOKEN_ID, 1002, targets)
    targets = torch.where((tokens >= 1002) & (tokens < 1007), tokens + 1, targets)
    return torch.where(tokens == 1007, STUB_EOS_TOKEN_ID, targets)
