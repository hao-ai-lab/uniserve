"""Defines a deterministic CPU model for exercising every worker execution route.

The model implements the same cache, forward, projection, vision, latent, and
diffusion boundaries as a neural worker_config while deriving outputs entirely from
request coordinates. Its fixed token cycle makes scheduler outcomes reproducible.
"""

from __future__ import annotations

import torch

from uniserve.attention.metadata import AttentionMode
from uniserve.model.batch import DiffusionBatch, EncodeBatch, TensorOutput, TextBatch, TextOutput
from uniserve.model.components import ComponentCall
from uniserve.model.decoder import DecoderMixin
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.encoder import EncodeKind, EncoderMixin
from uniserve.model.image_diffusion import BranchSource, ImageDiffusion, LatentLayout
from uniserve.model.model import Model
from uniserve.model.tensors import PositionLayout, TensorViews, TokenSelection, VocabularyPartition
from uniserve.model.text import TextMixin
from uniserve.nn.attention import CacheWrite
from uniserve.nn.diffusion.cfg import CfgRecipe
from uniserve.nn.diffusion.integrator import EulerSolver
from uniserve.nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain
from uniserve.nn.vae.patch import RgbDecoder
from uniserve.runtime.kv_cache import KVCacheConfig

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


class _Coordinates(torch.nn.Module):
    """Deterministic numerical token embeddings for the simulator."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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


class _StubText(torch.nn.Module):
    """Compose coordinate embeddings and the simulator's scalar cache writer."""

    dtype = torch.bfloat16
    max_tokens = STUB_MAX_LATENT_SIZE
    attention_mode = AttentionMode.PACKED

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = _STUB_HIDDEN_SIZE
        self.embed_tokens = _Coordinates()
        self.cache_write = CacheWrite()
        self.cache_config = KVCacheConfig(
            num_layers=STUB_NUM_LAYERS,
            total_layers=STUB_NUM_LAYERS,
            num_kv_heads=1,
            total_kv_heads=1,
            head_dim=1,
            dtype=torch.bfloat16,
            store_dtype=torch.bfloat16,
        )


class StubModel(TextMixin, EncoderMixin, DiffusionMixin, DecoderMixin, Model):
    """Implements every execution route with deterministic coordinate-derived output."""

    architectures = ("UniServeStubForUnifiedGeneration",)

    @property
    def vocabulary(self) -> VocabularyPartition:
        """Describe the complete unpadded deterministic vocabulary."""

        return VocabularyPartition(_STUB_VOCAB_SIZE, _STUB_VOCAB_SIZE, 0, (0,), None)

    @classmethod
    def component_calls(cls, config: object) -> tuple[ComponentCall, ...]:
        """Declare actual numerical methods and their mathematical participation."""

        return (
            ComponentCall("", "forward"),
            ComponentCall("", "forward_diffusion"),
            ComponentCall("", "encode:vision"),
            ComponentCall("", "encode:latent"),
            ComponentCall("", "decode:image"),
        )

    def __init__(self) -> None:
        """Declare fixed numerical feature, cache, and diffusion geometry."""

        super().__init__()
        self.architecture = "UniServeStubForUnifiedGeneration"

        self.text = _StubText()

        # Deterministic zero velocity keeps the diffusion route stable at every point.
        self.solver: EulerSolver = EulerSolver()
        self.generation = ImageDiffusion(
            latent_downsample=STUB_LATENT_DOWNSAMPLE,
            prediction_dtype=torch.bfloat16,
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

    @property
    def text_backbone(self):
        return self.text

    @property
    def text_pipeline(self) -> None:
        return None

    @torch.inference_mode()
    def forward(
        self, batch: TextBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> torch.Tensor:
        """Encode temporal coordinates while writing deterministic KV values."""

        query_tokens = sum(batch.attention.query_lens_cpu)
        kv = torch.zeros((query_tokens, 1, 1), dtype=torch.bfloat16, device=batch.positions.device)
        self.text.cache_write(batch.attention.out_cache_loc, kv, kv)
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
