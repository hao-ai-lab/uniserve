"""BAGEL neural model definition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..execution.batch import ForwardMode
from ..execution.forward_batch import (
    AttentionMode,
    ForwardBatch,
    ForwardOutput,
    TokenSelection,
)
from ..loader.handles import WeightHandle
from ..loader.mapping import LoadReport, WeightNameMap, stacked_weight_name
from ..loader.weight_loaders import load_parameter_weight
from ..nn import (
    LayerConfig,
    LinearBase,
    MLPConnector,
    ParallelLMHead,
    local_attention_head_count,
    local_kv_head_count,
)
from ..nn.decoder import MoTConfig, MoTModel
from ..nn.diffusion import (
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from ..nn.diffusion.cfg import CfgRecipe
from ..nn.vae import AutoEncoder, default_ae_params
from ..nn.vision import (
    PositionEmbedding,
    SiglipNavitConfig,
    SiglipNavitEncoder,
    get_flattened_position_ids_extrapolate,
    patchify_batch,
)
from .generation import (
    BranchSource,
    GenerationPipeline,
    LatentLayout,
    Materialization,
)
from .inputs import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    StrideResize,
    TowerTransform,
)
from .runtime import (
    CacheGeometry,
    ExecutionModel,
    PositionLayout,
    ResourceGeometry,
)

__all__ = [
    "LLMConfig",
    "BagelConfig",
    "BagelForConditionalGeneration",
]

_BAGEL_RMS_NORM_EPS = 1e-6
_BAGEL_ROPE_THETA = 1_000_000.0
_BAGEL_VIT_LAYER_NORM_EPS = 1e-6
_BAGEL_IMAGE_MARKER_TOKENS = 2
_BAGEL_VIT_MIN_SIZE = 224
_BAGEL_VAE_MIN_SIZE = 512
_BAGEL_VAE_MAX_SIZE = 1024
_BAGEL_VAE_STRIDE = 16
_BAGEL_MAX_IMAGE_PIXELS = 14 * 14 * 9 * 1024


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Language-model hyperparameters for the BAGEL stack."""

    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    rms_norm_eps: float = _BAGEL_RMS_NORM_EPS
    rope_theta: float = _BAGEL_ROPE_THETA
    qk_norm: bool = True
    max_position_embeddings: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True, slots=True)
class BagelConfig:
    """Top-level BAGEL model configuration (LLM, ViT, VAE, and latent settings)."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    start_of_image_id: int = 151652
    end_of_image_id: int = 151653
    vae_z_channels: int = 16
    vae_downsample: int = 8
    latent_patch_size: int = 2
    max_latent_size: int = 32
    timestep_shift: float = 1.0
    vit_hidden_size: int = 1152
    vit_intermediate_size: int = 4304
    vit_num_hidden_layers: int = 26
    vit_num_attention_heads: int = 16
    vit_patch_size: int = 14
    vit_image_size: int = 980
    vit_layer_norm_eps: float = _BAGEL_VIT_LAYER_NORM_EPS
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"

    @property
    def latent_downsample(self) -> int:
        return self.vae_downsample * self.latent_patch_size

    @property
    def latent_token_capacity(self) -> int:
        return self.max_latent_size * self.max_latent_size

    @property
    def vit_token_capacity(self) -> int:
        return (self.vit_image_size // self.vit_patch_size) ** 2

    @property
    def latent_channel(self) -> int:
        return self.vae_z_channels

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size**2 * self.latent_channel

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BagelConfig":
        """Construct model configuration from loader-resolved checkpoint data."""

        llm_raw = raw["llm_config"]
        llm = LLMConfig(
            hidden_size=llm_raw["hidden_size"],
            intermediate_size=llm_raw["intermediate_size"],
            num_hidden_layers=llm_raw["num_hidden_layers"],
            num_attention_heads=llm_raw["num_attention_heads"],
            num_key_value_heads=llm_raw["num_key_value_heads"],
            vocab_size=llm_raw["vocab_size"],
            rms_norm_eps=llm_raw.get("rms_norm_eps", _BAGEL_RMS_NORM_EPS),
            rope_theta=llm_raw.get("rope_theta", 1e6),
            qk_norm=llm_raw.get("qk_norm", True),
        )
        vae = raw.get("vae_config", {})
        vit = raw.get("vit_config", {})
        return cls(
            llm=llm,
            start_of_image_id=raw.get("start_of_image_id", 151652),
            end_of_image_id=raw.get("end_of_image_id", 151653),
            vae_z_channels=vae.get("z_channels", 16),
            vae_downsample=vae.get("downsample", 8),
            latent_patch_size=raw.get("latent_patch_size", 2),
            max_latent_size=int(raw.get("max_latent_size", 32)),
            timestep_shift=raw.get("timestep_shift", 1.0),
            vit_hidden_size=vit.get("hidden_size", 1152),
            vit_intermediate_size=vit.get("intermediate_size", 4304),
            vit_num_hidden_layers=vit.get("num_hidden_layers", 27) - 1,
            vit_num_attention_heads=vit.get("num_attention_heads", 16),
            vit_patch_size=vit.get("patch_size", 14),
            vit_image_size=vit.get("image_size", 980),
            vit_layer_norm_eps=vit.get("layer_norm_eps", _BAGEL_VIT_LAYER_NORM_EPS),
            vit_max_num_patch_per_side=raw.get("vit_max_num_patch_per_side", 70),
            connector_act=raw.get("connector_act", "gelu_pytorch_tanh"),
        )


class _BagelGraph(nn.Module):
    """BAGEL neural graph: MoT language model, VAE, ViT, and flow-matching connectors."""

    def __init__(self, cfg: BagelConfig, *, layer_config: LayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.llm.hidden_size
        self.lm = MoTModel(
            MoTConfig(
                hidden_size=cfg.llm.hidden_size,
                intermediate_size=cfg.llm.intermediate_size,
                num_hidden_layers=cfg.llm.num_hidden_layers,
                num_attention_heads=cfg.llm.num_attention_heads,
                num_key_value_heads=cfg.llm.num_key_value_heads,
                vocab_size=cfg.llm.vocab_size,
                rms_norm_eps=cfg.llm.rms_norm_eps,
                rope_theta=cfg.llm.rope_theta,
                head_dim=cfg.llm.head_dim,
            ),
            layer_config=layer_config,
        )
        self.lm_head = ParallelLMHead(
            hidden,
            cfg.llm.vocab_size,
            layer_config=layer_config,
            bias=False,
        )
        self.vae2llm = LinearBase(cfg.patch_latent_dim, hidden, layer_config=layer_config)
        self.llm2vae = LinearBase(hidden, cfg.patch_latent_dim, layer_config=layer_config)
        self.time_embedder = TimestepEmbedder(hidden)
        self.latent_pos_embed = PositionEmbedding(cfg.max_latent_size, hidden, init_sincos=False)
        self.vae = AutoEncoder(default_ae_params())
        self.vit_model = SiglipNavitEncoder(
            SiglipNavitConfig(
                patch_size=cfg.vit_patch_size,
                hidden_size=cfg.vit_hidden_size,
                image_size=cfg.vit_image_size,
                num_attention_heads=cfg.vit_num_attention_heads,
                intermediate_size=cfg.vit_intermediate_size,
                num_hidden_layers=cfg.vit_num_hidden_layers,
                layer_norm_eps=cfg.vit_layer_norm_eps,
            ),
            layer_config=layer_config,
        )
        self.connector = MLPConnector(cfg.vit_hidden_size, hidden, cfg.connector_act)
        self.vit_pos_embed = PositionEmbedding(
            cfg.vit_max_num_patch_per_side, hidden, init_sincos=False
        )

    @property
    def num_layers(self) -> int:
        return self.cfg.llm.num_hidden_layers

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    def embed_tokens(self, ids: torch.Tensor, context: ForwardBatch) -> torch.Tensor:
        return self.lm.embed_tokens(ids, context.mesh)

    def gen_segment_embeds(
        self,
        num_vae,
        vae_pos_ids,
        x_t,
        timestep,
        context: ForwardBatch,
    ) -> torch.Tensor:
        """Marker/VAE-latent/timestep embeddings for one gen segment, ``[num_vae+2, hidden]``.

        Shared by graph denoise and image-commit paths so marker and latent
        embeddings follow one model layout.
        """
        hidden = self.cfg.llm.hidden_size
        total = int(num_vae) + 2
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=self.device,
        )
        marker_emb = self.embed_tokens(marker_ids, context).to(torch.bfloat16)
        x_t = x_t.to(device=self.device, dtype=torch.bfloat16).reshape(int(num_vae), -1)
        timestep_value = torch.as_tensor(timestep, device=self.device).reshape(1)
        timesteps = timestep_value.expand(int(num_vae))
        vae_emb = (
            self.vae2llm(x_t)
            + self.time_embedder(timesteps)
            + self.latent_pos_embed(vae_pos_ids.to(self.device))
        ).to(torch.bfloat16)
        embeds = torch.empty(total, hidden, dtype=torch.bfloat16, device=self.device)
        embeds[0] = marker_emb[0]
        embeds[1 : 1 + int(num_vae)] = vae_emb
        embeds[1 + int(num_vae)] = marker_emb[1]
        return embeds

    @torch.no_grad()
    def logits(
        self,
        hidden_last_row: torch.Tensor,
        context: ForwardBatch,
    ) -> torch.Tensor:
        return self.lm_head(hidden_last_row, context.mesh)

    @torch.no_grad()
    def velocity_from_hidden(self, hidden, num_vae) -> torch.Tensor:
        return self.llm2vae(hidden[1 : 1 + int(num_vae)].to(torch.bfloat16))

    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        return height // self.cfg.latent_downsample, width // self.cfg.latent_downsample

    @torch.no_grad()
    def vit_encode_batch(
        self,
        image_tensors: torch.Tensor,
        context: ForwardBatch,
    ) -> torch.Tensor:
        if image_tensors.ndim != 4:
            raise ValueError("BAGEL batched ViT encode expects NCHW pixels")
        image_tensors = image_tensors.to(self.device)
        batch, _channels, height, width = image_tensors.shape
        patch = self.cfg.vit_patch_size
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            patch,
            self.cfg.vit_max_num_patch_per_side,
        ).to(self.device)
        patches = (
            patchify_batch(image_tensors, patch)
            .reshape(
                -1,
                patch * patch * int(image_tensors.shape[1]),
            )
            .to(self.device, torch.bfloat16)
        )
        tokens_per_image = (height // patch) * (width // patch)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * tokens_per_image,
            tokens_per_image,
            dtype=torch.int32,
            device=self.device,
        )
        packed_pos_ids = pos_ids.repeat(batch)
        vit_out = self.vit_model(
            patches,
            {
                "position_ids": packed_pos_ids,
                "cu_seqlens": cu_seqlens,
                "seq_lens": (tokens_per_image,) * batch,
            },
            context,
        )
        emb = self.connector(vit_out) + self.vit_pos_embed(packed_pos_ids)
        return emb.reshape(batch, tokens_per_image, -1).to(torch.bfloat16)

    @torch.no_grad()
    def vae_encode_clean_batch(
        self,
        image_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if image_tensors.ndim != 4:
            raise ValueError("BAGEL batched VAE encode expects NCHW pixels")
        image_tensors = image_tensors.to(self.device)
        _batch, _channels, height, width = image_tensors.shape
        vae_dtype = next(self.vae.parameters()).dtype
        latent_images = self.vae.encode(image_tensors.to(vae_dtype))
        patch = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        h = height // self.cfg.latent_downsample
        w = width // self.cfg.latent_downsample
        latents = latent_images[:, :, : h * patch, : w * patch].reshape(
            int(latent_images.shape[0]),
            channels,
            h,
            patch,
            w,
            patch,
        )
        latents = torch.einsum("nchpwq->nhwpqc", latents).reshape(
            int(latent_images.shape[0]),
            -1,
            patch * patch * channels,
        )
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            self.cfg.latent_downsample,
            self.cfg.max_latent_size,
        ).to(self.device)
        return latents.to(torch.bfloat16), pos_ids, (h, w)

    @torch.no_grad()
    def vae_decode_batch(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        h, w = self.latent_hw(height, width)
        patch = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        latent_images = latents.reshape(-1, h, w, patch, patch, channels)
        latent_images = torch.einsum("nhwpqc->nchpwq", latent_images)
        latent_images = latent_images.reshape(-1, channels, h * patch, w * patch)
        latent_images = latent_images.to(next(self.vae.parameters()).dtype)
        return (self.vae.decode(latent_images) * 0.5 + 0.5).clamp(0, 1)


_BAGEL_STACKED_WEIGHTS: WeightNameMap = (
    ("qkv_proj_moe_gen", "q_proj_moe_gen", "q"),
    ("qkv_proj_moe_gen", "k_proj_moe_gen", "k"),
    ("qkv_proj_moe_gen", "v_proj_moe_gen", "v"),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _bagel_checkpoint_name(name: str) -> str | None:
    exact = {
        "language_model.model.embed_tokens.weight": "lm.embed_tokens.weight",
        "language_model.model.norm.weight": "lm.norm.weight",
        "language_model.model.norm_moe_gen.weight": "lm.norm_moe_gen.weight",
        "language_model.lm_head.weight": "lm_head.weight",
    }
    if name in exact:
        return exact[name]
    if name.startswith("language_model.model.layers."):
        return ("lm.layers." + name.removeprefix("language_model.model.layers.")).replace(
            ".self_attn.", "."
        )
    if name.startswith("vit_model.vision_model.embeddings."):
        return "vit_model." + name.removeprefix("vit_model.vision_model.embeddings.")
    if name.startswith("vit_model.vision_model.encoder."):
        return (
            ("vit_model.encoder." + name.removeprefix("vit_model.vision_model.encoder."))
            .replace(".mlp.fc1.", ".mlp.0.")
            .replace(".mlp.fc2.", ".mlp.2.")
        )
    if name.startswith("vit_model.vision_model.post_layernorm."):
        return "vit_model.encoder.post_layernorm." + name.removeprefix(
            "vit_model.vision_model.post_layernorm."
        )
    if name.startswith(
        (
            "connector.",
            "vit_pos_embed.",
            "vae2llm.",
            "llm2vae.",
            "time_embedder.",
            "latent_pos_embed.",
        )
    ):
        return name
    return None


class BagelForConditionalGeneration(ExecutionModel):
    """Stateless BAGEL neural graph for the declared MoT, ViT, and VAE routes."""

    def load_weights(self, weights: Iterable[WeightHandle]) -> LoadReport:
        """Load BAGEL's root checkpoint into its language, vision, and connector graph."""

        parameter_names = set(dict(self.model.named_parameters()))
        report = LoadReport()
        for handle in weights:
            source_name = handle.name
            renamed = _bagel_checkpoint_name(source_name)
            if renamed is None:
                report.skipped.append(source_name)
                continue
            target_name, shard_id = stacked_weight_name(renamed, _BAGEL_STACKED_WEIGHTS)
            if target_name not in parameter_names:
                report.unexpected.append(source_name)
                continue
            parameter = dict(self.model.named_parameters())[target_name]
            load_parameter_weight(parameter, handle, shard_id)
            report.loaded.add(target_name)
        return report

    def load_autoencoder_weights(self, weights: Iterable[WeightHandle]) -> LoadReport:
        parameter_names = set(dict(self.model.vae.named_parameters()))
        report = LoadReport()
        for handle in weights:
            if handle.name not in parameter_names:
                report.unexpected.append(handle.name)
                continue
            parameter = dict(self.model.vae.named_parameters())[handle.name]
            load_parameter_weight(parameter, handle)
            report.loaded.add(handle.name)
        return report

    def checkpoint_parameter_names(self) -> set[str]:
        return {name for name, _ in self.model.named_parameters() if not name.startswith("vae.")}

    def __init__(
        self,
        config: BagelConfig,
        *,
        layer_config: LayerConfig,
        graph: _BagelGraph | None = None,
    ) -> None:
        super().__init__()
        if graph is not None and graph.cfg != config:
            raise ValueError("BAGEL graph and root must use the same immutable configuration")
        self.cfg = config
        self._parallel = layer_config.parallel
        self.model = graph if graph is not None else _BagelGraph(config, layer_config=layer_config)
        llm = self.cfg.llm
        self.architecture = "BagelForConditionalGeneration"
        self.generation = GenerationPipeline(
            latent_downsample=int(self.cfg.latent_downsample),
            prediction="velocity",
            prediction_dtype="bfloat16",
            schedule_direction=ScheduleDirection.DESCENDING,
            schedule_shift_domain=ScheduleShiftDomain.TIME,
            max_latent_tokens=int(self.cfg.latent_token_capacity),
            max_vae_grid_tokens=(int(self.cfg.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS),
            commit_marker_tokens=_BAGEL_IMAGE_MARKER_TOKENS,
            rope_advance=2,
            max_cfg_branches=3,
            latent_layout=LatentLayout.PATCH_TOKENS,
            latent_channels=int(self.cfg.latent_channel),
            latent_patch_size=int(self.cfg.latent_patch_size),
            positions=PositionLayout.TEMPORAL,
            materialization=Materialization.DECODE_ROUTE,
            text_unconditional=BranchSource.NEGATIVE_OR_START,
            image_unconditional=BranchSource.CONDITIONING,
            cfg_recipe=CfgRecipe.IMAGE_OVER_TEXT,
            timestep_shift=float(self.cfg.timestep_shift),
        )
        self.image_processor = ImageProcessor(
            vit=TowerTransform(
                resize=StrideResize(
                    max_size=int(self.cfg.vit_image_size),
                    min_size=_BAGEL_VIT_MIN_SIZE,
                    stride=int(self.cfg.vit_patch_size),
                    max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
                ),
            ),
            vae=TowerTransform(
                resize=StrideResize(
                    max_size=_BAGEL_VAE_MAX_SIZE,
                    min_size=_BAGEL_VAE_MIN_SIZE,
                    stride=_BAGEL_VAE_STRIDE,
                    max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
                ),
            ),
            feature_injection=FeatureInjection(
                layout=FeatureLayout.FRAMED,
                positions=PositionLayout.TEMPORAL,
                start_token_id=int(self.cfg.start_of_image_id),
                end_token_id=int(self.cfg.end_of_image_id),
            ),
        )
        self.cache_geometry = CacheGeometry(
            num_layers=int(llm.num_hidden_layers),
            num_attention_heads=local_attention_head_count(
                int(llm.num_attention_heads), parallel=self._parallel
            ),
            num_kv_heads=local_kv_head_count(int(llm.num_key_value_heads), parallel=self._parallel),
            head_dim=int(llm.head_dim),
            dtype="bfloat16",
            store_dtype="bfloat16",
        )
        self.resource_geometry = ResourceGeometry(
            encoder_cache_entries=256,
            latent_downsample=int(self.cfg.latent_downsample),
        )
        self.supported_work = frozenset(
            {
                ForwardMode.TOKEN_EXTEND,
                ForwardMode.TOKEN_DECODE,
                ForwardMode.TOKEN_VERIFY,
                ForwardMode.MEDIA_PREPARE,
                ForwardMode.MEDIA_DENOISE,
                ForwardMode.ENCODE_VISION,
                ForwardMode.ENCODE_LATENT,
                ForwardMode.MATERIALIZE,
                ForwardMode.TRANSFER_PRODUCT,
                ForwardMode.TRANSFER_KV_PUBLISH,
                ForwardMode.TRANSFER_KV_INSTALL,
            }
        )
        self.max_vit_grid_tokens = int(self.cfg.vit_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS
        self.vocab_size = int(llm.vocab_size)
        self.hidden_size = int(llm.hidden_size)
        self.text_max_tokens = max(
            int(llm.max_position_embeddings),
            int(self.cfg.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS,
        )
        self.text_topology = ("tp", "tower")
        self.tensorized_mixed = True

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        batch = forward_batch
        decode_positions: torch.Tensor | None = None
        if batch.forward_mode is AttentionMode.PAGED_DECODE:
            if batch.flow_row_indices:
                raise TypeError("BAGEL paged decode accepts token rows only")
            decode_positions = positions
        token_embeds = self.model.embed_tokens(input_ids.reshape(-1), batch)
        if batch.input_embeddings is not None:
            if batch.embedding_mask is None:
                raise RuntimeError("BAGEL embedding input lost its selection mask")
            token_embeds = torch.where(
                batch.embedding_mask.reshape(-1, 1),
                batch.input_embeddings.to(dtype=token_embeds.dtype),
                token_embeds,
            )
        chunks: list[torch.Tensor | None] = [None] * batch.row_count
        token_offset = 0
        for row_index, count in zip(
            batch.token_row_indices,
            tuple(batch.query_lens_cpu[index] for index in batch.token_row_indices),
            strict=True,
        ):
            chunks[row_index] = token_embeds[token_offset : token_offset + count]
            token_offset += count
        for flow_index, row_index in enumerate(batch.flow_row_indices):
            if batch.flow_conditioning[flow_index] is not None:
                raise TypeError("BAGEL denoise does not accept external feature conditioning")
            image_tokens = int(batch.flow_image_tokens[flow_index])
            latent_tokens = image_tokens - _BAGEL_IMAGE_MARKER_TOKENS
            latent = batch.flow_latents[flow_index]
            if latent_tokens < 1 or int(latent.shape[-2]) != latent_tokens:
                raise ValueError("BAGEL flow latent does not match its image-token geometry")
            chunks[row_index] = self.model.gen_segment_embeds(
                latent_tokens,
                batch.flow_positions[flow_index],
                latent,
                batch.flow_timesteps[flow_index],
                batch,
            )
        if any(value is None for value in chunks):
            raise RuntimeError("BAGEL forward batch contains an unbound row")
        typed_chunks = tuple(value for value in chunks if value is not None)
        normalized: list[torch.Tensor] = []
        for chunk in typed_chunks:
            chunk = chunk.reshape(-1, chunk.shape[-1]).to(torch.bfloat16)
            normalized.append(chunk)
        return self.model.lm(
            torch.cat(normalized, dim=0),
            batch,
            positions=decode_positions,
        )

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        row_lengths = [0] * batch.row_count
        for row_index, count in zip(
            batch.token_row_indices,
            tuple(batch.query_lens_cpu[index] for index in batch.token_row_indices),
            strict=True,
        ):
            row_lengths[row_index] = count
        for flow_index, row_index in enumerate(batch.flow_row_indices):
            row_lengths[row_index] = int(batch.flow_image_tokens[flow_index])
        row_hidden: list[torch.Tensor] = []
        offset = 0
        for count in row_lengths:
            row_hidden.append(hidden[offset : offset + count])
            offset += count
        selection_by_row = dict(zip(batch.token_row_indices, batch.token_selections, strict=True))
        row_hidden_values = tuple(row_hidden)
        projected_rows = tuple(
            index
            for index, selection in selection_by_row.items()
            if selection is not TokenSelection.HIDDEN
        )
        projected: torch.Tensor | None = None
        if projected_rows:
            if (
                len(projected_rows) == batch.row_count
                and all(
                    selection is TokenSelection.LAST_LOGITS
                    for selection in selection_by_row.values()
                )
                and all(int(value.shape[0]) == 1 for value in row_hidden_values)
            ):
                selected = hidden
            else:
                selected_rows = tuple(
                    row_hidden_values[index]
                    if selection_by_row[index] is TokenSelection.ALL_LOGITS
                    else row_hidden[index][-1:]
                    for index in projected_rows
                )
                selected = (
                    selected_rows[0] if len(selected_rows) == 1 else torch.cat(selected_rows, dim=0)
                )
            projected = self.model.logits(selected, batch)

        outputs: list[torch.Tensor] = []
        projected_offset = 0
        flow_by_row = {row_index: index for index, row_index in enumerate(batch.flow_row_indices)}
        for index in range(batch.row_count):
            value_hidden = row_hidden_values[index]
            selection = selection_by_row.get(index)
            if selection is not None:
                if selection is TokenSelection.HIDDEN:
                    value = value_hidden
                else:
                    if projected is None:
                        raise RuntimeError("BAGEL projected output buffer is missing")
                    count = (
                        int(value_hidden.shape[0]) if selection is TokenSelection.ALL_LOGITS else 1
                    )
                    value = projected[projected_offset : projected_offset + count]
                    projected_offset += count
                outputs.append(value)
            else:
                flow_index = flow_by_row[index]
                prediction = self.model.velocity_from_hidden(
                    value_hidden,
                    int(batch.flow_image_tokens[flow_index]) - _BAGEL_IMAGE_MARKER_TOKENS,
                )
                outputs.append(prediction)
        return ForwardOutput(tuple(outputs))

    def encode(self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch) -> ForwardOutput:
        features = self.model.vit_encode_batch(torch.stack(pixels, dim=0), batch)
        return ForwardOutput(tuple(features[index] for index in range(len(pixels))))

    def encode_latent(self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch) -> ForwardOutput:
        del batch
        latents, _positions, _shape = self.model.vae_encode_clean_batch(torch.stack(pixels, dim=0))
        return ForwardOutput(tuple(latents[index] for index in range(len(pixels))))

    def decode_latent(
        self, latents: tuple[torch.Tensor, ...], batch: ForwardBatch
    ) -> ForwardOutput:
        geometry = set(zip(batch.decode_heights, batch.decode_widths, strict=True))
        if len(geometry) != 1:
            raise ValueError("BAGEL latent decode requires one image geometry")
        height, width = next(iter(geometry))
        decoded = self.model.vae_decode_batch(
            torch.stack(latents, dim=0),
            height,
            width,
        )
        return ForwardOutput(tuple(decoded[index] for index in range(len(latents))))
