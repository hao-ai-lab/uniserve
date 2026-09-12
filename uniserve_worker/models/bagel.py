"""Defines BAGEL language, vision, latent, and flow-matching execution routes.

The model shares one Mixture-of-Transformers graph across autoregressive token
rows and image-generation rows. Vision features and VAE latents are projected
into that graph's hidden space, while route-specific projection restores logits,
latent velocity, image features, or decoded pixels for the scheduler.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.nn as nn

from uniserve_worker.execution.batch import ForwardMode, PipelineStage, TransferMode

from ..execution.forward_batch import (
    AttentionMode,
    ForwardBatch,
    ForwardOutput,
)
from ..loader.component import construct_owned_module
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
    local_kv_head_offset,
)
from ..nn.decoder import MoTConfig, MoTModel
from ..nn.diffusion import (
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from ..nn.diffusion.cfg import CfgRecipe
from ..nn.logits import project_outputs
from ..nn.vae import AutoEncoder, default_ae_params
from ..nn.vision import (
    PositionEmbedding,
    SiglipNavitConfig,
    SiglipNavitEncoder,
    get_flattened_position_ids_extrapolate,
    patchify_batch,
)
from ..nn.vocab_parallel_embedding import vocabulary_partition
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

if TYPE_CHECKING:
    from ..loader.component import CheckpointComponent, ModelBuildContext, ModelConstruction


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
        """Return the per-head query/key/value width."""

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
        """Return the pixel-to-latent-token downsampling factor."""

        return self.vae_downsample * self.latent_patch_size

    @property
    def latent_token_capacity(self) -> int:
        """Return the maximum flattened latent patch count."""

        return self.max_latent_size * self.max_latent_size

    @property
    def vit_token_capacity(self) -> int:
        """Return the maximum square ViT patch count."""

        return (self.vit_image_size // self.vit_patch_size) ** 2

    @property
    def latent_channel(self) -> int:
        """Return the VAE channel count exposed to latent patch packing."""

        return self.vae_z_channels

    @property
    def patch_latent_dim(self) -> int:
        """Return the flattened feature width of one latent patch."""

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
    """Owns the MoT, VAE, ViT, and flow-matching projections as one neural graph."""

    lm_head: ParallelLMHead | None
    vae2llm: LinearBase | None
    llm2vae: LinearBase | None
    time_embedder: TimestepEmbedder | None
    latent_pos_embed: PositionEmbedding | None

    def __init__(
        self,
        cfg: BagelConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Construct all route components against one params and quantization policy."""

        super().__init__()
        self.generation_device = generation_device
        self.cfg = cfg
        hidden = cfg.llm.hidden_size

        # Text and flow rows meet in the common MoT hidden space.
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
            layer_config=layer_config.child("language_model.model"),
            generation_device=generation_device,
        )
        self.nonresident_parameters = frozenset(
            f"lm.{name}" for name in self.lm.nonresident_parameters
        )
        pipeline = self.lm.pipeline
        declarations = (
            (
                "lm_head",
                pipeline.last,
                lambda: ParallelLMHead(
                    hidden,
                    cfg.llm.vocab_size,
                    layer_config=layer_config,
                    prefix="language_model.lm_head",
                    bias=False,
                ),
            ),
            (
                "vae2llm",
                pipeline.first,
                lambda: LinearBase(
                    cfg.patch_latent_dim, hidden, layer_config=layer_config, prefix="vae2llm"
                ),
            ),
            (
                "llm2vae",
                pipeline.last,
                lambda: LinearBase(
                    hidden, cfg.patch_latent_dim, layer_config=layer_config, prefix="llm2vae"
                ),
            ),
            ("time_embedder", pipeline.first, lambda: TimestepEmbedder(hidden)),
            (
                "latent_pos_embed",
                pipeline.first,
                lambda: PositionEmbedding(cfg.max_latent_size, hidden, init_sincos=False),
            ),
        )
        for name, resident, factory in declarations:
            module, nonresident = construct_owned_module(factory, resident=resident)
            setattr(self, name, module)
            self.nonresident_parameters |= {f"{name}.{parameter}" for parameter in nonresident}
        self.vae = AutoEncoder(default_ae_params())

        # Vision patches retain their own encoder before projection into MoT width.
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
            layer_config=layer_config.child("vit_model.vision_model"),
        )
        self.connector = MLPConnector(cfg.vit_hidden_size, hidden, cfg.connector_act)
        self.vit_pos_embed = PositionEmbedding(
            cfg.vit_max_num_patch_per_side, hidden, init_sincos=False
        )

    @property
    def num_layers(self) -> int:
        """Return the transformer layer count used by KV-cache allocation."""

        return self.cfg.llm.num_hidden_layers

    @property
    def device(self) -> torch.device:
        """Return the device that owns route inputs and model outputs."""

        return next(self.lm.parameters()).device

    def embed_tokens(self, ids: torch.Tensor, context: ForwardBatch) -> torch.Tensor:
        """Embed token identifiers with the batch's tensor-parallel mesh."""

        if self.lm.embed_tokens is None:
            raise RuntimeError("token embedding belongs to the first pipeline stage")
        return self.lm.embed_tokens(ids)

    def gen_segment_embeds(
        self,
        num_vae,
        vae_pos_ids,
        x_t,
        timestep,
        context: ForwardBatch,
    ) -> torch.Tensor:
        """Build a ``[num_vae + 2, hidden]`` embedding sequence for one generation segment.

        Start/end image markers frame VAE latents augmented with timestep and
        position embeddings. Graph denoise and image commit share this layout.
        """
        assert self.vae2llm is not None and self.time_embedder is not None
        assert self.latent_pos_embed is not None
        hidden = self.cfg.llm.hidden_size
        total = int(num_vae) + 2

        # Image markers frame the latent span in the language-model sequence.
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=self.device,
        )
        marker_emb = self.embed_tokens(marker_ids, context).to(torch.bfloat16)
        x_t = x_t.to(device=self.device, dtype=torch.bfloat16).reshape(int(num_vae), -1)
        timestep_value = torch.as_tensor(timestep, device=self.device).reshape(1)
        timesteps = timestep_value.expand(int(num_vae))

        # Content, diffusion time, and two-dimensional location share the same width.
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
    def velocity_from_hidden(self, hidden, num_vae) -> torch.Tensor:
        """Project the latent span, excluding its two marker rows, into velocity."""

        assert self.llm2vae is not None
        return self.llm2vae(hidden[1 : 1 + int(num_vae)].to(torch.bfloat16))

    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        """Convert output pixel geometry to latent patch-grid geometry."""

        return height // self.cfg.latent_downsample, width // self.cfg.latent_downsample

    @torch.no_grad()
    def vit_encode_batch(
        self,
        image_tensors: torch.Tensor,
        context: ForwardBatch,
    ) -> torch.Tensor:
        """Encode a uniform NCHW image batch into MoT-width patch features."""

        if image_tensors.ndim != 4:
            raise ValueError("BAGEL batched ViT encode expects NCHW pixels")

        image_tensors = image_tensors.to(self.device)
        batch, _channels, height, width = image_tensors.shape
        patch = self.cfg.vit_patch_size

        # Every image shares one extrapolated grid and one packed sequence length.
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

        # Packed attention isolates images through cumulative sequence boundaries.
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
        """Encode NCHW pixels and pack the clean latent grid into patch tokens."""

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

        # Patch pixels become the feature axis while the spatial grid becomes sequence.
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
        """Unpack latent patch tokens and decode them into unit-range NCHW pixels."""

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
    """Map an external BAGEL checkpoint name into the owned graph namespace."""

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
    """Exposes the stateless BAGEL graph through scheduler-owned execution routes."""

    ordered_collective_execution = True

    @classmethod
    def build_checkpoint(
        cls, config: dict[str, Any], context: ModelBuildContext
    ) -> ModelConstruction:
        """Interpret BAGEL component metadata and declare its two checkpoint namespaces."""

        import json
        import math

        from ..foundation.errors import unsupported_setup
        from ..loader.component import ModelConstruction

        raw = dict(config)
        for config_field, filename in (
            ("llm_config", "llm_config.json"),
            ("vit_config", "vit_config.json"),
            ("vae_config", "vae_config.json"),
        ):
            if config_field in raw:
                continue
            path = context.root / filename
            if not path.is_file():
                raise unsupported_setup(
                    f"BAGEL checkpoint is missing {filename!r} for {config_field!r}"
                )
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise unsupported_setup(
                    f"BAGEL checkpoint file {filename!r} must contain an object"
                )
            raw[config_field] = value
        positions = context.sources[0].preview_shape("latent_pos_embed.pos_embed")[0]
        max_latent_size = math.isqrt(positions)
        if max_latent_size * max_latent_size != positions:
            raise unsupported_setup(f"BAGEL latent position count {positions} is not square")
        raw["max_latent_size"] = max_latent_size
        prepared = BagelConfig.from_mapping(raw)
        model = cls(
            prepared,
            layer_config=context.packed_decoder_layers("model"),
            generation_device=(
                None
                if context.request.execution.generation_device is None
                else torch.device(context.request.execution.generation_device)
            ),
        )
        return ModelConstruction(model.checkpoint_components(), lambda: model, prepared)

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare language/vision tensors and the independently serialized autoencoder."""

        from ..loader.component import CheckpointComponent

        return (
            CheckpointComponent(
                self.model,
                map_weights=self.load_weights,
                included=frozenset(self.checkpoint_parameter_names()),
                nonresident=self.model.nonresident_parameters,
                module_devices=(
                    ()
                    if self.generation_device is None
                    else tuple(
                        (name, self.generation_device)
                        for name, _ in self.model.named_modules()
                        if name.endswith("_moe_gen")
                    )
                ),
            ),
            CheckpointComponent(
                self.model.vae,
                source="autoencoder",
                map_weights=self.load_autoencoder_weights,
                optional=frozenset(
                    name
                    for name, _ in self.model.vae.named_parameters()
                    if name == "reg" or name.startswith("reg.")
                ),
            ),
        )

    def load_weights(self, weights: Iterable[WeightHandle]) -> LoadReport:
        """Load BAGEL's root checkpoint into its language, vision, and connector graph."""

        parameters = dict(self.model.named_parameters())
        report = LoadReport()

        # Decoder projections are packed; the vision encoder owns separate
        # Q/K/V parameters despite sharing their checkpoint suffixes.
        for handle in weights:
            source_name = handle.name
            renamed = _bagel_checkpoint_name(source_name)
            if renamed is None:
                report.unexpected.append(source_name)
                continue
            target_name, shard_id = (
                stacked_weight_name(renamed, _BAGEL_STACKED_WEIGHTS)
                if renamed.startswith("lm.layers.")
                else (renamed, None)
            )
            if target_name not in parameters:
                if target_name in self.model.nonresident_parameters:
                    report.skipped.append(source_name)
                    continue
                report.unexpected.append(source_name)
                continue
            parameter = parameters[target_name]
            load_parameter_weight(parameter, handle, shard_id)
            report.loaded.add(target_name)
        return report

    def load_autoencoder_weights(self, weights: Iterable[WeightHandle]) -> LoadReport:
        """Load a VAE-only checkpoint directly into the autoencoder namespace."""

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
        """Return root-checkpoint parameters, excluding the separately loaded VAE."""

        return {name for name, _ in self.model.named_parameters() if not name.startswith("vae.")}

    def __init__(
        self,
        config: BagelConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
        graph: _BagelGraph | None = None,
    ) -> None:
        """Bind graph geometry and route capabilities to the worker execution model."""

        super().__init__()
        self.generation_device = generation_device
        if graph is not None and graph.cfg != config:
            raise ValueError("BAGEL graph and root must use the same immutable configuration")
        self.cfg = config
        self._parallel = layer_config.communicator
        self.model = (
            graph
            if graph is not None
            else _BagelGraph(config, layer_config=layer_config, generation_device=generation_device)
        )
        llm = self.cfg.llm
        self.architecture = "BagelForConditionalGeneration"

        # Generation metadata defines how scheduler coordinates map to model rows.
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

        # Vision and VAE routes use independent raster bounds but one feature contract.
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

        # Runtime pools are sized from rank-local attention and latent geometry.
        self.cache_geometry = CacheGeometry(
            num_layers=len(self.model.lm.pipeline.layers),
            total_layers=int(llm.num_hidden_layers),
            layer_offset=self.model.lm.pipeline.layers.start,
            num_attention_heads=local_attention_head_count(
                int(llm.num_attention_heads),
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
            num_kv_heads=local_kv_head_count(
                int(llm.num_key_value_heads),
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
            total_kv_heads=int(llm.num_key_value_heads),
            kv_head_offset=local_kv_head_offset(
                int(llm.num_key_value_heads),
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
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
                ForwardMode.PREFILL,
                ForwardMode.DECODE,
                ForwardMode.VERIFY,
                PipelineStage.LATENT_PREPARATION,
                PipelineStage.DENOISING,
                PipelineStage.VISION_ENCODING,
                PipelineStage.LATENT_ENCODING,
                PipelineStage.IMAGE_DECODING,
                TransferMode.TENSOR,
                TransferMode.KV_PUBLISH,
                TransferMode.KV_INSTALL,
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
        """Build mixed token/flow rows and execute their shared transformer forward."""

        batch = forward_batch
        decode_positions: torch.Tensor | None = None

        # Paged decode has a compact positional contract and cannot carry flow rows.
        if batch.attention_mode is AttentionMode.PAGED_DECODE:
            if batch.flow_row_indices:
                raise TypeError("BAGEL paged decode accepts token rows only")
            decode_positions = positions

        if not self.model.lm.pipeline.first:
            return self.model.lm(None, batch, positions=decode_positions)

        # External image features replace only positions selected by the embedding mask.
        token_embeds = self.model.embed_tokens(input_ids.reshape(-1), batch)
        if batch.input_embeddings is not None:
            if batch.embedding_mask is None:
                raise RuntimeError("BAGEL embedding input lost its selection mask")
            token_embeds = torch.where(
                batch.embedding_mask.reshape(-1, 1),
                batch.input_embeddings.to(dtype=token_embeds.dtype),
                token_embeds,
            )

        # Restore packed token embeddings to scheduler row order.
        chunks: list[torch.Tensor | None] = [None] * batch.row_count
        token_offset = 0
        for row_index, count in zip(
            batch.token_row_indices,
            tuple(batch.query_lens_cpu[index] for index in batch.token_row_indices),
            strict=True,
        ):
            chunks[row_index] = token_embeds[token_offset : token_offset + count]
            token_offset += count

        # Flow rows contribute framed latent segments in the same hidden space.
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

        # The transformer consumes one contiguous BF16 sequence with batch metadata.
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
        """Project text through the shared head and preserve latent prediction math."""

        return project_outputs(
            hidden,
            batch,
            self.model.lm_head,
            project_flow=lambda rows, index: self.model.velocity_from_hidden(
                rows, int(batch.flow_image_tokens[index]) - _BAGEL_IMAGE_MARKER_TOKENS
            ),
            pipeline=self.model.lm.pipeline,
            vocabulary=vocabulary_partition(self.vocab_size, self._parallel),
            flow_dtype=torch.bfloat16,
        )

    def encode(self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch) -> ForwardOutput:
        """Encode a uniform image batch into language-width vision features."""

        features = self.model.vit_encode_batch(torch.stack(pixels, dim=0), batch)
        return ForwardOutput(tuple(features[index] for index in range(len(pixels))))

    def encoder_latent(
        self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch
    ) -> ForwardOutput:
        """Encode a uniform image batch into clean VAE latent patch tokens."""

        del batch
        latents, _positions, _shape = self.model.vae_encode_clean_batch(torch.stack(pixels, dim=0))
        return ForwardOutput(tuple(latents[index] for index in range(len(pixels))))

    def decode_latent(
        self, latents: tuple[torch.Tensor, ...], batch: ForwardBatch
    ) -> ForwardOutput:
        """Decode a uniform latent batch into unit-range image tensors."""

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
