"""BAGEL language, vision, latent, and flow-matching numerical composition.

Independent text and diffusion calls share one Mixture-of-Transformers backbone.
Vision features and VAE latents enter its hidden space through ordinary modules;
route-specific projections restore vocabulary logits and latent velocity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.nn as nn

from uniserve_worker.modeling.geometry import CacheGeometry
from uniserve_worker.modeling.tensors import AttentionMode, PositionLayout

from ..loader.component import construct_owned_module
from ..loader.handles import WeightHandle
from ..loader.mapping import LoadReport, WeightNameMap, stacked_weight_name
from ..loader.weight_loaders import load_parameter_weight
from ..modeling.batch import (
    DiffusionBatch,
    TensorOutput,
)
from ..modeling.components import Call, CallSpec, ComponentSpec
from ..modeling.context import BuildContext
from ..modeling.decoder import DecoderMixin
from ..modeling.diffusion import DiffusionMixin
from ..modeling.encoder import EncodeKind, EncoderMixin
from ..modeling.image_diffusion import (
    BranchSource,
    ImageDiffusion,
    LatentLayout,
)
from ..modeling.inputs import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    StrideResize,
    TowerTransform,
)
from ..modeling.model import Model
from ..modeling.tensors import TensorViews
from ..modeling.text import TextMixin
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
from ..nn.vae import AutoEncoder, default_ae_params
from ..nn.vae.patch import PatchAutoencoder
from ..nn.vision import (
    PatchEncoder,
    PositionEmbedding,
    SiglipNavitConfig,
    SiglipNavitEncoder,
)
from ..nn.vocab_parallel_embedding import vocabulary_partition

if TYPE_CHECKING:
    from ..loader.component import CheckpointComponent


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
    ) -> None:
        """Construct all route components against one params and quantization policy."""

        super().__init__()
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
        self.vae = PatchAutoencoder(
            AutoEncoder(default_ae_params()),
            patch_size=cfg.latent_patch_size,
            downsample=cfg.latent_downsample,
            channels=cfg.latent_channel,
            latent_dtype=torch.bfloat16,
        )

        # The shared encoder owns patch packing and projection mathematics.
        self.vision = PatchEncoder(
            SiglipNavitEncoder(
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
            ),
            MLPConnector(cfg.vit_hidden_size, hidden, cfg.connector_act),
            PositionEmbedding(cfg.vit_max_num_patch_per_side, hidden, init_sincos=False),
            patch_size=cfg.vit_patch_size,
            position_grid=cfg.vit_max_num_patch_per_side,
            hidden_size=hidden,
            dtype=torch.bfloat16,
        )

    @property
    def num_layers(self) -> int:
        """Return the transformer layer count used by KV-cache allocation."""

        return self.cfg.llm.num_hidden_layers

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed token identifiers with the batch's tensor-parallel mesh."""

        if self.lm.embed_tokens is None:
            raise RuntimeError("token embedding belongs to the first pipeline stage")
        return self.lm.embed_tokens(ids)

    def embed_latents(
        self,
        latent_tokens: int,
        positions: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Build a ``[latent_tokens + 2, hidden]`` embedding sequence for one generation segment.

        Start/end image markers frame VAE latents augmented with timestep and
        position embeddings. Inputs must already reside with these modules;
        conversion to BF16 preserves the checkpoint's projection arithmetic.
        """

        assert self.vae2llm is not None and self.time_embedder is not None
        assert self.latent_pos_embed is not None
        if positions.device != latents.device or timestep.device != latents.device:
            raise ValueError(
                "latent embeddings require colocated latent, position, and time inputs"
            )
        hidden = self.cfg.llm.hidden_size
        total = latent_tokens + 2

        # Image markers frame the latent span in the language-model sequence.
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=latents.device,
        )
        marker_emb = self.embed_tokens(marker_ids).to(torch.bfloat16)
        latents = latents.to(dtype=torch.bfloat16).reshape(latent_tokens, -1)
        timestep_value = timestep.reshape(1)
        timesteps = timestep_value.expand(latent_tokens)

        # Content, diffusion time, and two-dimensional location share the same width.
        vae_emb = (
            self.vae2llm(latents) + self.time_embedder(timesteps) + self.latent_pos_embed(positions)
        ).to(torch.bfloat16)
        embeds = torch.empty(total, hidden, dtype=torch.bfloat16, device=latents.device)
        embeds[0] = marker_emb[0]
        embeds[1 : 1 + latent_tokens] = vae_emb
        embeds[1 + latent_tokens] = marker_emb[1]
        return embeds

    @torch.no_grad()
    def velocity_from_hidden(self, hidden, num_vae) -> torch.Tensor:
        """Project the latent span, excluding its two marker rows, into velocity."""

        assert self.llm2vae is not None
        return self.llm2vae(hidden[1 : 1 + int(num_vae)].to(torch.bfloat16))


_BAGEL_STACKED_WEIGHTS: WeightNameMap = (
    ("flow_qkv.projection", "q_proj_moe_gen", "q"),
    ("flow_qkv.projection", "k_proj_moe_gen", "k"),
    ("flow_qkv.projection", "v_proj_moe_gen", "v"),
    ("flow_qkv.projection", "qkv_proj_moe_gen", None),
    ("flow_qkv.query_norm", "q_norm_moe_gen", None),
    ("flow_qkv.key_norm", "k_norm_moe_gen", None),
    ("text_qkv.projection", "q_proj", "q"),
    ("text_qkv.projection", "k_proj", "k"),
    ("text_qkv.projection", "v_proj", "v"),
    ("text_qkv.projection", "qkv_proj", None),
    ("text_qkv.query_norm", "q_norm", None),
    ("text_qkv.key_norm", "k_norm", None),
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
        return "vision.encoder." + name.removeprefix("vit_model.vision_model.embeddings.")
    if name.startswith("vit_model.vision_model.encoder."):
        return (
            ("vision.encoder.encoder." + name.removeprefix("vit_model.vision_model.encoder."))
            .replace(".mlp.fc1.", ".mlp.0.")
            .replace(".mlp.fc2.", ".mlp.2.")
        )
    if name.startswith("vit_model.vision_model.post_layernorm."):
        return "vision.encoder.encoder.post_layernorm." + name.removeprefix(
            "vit_model.vision_model.post_layernorm."
        )
    if name.startswith("connector."):
        return "vision.projection." + name.removeprefix("connector.")
    if name.startswith("vit_pos_embed."):
        return "vision.position_embed." + name.removeprefix("vit_pos_embed.")
    if name.startswith(
        (
            "vae2llm.",
            "llm2vae.",
            "time_embedder.",
            "latent_pos_embed.",
        )
    ):
        return name
    return None


class BagelForConditionalGeneration(TextMixin, EncoderMixin, DiffusionMixin, DecoderMixin, Model):
    """Compose BAGEL text, vision, latent, and diffusion capabilities."""

    config: BagelConfig

    @classmethod
    def components(cls, config: object) -> tuple[ComponentSpec, ...]:
        """Declare the numerical calls sharing this model graph."""

        return (
            ComponentSpec(
                "model",
                (
                    CallSpec(Call.TEXT, groups=("tp", "sp", "pp")),
                    CallSpec(Call.DIFFUSION, groups=("tp", "sp", "pp")),
                    CallSpec(Call.ENCODE_VISION),
                    CallSpec(Call.ENCODE_LATENT),
                    CallSpec(Call.DECODE_IMAGE),
                ),
            ),
        )

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare language/vision tensors and the independently serialized autoencoder."""

        from ..loader.component import CheckpointComponent

        return (
            CheckpointComponent(
                self.model,
                map_weights=self.load_weights,
                included=frozenset(self.checkpoint_parameter_names()),
                nonresident=self.model.nonresident_parameters,
            ),
            CheckpointComponent(
                self.model.vae.autoencoder,
                source="autoencoder",
                map_weights=self.load_autoencoder_weights,
                optional=frozenset(
                    name
                    for name, _ in self.model.vae.autoencoder.named_parameters()
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

        parameter_names = set(dict(self.model.vae.autoencoder.named_parameters()))
        report = LoadReport()
        for handle in weights:
            if handle.name not in parameter_names:
                report.unexpected.append(handle.name)
                continue
            parameter = dict(self.model.vae.autoencoder.named_parameters())[handle.name]
            load_parameter_weight(parameter, handle)
            report.loaded.add(handle.name)
        return report

    def checkpoint_parameter_names(self) -> set[str]:
        """Return root-checkpoint parameters, excluding the separately loaded VAE."""

        return {name for name, _ in self.model.named_parameters() if not name.startswith("vae.")}

    def __init__(self, config: BagelConfig, context: BuildContext) -> None:
        """Compose the BAGEL graph from numerical configuration and bound layers."""

        super().__init__(config)
        layer_config = context.layers["model"]
        self._parallel = layer_config.communicator
        self.model = _BagelGraph(config, layer_config=layer_config)
        llm = self.config.llm
        self.architecture = "BagelForConditionalGeneration"

        # Diffusion geometry preserves the checkpoint's latent rows and framing.
        self.generation = ImageDiffusion(
            latent_downsample=int(self.config.latent_downsample),
            prediction="velocity",
            prediction_dtype="bfloat16",
            schedule_direction=ScheduleDirection.DESCENDING,
            schedule_shift_domain=ScheduleShiftDomain.TIME,
            max_latent_tokens=int(self.config.latent_token_capacity),
            max_vae_grid_tokens=(
                int(self.config.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS
            ),
            marker_tokens=_BAGEL_IMAGE_MARKER_TOKENS,
            rope_advance=2,
            max_cfg_branches=3,
            latent_layout=LatentLayout.PATCH_TOKENS,
            latent_channels=int(self.config.latent_channel),
            latent_patch_size=int(self.config.latent_patch_size),
            positions=PositionLayout.TEMPORAL,
            text_unconditional=BranchSource.NEGATIVE_OR_START,
            image_unconditional=BranchSource.CONDITIONING,
            cfg_recipe=CfgRecipe.IMAGE_OVER_TEXT,
            timestep_shift=float(self.config.timestep_shift),
        )

        # Vision and VAE routes use independent raster bounds but one feature contract.
        self.image_processor = ImageProcessor(
            vit=TowerTransform(
                resize=StrideResize(
                    max_size=int(self.config.vit_image_size),
                    min_size=_BAGEL_VIT_MIN_SIZE,
                    stride=int(self.config.vit_patch_size),
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
                start_token_id=int(self.config.start_of_image_id),
                end_token_id=int(self.config.end_of_image_id),
            ),
        )

        # KV geometry describes this rank's mathematical attention partition.
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

        self.max_vit_grid_tokens = int(self.config.vit_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS
        self.vocab_size = int(llm.vocab_size)
        self.hidden_size = int(llm.hidden_size)
        self.text_max_tokens = max(
            int(llm.max_position_embeddings),
            int(self.config.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS,
        )
        self.text_topology = ("tp", "tower")
        self.text_attention_mode = AttentionMode.PACKED

    @property
    def text_backbone(self):
        return self.model.lm

    @property
    def vocabulary(self):
        return vocabulary_partition(self.vocab_size, self._parallel)

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def diffusion_pipeline(self):
        return self.model.lm.pipeline

    def forward_diffusion(
        self,
        batch: DiffusionBatch,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Evaluate framed latent sequences through the shared MoT backbone."""

        embeddings = None
        pipeline = self.diffusion_pipeline
        if pipeline.first:
            chunks = []
            for index, latent in enumerate(batch.latents["image"]):
                if batch.conditioning["image"][index] is not None:
                    raise TypeError("BAGEL denoise does not accept external feature conditioning")
                latent_tokens = batch.sequence_lengths[index] - _BAGEL_IMAGE_MARKER_TOKENS
                if latent_tokens < 1 or int(latent.shape[-2]) != latent_tokens:
                    raise ValueError("BAGEL latent does not match its image-token geometry")
                chunks.append(
                    self.model.embed_latents(
                        latent_tokens,
                        batch.positions[index],
                        latent,
                        batch.timesteps["image"][index],
                    )
                )
            embeddings = torch.cat(chunks, dim=0).to(torch.bfloat16)
        hidden = self.model.lm(embeddings, batch.attention)
        if not pipeline.last:
            return TensorOutput({"image": (None,) * batch.row_count})
        rows = hidden[: sum(batch.sequence_lengths)].split(batch.sequence_lengths)
        return TensorOutput(
            {
                "image": tuple(
                    self.model.velocity_from_hidden(row, count - _BAGEL_IMAGE_MARKER_TOKENS)
                    for row, count in zip(rows, batch.sequence_lengths, strict=True)
                )
            }
        )

    encoder_kinds: frozenset[EncodeKind] = frozenset({"vision", "latent"})

    @property
    def latent_encoder(self):
        return self.model.vae

    @property
    def image_decoder(self):
        return self.model.vae

    @property
    def vision_encoder(self) -> PatchEncoder:
        """Return the composed patch encoder without duplicating parameter ownership."""

        return self.model.vision
