"""BAGEL neural model definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..forward import (
    DecodeOutput,
    DecodeRow,
    EncodeKind,
    EncodeOutput,
    EncodeRow,
    FlowOutput,
    FlowRow,
    ForwardBatch,
    ForwardContext,
    ForwardOutput,
    NoFlowConditioning,
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
from ..nn import (
    LayerSpec,
    LinearBase,
    MLPConnector,
    ParallelLMHead,
    local_attention_head_count,
    local_kv_head_count,
)
from ..nn.decoder import MoTModel
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
from ..spec import (
    CacheSpec,
    FeatureInjectionSpec,
    FeatureLayout,
    FlowBranchSource,
    FlowConditioningKind,
    FlowSpec,
    ImageInputSpec,
    ImageTowerSpec,
    InputSpec,
    LatentLayout,
    MaterializationKind,
    ModelSpec,
    NoiseScaleSpec,
    OperationSpec,
    OperationStageCondition,
    OperationStagePurpose,
    OperationStageSpec,
    OperationType,
    PositionLayout,
    Rename,
    RouteOutputKind,
    RoutePlacement,
    RouteRowKind,
    RouteShape,
    RouteShapeGrouping,
    RouteSpec,
    Sidecar,
    Stack,
    StrideResizeSpec,
    UnmatchedWeightPolicy,
    WeightSpec,
)

__all__ = [
    "LLMConfig",
    "BagelConfig",
    "BagelForUnifiedGeneration",
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
    bos_token_id: int = 151644
    eos_token_id: int = 151645
    max_position_embeddings: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True, slots=True)
class BagelConfig:
    """Top-level BAGEL model configuration (LLM, ViT, VAE, and latent settings)."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    visual_gen: bool = True
    visual_und: bool = True
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
        """Construct the immutable declaration from loader-resolved config data."""

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
            bos_token_id=llm_raw.get("bos_token_id", 151644),
            eos_token_id=llm_raw.get("eos_token_id", 151645),
        )
        vae = raw.get("vae_config", {})
        vit = raw.get("vit_config", {})
        return cls(
            llm=llm,
            visual_gen=raw.get("visual_gen", True),
            visual_und=raw.get("visual_und", True),
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

    def __init__(self, cfg: BagelConfig, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.llm.hidden_size
        self.lm = MoTModel(cfg.llm, spec=layer_spec)
        self.lm_head = ParallelLMHead(
            hidden,
            cfg.llm.vocab_size,
            spec=layer_spec,
            bias=False,
        )
        self.vae2llm = LinearBase(cfg.patch_latent_dim, hidden, spec=layer_spec)
        self.llm2vae = LinearBase(hidden, cfg.patch_latent_dim, spec=layer_spec)
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
            spec=layer_spec,
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

    def embed_tokens(self, ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        return self.lm.embed_tokens(ids, context.mesh)

    def gen_segment_embeds(
        self,
        num_vae,
        vae_pos_ids,
        x_t,
        timestep,
        context: ForwardContext,
    ) -> torch.Tensor:
        """Marker/VAE-latent/timestep embeddings for one gen segment, ``[num_vae+2, hidden]``.

        Shared by graph denoise and image-commit paths so marker and latent
        embeddings follow one model contract.
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
        context: ForwardContext,
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
        context: ForwardContext,
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
            {"position_ids": packed_pos_ids, "cu_seqlens": cu_seqlens},
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


# Checkpoint tensor names map onto the ``_BagelGraph`` parameter tree through
# these ordered rules; a name outside them is not a graph target (the VAE loads
# from its sidecar file).
_BAGEL_RENAMES = (
    Rename("language_model.model.embed_tokens.weight", "lm.embed_tokens.weight", exact=True),
    Rename("language_model.model.norm.weight", "lm.norm.weight", exact=True),
    Rename("language_model.model.norm_moe_gen.weight", "lm.norm_moe_gen.weight", exact=True),
    Rename("language_model.lm_head.weight", "lm_head.weight", exact=True),
    Rename(
        "language_model.model.layers.",
        "lm.layers.",
        substitutions=((".self_attn.", "."),),
    ),
    Rename("vit_model.vision_model.embeddings.", "vit_model."),
    Rename(
        "vit_model.vision_model.encoder.",
        "vit_model.encoder.",
        substitutions=((".mlp.fc1.", ".mlp.0."), (".mlp.fc2.", ".mlp.2.")),
    ),
    Rename(
        "vit_model.vision_model.post_layernorm.",
        "vit_model.encoder.post_layernorm.",
    ),
    Rename("connector.", "connector."),
    Rename("vit_pos_embed.", "vit_pos_embed."),
    Rename("vae2llm.", "vae2llm."),
    Rename("llm2vae.", "llm2vae."),
    Rename("time_embedder.", "time_embedder."),
    Rename("latent_pos_embed.", "latent_pos_embed."),
)

# Stack declarations bind exact parameter-path segments and preserve the target
# module's packed projection order.
_BAGEL_STACKED = (
    Stack("qkv_proj_moe_gen", "q_proj_moe_gen", "q"),
    Stack("qkv_proj_moe_gen", "k_proj_moe_gen", "k"),
    Stack("qkv_proj_moe_gen", "v_proj_moe_gen", "v"),
    Stack("qkv_proj", "q_proj", "q"),
    Stack("qkv_proj", "k_proj", "k"),
    Stack("qkv_proj", "v_proj", "v"),
    Stack("gate_up_proj", "gate_proj", 0),
    Stack("gate_up_proj", "up_proj", 1),
)


class BagelForUnifiedGeneration(nn.Module):
    """Stateless BAGEL neural graph for the declared MoT, ViT, and VAE routes."""

    weight_spec = WeightSpec(
        files=("ema.safetensors", "model.safetensors"),
        transforms=(*_BAGEL_RENAMES, *_BAGEL_STACKED),
        adapter_renames=(Rename("base_model.model.", ""),),
        unmatched=UnmatchedWeightPolicy.SKIP,
        sidecars=(Sidecar(file="ae.safetensors", module="vae", optional_substrings=("reg",)),),
    )

    def __init__(
        self,
        config: BagelConfig,
        *,
        layer_spec: LayerSpec,
        graph: _BagelGraph | None = None,
    ) -> None:
        super().__init__()
        if graph is not None and graph.cfg != config:
            raise ValueError("BAGEL graph and root must use the same immutable configuration")
        self.cfg = config
        self._parallel = layer_spec.parallel
        self.model = graph if graph is not None else _BagelGraph(config, layer_spec=layer_spec)
        self.spec = self._build_spec()

    def _build_spec(self) -> ModelSpec:
        llm = self.cfg.llm
        flow = FlowSpec(
            latent_downsample=int(self.cfg.latent_downsample),
            prediction="velocity",
            prediction_dtype="bfloat16",
            schedule_direction=ScheduleDirection.DESCENDING.value,
            schedule_shift_domain=ScheduleShiftDomain.TIME.value,
            max_latent_tokens=int(self.cfg.latent_token_capacity),
            max_vae_grid_tokens=(int(self.cfg.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS),
            commit_marker_tokens=_BAGEL_IMAGE_MARKER_TOKENS,
            rope_advance=2,
            max_cfg_branches=3,
            latent_layout=LatentLayout.PATCH_TOKENS,
            latent_channels=int(self.cfg.latent_channel),
            latent_patch_size=int(self.cfg.latent_patch_size),
            positions=PositionLayout.TEMPORAL,
            conditioning=FlowConditioningKind.NONE,
            materialization=MaterializationKind.DECODE_ROUTE,
            noise_scale=NoiseScaleSpec(),
            text_unconditional=FlowBranchSource.NEGATIVE_OR_START,
            image_unconditional=FlowBranchSource.CONDITIONING,
            cfg_recipe=CfgRecipe.IMAGE_OVER_TEXT.value,
            timestep_shift=float(self.cfg.timestep_shift),
        )
        return ModelSpec(
            architecture="BagelForUnifiedGeneration",
            routes=(
                # Token and flow rows share the MoT backbone in one forward.
                RouteSpec(
                    name="mot",
                    row_kinds=(RouteRowKind.TOKEN, RouteRowKind.FLOW),
                    output_kinds=(RouteOutputKind.TOKEN, RouteOutputKind.FLOW),
                    mixed_combinations=((RouteRowKind.TOKEN, RouteRowKind.FLOW),),
                    dtype="bfloat16",
                    placement=RoutePlacement.PRIMARY,
                    topology_axes=("tp",),
                    shape=RouteShape(
                        max_tokens_per_row=max(
                            int(llm.max_position_embeddings),
                            int(self.cfg.latent_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS,
                        ),
                        token_multiple=1,
                    ),
                    graph_eligible=True,
                ),
                RouteSpec(
                    name="vae",
                    row_kinds=(RouteRowKind.ENCODE, RouteRowKind.DECODE),
                    output_kinds=(RouteOutputKind.ENCODE, RouteOutputKind.DECODE),
                    mixed_combinations=(),
                    dtype="bfloat16",
                    placement=RoutePlacement.GENERATION,
                    topology_axes=("tp",),
                    shape=RouteShape(
                        max_tokens_per_row=int(self.cfg.latent_token_capacity),
                        token_multiple=1,
                        grouping=RouteShapeGrouping.IMAGE_GEOMETRY,
                    ),
                    graph_eligible=False,
                ),
                RouteSpec(
                    name="vit",
                    row_kinds=(RouteRowKind.ENCODE,),
                    output_kinds=(RouteOutputKind.ENCODE,),
                    mixed_combinations=(),
                    dtype="bfloat16",
                    placement=RoutePlacement.PRIMARY,
                    topology_axes=("tp",),
                    shape=RouteShape(
                        max_tokens_per_row=int(self.cfg.vit_token_capacity),
                        token_multiple=1,
                        grouping=RouteShapeGrouping.INPUT_SHAPE,
                    ),
                    graph_eligible=False,
                ),
            ),
            operations=(
                OperationSpec(
                    OperationType.SEQUENCE_EXTEND, (OperationStageSpec("mot", RouteRowKind.TOKEN),)
                ),
                OperationSpec(
                    OperationType.SEQUENCE_DECODE, (OperationStageSpec("mot", RouteRowKind.TOKEN),)
                ),
                OperationSpec(
                    OperationType.SEQUENCE_VERIFY, (OperationStageSpec("mot", RouteRowKind.TOKEN),)
                ),
                OperationSpec(OperationType.FLOW, (OperationStageSpec("mot", RouteRowKind.FLOW),)),
                OperationSpec(
                    OperationType.ENCODE_VISION,
                    (
                        OperationStageSpec("vit", RouteRowKind.ENCODE),
                        OperationStageSpec("mot", RouteRowKind.TOKEN, OperationStagePurpose.STATE),
                    ),
                ),
                OperationSpec(
                    OperationType.ENCODE_LATENT,
                    (
                        OperationStageSpec("vae", RouteRowKind.ENCODE),
                        OperationStageSpec("mot", RouteRowKind.FLOW, OperationStagePurpose.STATE),
                    ),
                ),
                OperationSpec(
                    OperationType.MATERIALIZE_IMAGE,
                    (
                        OperationStageSpec("vae", RouteRowKind.DECODE),
                        OperationStageSpec(
                            "mot",
                            RouteRowKind.FLOW,
                            OperationStagePurpose.STATE,
                            OperationStageCondition.RETAIN_IMAGE,
                        ),
                    ),
                ),
                OperationSpec(OperationType.TRANSFER_PRODUCT),
                OperationSpec(
                    OperationType.TRANSFER_KV,
                    (OperationStageSpec("mot", RouteRowKind.FLOW),),
                ),
            ),
            weights=self.weight_spec,
            # The host tokenizes for BAGEL; the worker holds no tokenizer.
            inputs=InputSpec(
                requires_worker_tokenizer=False,
                images=ImageInputSpec(
                    vit=ImageTowerSpec(
                        resize=StrideResizeSpec(
                            max_size=int(self.cfg.vit_image_size),
                            min_size=_BAGEL_VIT_MIN_SIZE,
                            stride=int(self.cfg.vit_patch_size),
                            max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
                        ),
                        normalization="signed_unit",
                    ),
                    vae=ImageTowerSpec(
                        resize=StrideResizeSpec(
                            max_size=_BAGEL_VAE_MAX_SIZE,
                            min_size=_BAGEL_VAE_MIN_SIZE,
                            stride=_BAGEL_VAE_STRIDE,
                            max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
                        ),
                        normalization="signed_unit",
                    ),
                    feature_injection=FeatureInjectionSpec(
                        layout=FeatureLayout.FRAMED,
                        positions=PositionLayout.TEMPORAL,
                        start_token_id=int(self.cfg.start_of_image_id),
                        end_token_id=int(self.cfg.end_of_image_id),
                    ),
                ),
                encoder_cache_budget=256,
                max_vit_grid_tokens=(int(self.cfg.vit_token_capacity) + _BAGEL_IMAGE_MARKER_TOKENS),
            ),
            cache=CacheSpec(
                num_layers=int(llm.num_hidden_layers),
                num_attention_heads=local_attention_head_count(
                    int(llm.num_attention_heads),
                    parallel=self._parallel,
                ),
                num_kv_heads=local_kv_head_count(
                    int(llm.num_key_value_heads),
                    parallel=self._parallel,
                ),
                head_dim=int(llm.head_dim),
                dtype="bfloat16",
                store_dtype="bfloat16",
                position_layout=PositionLayout.TEMPORAL,
            ),
            flow=flow,
        )

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        if batch.route == "mot":
            return self._forward_mot(batch)
        if batch.route == "vit":
            return self._forward_vit(batch)
        if batch.route == "vae":
            return self._forward_vae(batch)
        raise ValueError(f"BAGEL received unknown route {batch.route!s}")

    def _forward_mot(self, batch: ForwardBatch) -> ForwardOutput:
        rows: list[TokenRow | FlowRow] = []
        for row in batch.rows:
            if not isinstance(row, (TokenRow, FlowRow)):
                raise TypeError("BAGEL mot route accepts TokenRow and FlowRow values")
            rows.append(row)
        chunks: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for row in rows:
            if isinstance(row, TokenRow):
                chunk = self._token_embeddings(row, batch.context)
            else:
                if not isinstance(row.conditioning, NoFlowConditioning):
                    raise TypeError("BAGEL flow rows do not accept external feature conditioning")
                latent_tokens = int(row.image_tokens) - _BAGEL_IMAGE_MARKER_TOKENS
                if latent_tokens < 1 or int(row.latent.shape[-2]) != latent_tokens:
                    raise ValueError("BAGEL flow latent does not match its image-token geometry")
                chunk = self.model.gen_segment_embeds(
                    latent_tokens,
                    row.positions,
                    row.latent,
                    row.timestep,
                    batch.context,
                )
            chunk = chunk.reshape(-1, chunk.shape[-1]).to(torch.bfloat16)
            chunks.append(chunk)
            spans.append((offset, offset + int(chunk.shape[0])))
            offset += int(chunk.shape[0])
        hidden = self.model.lm(torch.cat(chunks, dim=0), batch.context)
        outputs: list[TokenOutput | FlowOutput] = []
        for row, (begin, end) in zip(rows, spans, strict=True):
            row_hidden = hidden[begin:end]
            if isinstance(row, TokenRow):
                value: TokenHidden | TokenLogits
                if row.selection is TokenSelection.HIDDEN:
                    value = TokenHidden(row_hidden)
                elif row.selection is TokenSelection.ALL_LOGITS:
                    value = TokenLogits(self.model.logits(row_hidden, batch.context))
                else:
                    value = TokenLogits(self.model.logits(row_hidden[-1:], batch.context))
                outputs.append(TokenOutput(row.row_id, row.output_slot, value))
            else:
                prediction = self.model.velocity_from_hidden(
                    row_hidden,
                    int(row.image_tokens) - _BAGEL_IMAGE_MARKER_TOKENS,
                )
                outputs.append(FlowOutput(row.row_id, row.output_slot, prediction))
        return ForwardOutput(tuple(outputs))

    def _token_embeddings(
        self,
        row: TokenRow,
        context: ForwardContext,
    ) -> torch.Tensor:
        if isinstance(row.inputs, TokenIds):
            return self.model.embed_tokens(row.inputs.values.reshape(-1), context)
        if isinstance(row.inputs, TokenEmbeddings):
            return row.inputs.values.reshape(-1, row.inputs.values.shape[-1])
        if not isinstance(row.inputs, TokenSegments):
            raise TypeError("BAGEL token row has an unknown input variant")
        return torch.cat(
            tuple(
                self.model.embed_tokens(segment.values.reshape(-1), context)
                if isinstance(segment, TokenIds)
                else segment.values.reshape(-1, segment.values.shape[-1])
                for segment in row.inputs.values
            ),
            dim=0,
        )

    def _forward_vit(self, batch: ForwardBatch) -> ForwardOutput:
        rows = tuple(row for row in batch.rows if isinstance(row, EncodeRow))
        if len(rows) != len(batch.rows) or any(
            row.kind is not EncodeKind.VISION or not isinstance(row.inputs, TowerInput)
            for row in rows
        ):
            raise TypeError("BAGEL vit route requires tower vision encode rows")
        pixels = torch.stack(tuple(row.inputs.pixels for row in rows), dim=0)
        features = self.model.vit_encode_batch(pixels, batch.context)
        return ForwardOutput(
            tuple(
                EncodeOutput(row.row_id, row.output_slot, features[index])
                for index, row in enumerate(rows)
            )
        )

    def _forward_vae(self, batch: ForwardBatch) -> ForwardOutput:
        first = batch.rows[0]
        if isinstance(first, EncodeRow):
            encode_rows = tuple(row for row in batch.rows if isinstance(row, EncodeRow))
            if len(encode_rows) != len(batch.rows) or any(
                row.kind is not EncodeKind.LATENT or not isinstance(row.inputs, TowerInput)
                for row in encode_rows
            ):
                raise TypeError("BAGEL vae encode route requires latent tower rows")
            pixels = torch.stack(tuple(row.inputs.pixels for row in encode_rows), dim=0)
            latents, _positions, _shape = self.model.vae_encode_clean_batch(pixels)
            return ForwardOutput(
                tuple(
                    EncodeOutput(row.row_id, row.output_slot, latents[index])
                    for index, row in enumerate(encode_rows)
                )
            )
        decode_rows = tuple(row for row in batch.rows if isinstance(row, DecodeRow))
        if len(decode_rows) != len(batch.rows):
            raise TypeError("BAGEL vae route cannot mix encode and decode rows")
        geometry = {(row.image_height, row.image_width) for row in decode_rows}
        if len(geometry) != 1:
            raise ValueError("BAGEL vae route requires one image geometry")
        height, width = next(iter(geometry))
        decoded = self.model.vae_decode_batch(
            torch.stack(tuple(row.latent for row in decode_rows), dim=0),
            height,
            width,
        )
        return ForwardOutput(
            tuple(
                DecodeOutput(row.row_id, row.output_slot, decoded[index])
                for index, row in enumerate(decode_rows)
            )
        )
