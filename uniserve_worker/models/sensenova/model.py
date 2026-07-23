"""SenseNova-U1 immutable declarations and neural equations."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...forward import (
    EncodeKind,
    EncodeOutput,
    EncodeRow,
    FlowOutput,
    FlowPatches,
    FlowRow,
    ForwardBatch,
    ForwardContext,
    ForwardOutput,
    PackedAttentionPlan,
    PatchInput,
    TokenEmbeddings,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSegments,
    TokenSelection,
)
from ...nn.attention import RadixAttention
from ...nn.decoder.qwen import Qwen3MLP
from ...nn.diffusion import (
    ConvDecoder,
    FlowMatchingHead,
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from ...nn.diffusion.cfg import CfgRecipe
from ...nn.layer import LayerSpec
from ...nn.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
    local_attention_head_count,
    local_kv_head_count,
)
from ...nn.norm import RMSNorm
from ...nn.placement import WeightMode, set_tower_coord
from ...nn.rope import HFRotaryEmbedding, RotaryEmbedding, get_rope, qk_norm_rope
from ...nn.vision import NeoVitConfig, NeoVitEncoder
from ...nn.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ...spec import (
    CacheSpec,
    FeatureInjectionSpec,
    FeatureLayout,
    FlowBranchSource,
    FlowConditioningKind,
    FlowPromptSpec,
    FlowSpec,
    ImageInputSpec,
    ImagePatchSpec,
    InputSpec,
    LatentLayout,
    MaterializationKind,
    ModelSpec,
    NoiseScaleMode,
    NoiseScaleSpec,
    OperationSpec,
    OperationStageCondition,
    OperationStagePurpose,
    OperationStageSpec,
    OperationType,
    PositionLayout,
    RouteOutputKind,
    RoutePlacement,
    RouteRowKind,
    RouteShape,
    RouteShapeGrouping,
    RouteSpec,
    Stack,
    TowerSplit,
    WeightSpec,
)
from .config import NeoChatConfig, NeoLlmConfig, NeoVisionConfig

__all__ = ["NEOChatModel"]

_MODEL_CODE_VERSION = "0.1.0"
_TEXT_COORDINATE = 0
_FLOW_COORDINATE = 1
_MAX_VISION_TOKENS = 70 * 70
_MAX_CFG_BRANCHES = 3
_GENERATION_EPSILON = 0.02

_FLOW_SYSTEM_MESSAGE = (
    "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
    "You support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n"
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language."
)

_STACKED_WEIGHTS = (
    Stack("qkv_proj", "q_proj", "q"),
    Stack("qkv_proj", "k_proj", "k"),
    Stack("qkv_proj", "v_proj", "v"),
    Stack("qkv_proj_mot_gen", "q_proj_mot_gen", "q"),
    Stack("qkv_proj_mot_gen", "k_proj_mot_gen", "k"),
    Stack("qkv_proj_mot_gen", "v_proj_mot_gen", "v"),
    Stack("gate_up_proj", "gate_proj", 0),
    Stack("gate_up_proj", "up_proj", 1),
)

_TOWER_SPLIT = TowerSplit(
    generation_prefixes=("fm_modules.",),
    generation_infixes=("_mot_gen.",),
)


def _module_tensor(
    module: nn.Module,
    value: torch.Tensor,
    *,
    context: ForwardContext,
    coordinate: int,
    target: torch.device,
    call: Callable[[nn.Module, torch.Tensor, ForwardContext], torch.Tensor],
) -> torch.Tensor:
    staged = context.mesh.dispatch(value, "tower", coordinate)
    result = call(module, staged, context)
    if not isinstance(result, torch.Tensor):
        raise TypeError("SenseNova neural sublayer must return a tensor")
    return context.mesh.combine(result, "tower", coordinate, target)


def _route_tensor(
    value: torch.Tensor,
    *,
    plan: PackedAttentionPlan,
    text_module: nn.Module,
    flow_module: nn.Module,
    context: ForwardContext,
    call: Callable[[nn.Module, torch.Tensor, ForwardContext], torch.Tensor],
) -> torch.Tensor:
    target = value.device
    if plan.has_text and plan.has_flow:
        result = _module_tensor(
            flow_module,
            value,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=call,
        )
        text = _module_tensor(
            text_module,
            value.index_select(0, plan.text_indices),
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=target,
            call=call,
        )
        result.index_copy_(0, plan.text_indices, text)
        return result
    if plan.has_flow:
        return _module_tensor(
            flow_module,
            value,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=call,
        )
    if plan.has_text:
        return _module_tensor(
            text_module,
            value,
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=target,
            call=call,
        )
    raise ValueError("SenseNova packed route contains no neural tokens")


def _plain_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardContext,
) -> torch.Tensor:
    del context
    return cast(torch.Tensor, module(value))


def _parallel_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardContext,
) -> torch.Tensor:
    return cast(torch.Tensor, module(value, context.mesh))


@dataclass(frozen=True, slots=True)
class _PackedRope:
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def select(self, indices: torch.Tensor) -> _PackedRope:
        return _PackedRope(
            (
                self.cos[0].index_select(0, indices),
                self.cos[1].index_select(0, indices),
                self.cos[2].index_select(0, indices),
            ),
            (
                self.sin[0].index_select(0, indices),
                self.sin[1].index_select(0, indices),
                self.sin[2].index_select(0, indices),
            ),
        )


class _VisionModel(nn.Module):
    """NEO vision tower returning its raw language-width features."""

    def __init__(self, config: NeoVisionConfig) -> None:
        super().__init__()
        self.embeddings = NeoVitEncoder(
            NeoVitConfig(
                hidden_size=int(getattr(config, "hidden_size")),
                llm_hidden_size=int(getattr(config, "llm_hidden_size")),
                downsample_ratio=float(getattr(config, "downsample_ratio")),
                patch_size=int(getattr(config, "patch_size")),
                num_channels=int(getattr(config, "num_channels")),
                rope_theta_vision=float(getattr(config, "rope_theta_vision")),
            )
        )

    def forward(self, pixels: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return self.embeddings(pixels, grid)


class _SenseAttention(nn.Module):
    """SenseNova dual-expert QKV projection over one explicit attention plan."""

    def __init__(self, config: NeoLlmConfig, layer: int, *, spec: LayerSpec) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        total_heads = int(getattr(config, "num_attention_heads"))
        total_kv_heads = int(getattr(config, "num_key_value_heads"))
        self.head_dim = int(getattr(config, "head_dim", hidden_size // total_heads))
        self.scaling = self.head_dim**-0.5
        bias = bool(getattr(config, "attention_bias"))
        epsilon = float(getattr(config, "rms_norm_eps"))
        query_width = total_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_heads,
            total_kv_heads,
            spec=spec,
            bias=bias,
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_heads,
            total_kv_heads,
            spec=spec,
            bias=bias,
        )
        self.num_heads = int(self.qkv_proj.output_sizes[0]) // self.head_dim
        self.num_kv_heads = int(self.qkv_proj.output_sizes[1]) // self.head_dim
        if self.num_heads < 1 or self.num_kv_heads < 1:
            raise ValueError("SenseNova local attention geometry is invalid")
        self.attention = RadixAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            layer_id=layer,
        )
        self.o_proj = RowParallelLinear(query_width, hidden_size, spec=spec, bias=bias)
        self.o_proj_mot_gen = RowParallelLinear(
            query_width,
            hidden_size,
            spec=spec,
            bias=bias,
        )

        half = self.head_dim // 2
        self.q_norm = RMSNorm(half, eps=epsilon)
        self.q_norm_mot_gen = RMSNorm(half, eps=epsilon)
        self.q_norm_hw = RMSNorm(half, eps=epsilon)
        self.q_norm_hw_mot_gen = RMSNorm(half, eps=epsilon)
        self.k_norm = RMSNorm(half, eps=epsilon)
        self.k_norm_mot_gen = RMSNorm(half, eps=epsilon)
        self.k_norm_hw = RMSNorm(half, eps=epsilon)
        self.k_norm_hw_mot_gen = RMSNorm(half, eps=epsilon)

        temporal_config = copy.copy(config)
        temporal_config.head_dim = half
        self.rotary_emb = get_rope(config=temporal_config, keep_freq_range=True)
        spatial_config = copy.copy(config)
        spatial_config.head_dim = self.head_dim // 4
        spatial_config.rope_theta = getattr(config, "rope_theta_hw")
        spatial_config.max_position_embeddings = getattr(
            config,
            "max_position_embeddings_hw",
        )
        self.rotary_emb_hw = get_rope(config=spatial_config, keep_freq_range=True)

        for module in (
            self.qkv_proj_mot_gen,
            self.o_proj_mot_gen,
            self.q_norm_mot_gen,
            self.q_norm_hw_mot_gen,
            self.k_norm_mot_gen,
            self.k_norm_hw_mot_gen,
        ):
            set_tower_coord(module, _FLOW_COORDINATE)

    def rope(self, indexes: torch.Tensor) -> _PackedRope:
        if indexes.ndim != 2 or tuple(indexes.shape[:1]) != (3,):
            raise ValueError("SenseNova positions must have shape [3, tokens]")
        target = indexes.device

        def frequencies(
            module: RotaryEmbedding | HFRotaryEmbedding,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            frequency = cast(torch.Tensor, getattr(module, "inv_freq"))
            local_positions = positions.to(frequency.device)
            cosine, sine = module.cos_sin_1d(local_positions)
            return cosine.to(target), sine.to(target)

        cos_t, sin_t = frequencies(self.rotary_emb, indexes[0])
        cos_h, sin_h = frequencies(self.rotary_emb_hw, indexes[1])
        cos_w, sin_w = frequencies(self.rotary_emb_hw, indexes[2])
        return _PackedRope((cos_t, cos_h, cos_w), (sin_t, sin_h, sin_w))

    def _project(
        self,
        hidden: torch.Tensor,
        rope: _PackedRope,
        *,
        generation: bool,
        context: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coordinate = _FLOW_COORDINATE if generation else _TEXT_COORDINATE
        target = hidden.device
        staged = context.mesh.dispatch(hidden, "tower", coordinate)
        qkv_module = self.qkv_proj_mot_gen if generation else self.qkv_proj
        split_sizes = tuple(int(size) for size in qkv_module.output_sizes)
        if generation and int(staged.shape[0]) == 1:
            query_weight, key_weight, value_weight = qkv_module.weight.split(
                split_sizes,
                dim=0,
            )
            if qkv_module.bias is None:
                query_bias = key_bias = value_bias = None
            else:
                query_bias, key_bias, value_bias = qkv_module.bias.split(
                    split_sizes,
                    dim=0,
                )
            query_flat = F.linear(staged, query_weight, query_bias)
            key_flat = F.linear(staged, key_weight, key_bias)
            value_flat = F.linear(staged, value_weight, value_bias)
        else:
            projected = qkv_module(staged)
            query_flat, key_flat, value_flat = projected.split(split_sizes, dim=-1)

        query = query_flat.view(-1, self.num_heads, self.head_dim)
        key = key_flat.view(-1, self.num_kv_heads, self.head_dim)
        value = value_flat.view(-1, self.num_kv_heads, self.head_dim)
        query_norm = self.q_norm_mot_gen if generation else self.q_norm
        query_norm_hw = self.q_norm_hw_mot_gen if generation else self.q_norm_hw
        key_norm = self.k_norm_mot_gen if generation else self.k_norm
        key_norm_hw = self.k_norm_hw_mot_gen if generation else self.k_norm_hw
        local_cos = tuple(context.mesh.dispatch(item, "tower", coordinate) for item in rope.cos)
        local_sin = tuple(context.mesh.dispatch(item, "tower", coordinate) for item in rope.sin)
        query, key = qk_norm_rope(
            query,
            key,
            (query_norm.weight, query_norm_hw.weight, query_norm_hw.weight),
            (key_norm.weight, key_norm_hw.weight, key_norm_hw.weight),
            local_cos,
            local_sin,
            query_norm.eps,
            axis_dims=(self.head_dim // 2, self.head_dim // 4, self.head_dim // 4),
        )
        return (
            context.mesh.combine(query, "tower", coordinate, target),
            context.mesh.combine(key, "tower", coordinate, target),
            context.mesh.combine(value, "tower", coordinate, target),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        context: ForwardContext,
        plan: PackedAttentionPlan,
        rope: _PackedRope,
    ) -> torch.Tensor:
        if plan.has_text and plan.has_flow:
            query, key, value = self._project(
                hidden,
                rope,
                generation=True,
                context=context,
            )
            text_rope = rope.select(plan.text_indices)
            text_query, text_key, text_value = self._project(
                hidden.index_select(0, plan.text_indices),
                text_rope,
                generation=False,
                context=context,
            )
            query.index_copy_(0, plan.text_indices, text_query)
            key.index_copy_(0, plan.text_indices, text_key)
            value.index_copy_(0, plan.text_indices, text_value)
        else:
            query, key, value = self._project(
                hidden,
                rope,
                generation=plan.has_flow,
                context=context,
            )
        attended = self.attention(
            query,
            key,
            value,
            context,
            causal=False,
            scale=self.scaling,
        ).reshape(hidden.shape[0], -1)
        return _route_tensor(
            attended,
            plan=plan,
            text_module=self.o_proj,
            flow_module=self.o_proj_mot_gen,
            context=context,
            call=_parallel_call,
        )


class _SenseLayer(nn.Module):
    def __init__(self, config: NeoLlmConfig, layer: int, *, spec: LayerSpec) -> None:
        super().__init__()
        hidden = int(getattr(config, "hidden_size"))
        epsilon = float(getattr(config, "rms_norm_eps"))
        self.self_attn = _SenseAttention(config, layer, spec=spec)
        self.mlp = Qwen3MLP(
            config,
            spec=spec,
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.mlp_mot_gen = Qwen3MLP(
            config,
            spec=spec,
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.input_layernorm = RMSNorm(hidden, eps=epsilon)
        self.input_layernorm_mot_gen = RMSNorm(hidden, eps=epsilon)
        self.post_attention_layernorm = RMSNorm(hidden, eps=epsilon)
        self.post_attention_layernorm_mot_gen = RMSNorm(hidden, eps=epsilon)
        for module in (
            self.mlp_mot_gen,
            self.input_layernorm_mot_gen,
            self.post_attention_layernorm_mot_gen,
        ):
            set_tower_coord(module, _FLOW_COORDINATE)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        context: ForwardContext,
        plan: PackedAttentionPlan,
        rope: _PackedRope,
    ) -> torch.Tensor:
        normalized = _route_tensor(
            hidden,
            plan=plan,
            text_module=self.input_layernorm,
            flow_module=self.input_layernorm_mot_gen,
            context=context,
            call=_plain_call,
        )
        hidden = hidden + self.self_attn(
            normalized,
            context=context,
            plan=plan,
            rope=rope,
        )
        normalized = _route_tensor(
            hidden,
            plan=plan,
            text_module=self.post_attention_layernorm,
            flow_module=self.post_attention_layernorm_mot_gen,
            context=context,
            call=_plain_call,
        )
        feed_forward = _route_tensor(
            normalized,
            plan=plan,
            text_module=self.mlp,
            flow_module=self.mlp_mot_gen,
            context=context,
            call=_parallel_call,
        )
        return hidden + feed_forward


class _SenseDecoder(nn.Module):
    """One packed text/flow decoder with no serving state."""

    def __init__(self, config: NeoLlmConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        hidden = int(getattr(config, "hidden_size"))
        self.embed_tokens = VocabParallelEmbedding(
            int(getattr(config, "vocab_size")),
            hidden,
            int(getattr(config, "pad_token_id")),
            spec=spec,
        )
        self.layers = nn.ModuleList(
            _SenseLayer(config, index, spec=spec)
            for index in range(int(getattr(config, "num_hidden_layers")))
        )
        epsilon = float(getattr(config, "rms_norm_eps"))
        self.norm = RMSNorm(hidden, eps=epsilon)
        self.norm_mot_gen = RMSNorm(hidden, eps=epsilon)
        set_tower_coord(self.norm_mot_gen, _FLOW_COORDINATE)

    def forward(
        self,
        inputs: torch.Tensor,
        context: ForwardContext,
    ) -> torch.Tensor:
        plan = context.attention
        if not isinstance(plan, PackedAttentionPlan):
            raise ValueError("SenseNova decoder requires a packed attention plan")
        if inputs.ndim != 2:
            raise ValueError("SenseNova decoder inputs must have shape [tokens, hidden]")
        token_count = int(inputs.shape[0])
        if tuple(plan.route_indicators.shape) != (token_count,):
            raise ValueError("SenseNova route indicators must align with decoder inputs")
        if tuple(plan.indexes.shape) != (3, token_count):
            raise ValueError("SenseNova positions must have shape [3, tokens]")
        if plan.text_indices.ndim != 1:
            raise ValueError("SenseNova text indices must be one-dimensional")
        if plan.has_text != (int(plan.text_indices.numel()) > 0):
            raise ValueError("SenseNova text presence does not match its static indices")
        if not plan.has_text and not plan.has_flow:
            raise ValueError("SenseNova decoder plan contains no tokens")
        if not self.layers:
            raise ValueError("SenseNova decoder requires at least one layer")

        first = cast(_SenseLayer, self.layers[0])
        rope = first.self_attn.rope(plan.indexes)
        hidden = inputs
        for layer_module in self.layers:
            layer = cast(_SenseLayer, layer_module)
            hidden = layer(hidden, context=context, plan=plan, rope=rope)
        return _route_tensor(
            hidden,
            plan=plan,
            text_module=self.norm,
            flow_module=self.norm_mot_gen,
            context=context,
            call=_plain_call,
        )


class _LanguageModel(nn.Module):
    def __init__(self, config: NeoLlmConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        self.model = _SenseDecoder(config, spec=spec)
        self.lm_head = ParallelLMHead(
            int(getattr(config, "hidden_size")),
            int(getattr(config, "vocab_size")),
            spec=spec,
            bias=False,
        )


class NEOChatModel(nn.Module):
    """Concrete stateless SenseNova model for mixed text, flow, and vision rows."""

    weight_spec = WeightSpec(
        transforms=_STACKED_WEIGHTS,
        tower=_TOWER_SPLIT,
    )

    def __init__(self, config: NeoChatConfig, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        vision = config.vision_config
        hidden = int(config.llm_config.hidden_size)
        self.vision_model = _VisionModel(vision)
        self._parallel = layer_spec.parallel
        self.language_model = _LanguageModel(config.llm_config, spec=layer_spec)
        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": _VisionModel(vision),
                "timestep_embedder": TimestepEmbedder(hidden),
                "fm_head": self._flow_head(config, hidden, layer_spec),
            }
        )
        self._patch_size = int(vision.patch_size)
        self._downsample_ratio = float(config.downsample_ratio)
        self._use_deep_head = bool(getattr(config, "fm_head_layers", 2) > 2)
        self._use_pixel_head = bool(getattr(config, "use_pixel_head", False))
        if self._use_pixel_head:
            self.fm_modules["fm_head"] = ConvDecoder(hidden)
        self._add_noise_embedding = bool(getattr(config, "add_noise_scale_embedding", False))
        self._noise_scale_max = float(getattr(config, "noise_scale_max_value", 1.0))
        if self._add_noise_embedding:
            self.fm_modules["noise_scale_embedder"] = TimestepEmbedder(hidden)
        set_tower_coord(self.fm_modules, _FLOW_COORDINATE)
        self.spec = self._build_spec(config)

    @staticmethod
    def _flow_head(
        config: NeoChatConfig,
        hidden: int,
        layer_spec: LayerSpec,
    ) -> nn.Module:
        merge = int(1 / float(config.downsample_ratio))
        output_dim = 3 * (int(config.vision_config.patch_size) * merge) ** 2
        if int(getattr(config, "fm_head_layers", 2)) > 2:
            return FlowMatchingHead(
                hidden,
                output_dim,
                spec=layer_spec,
                dim=int(getattr(config, "fm_head_dim")),
                layers=int(getattr(config, "fm_head_layers")),
                mlp_ratio=float(getattr(config, "fm_head_mlp_ratio")),
            )
        return nn.Sequential(
            LinearBase(hidden, 4096, spec=layer_spec, bias=True),
            nn.GELU(),
            LinearBase(4096, output_dim, spec=layer_spec, bias=True),
        )

    def _build_spec(self, config: NeoChatConfig) -> ModelSpec:
        llm = config.llm_config
        vision = config.vision_config
        max_text = int(getattr(llm, "max_position_embeddings", 32768))
        max_image = max(1, int(getattr(config, "max_image_seq_len", 4096)))
        latent_downsample = int(
            int(vision.patch_size) * round(1 / float(config.downsample_ratio))
        )
        flow = FlowSpec(
            latent_downsample=latent_downsample,
            prediction="velocity",
            prediction_dtype="float32",
            schedule_direction=ScheduleDirection.ASCENDING.value,
            schedule_shift_domain=ScheduleShiftDomain.SIGMA.value,
            max_latent_tokens=max_image,
            max_vae_grid_tokens=max_image,
            commit_marker_tokens=2,
            rope_advance=2,
            max_cfg_branches=_MAX_CFG_BRANCHES,
            latent_layout=LatentLayout.IMAGE_NCHW,
            latent_channels=3,
            latent_patch_size=latent_downsample,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            conditioning=FlowConditioningKind.IMAGE_PATCHES,
            materialization=MaterializationKind.RGB_LATENT,
            noise_scale=NoiseScaleSpec(
                value=float(getattr(config, "noise_scale", 1.0)),
                mode=NoiseScaleMode(
                    str(
                        getattr(
                            getattr(config, "noise_scale_mode", "constant"),
                            "value",
                            getattr(config, "noise_scale_mode", "constant"),
                        )
                    )
                ),
                base_image_tokens=float(
                    getattr(config, "noise_scale_base_image_seq_len", 1.0)
                ),
                maximum=float(getattr(config, "noise_scale_max_value", 1.0)),
            ),
            text_unconditional=FlowBranchSource.NEGATIVE_OR_START,
            image_unconditional=FlowBranchSource.START,
            cfg_recipe=CfgRecipe.ADDITIVE_DELTAS.value,
        )
        return ModelSpec(
            architecture="NEOChatModel",
            routes=(
                RouteSpec(
                    name="mot",
                    row_kinds=(RouteRowKind.TOKEN, RouteRowKind.FLOW),
                    output_kinds=(RouteOutputKind.TOKEN, RouteOutputKind.FLOW),
                    mixed_combinations=((RouteRowKind.TOKEN, RouteRowKind.FLOW),),
                    dtype="bfloat16",
                    placement=RoutePlacement.MESH,
                    topology_axes=("tp", "tower"),
                    shape=RouteShape(
                        max_tokens_per_row=max(max_text, max_image),
                        token_multiple=1,
                    ),
                    graph_eligible=True,
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
                        max_tokens_per_row=_MAX_VISION_TOKENS,
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
                    OperationType.MATERIALIZE_IMAGE,
                    (
                        OperationStageSpec(
                            "vit",
                            RouteRowKind.ENCODE,
                            OperationStagePurpose.STATE,
                            OperationStageCondition.RETAIN_IMAGE,
                        ),
                        OperationStageSpec(
                            "mot",
                            RouteRowKind.TOKEN,
                            OperationStagePurpose.STATE,
                            OperationStageCondition.RETAIN_IMAGE,
                        ),
                    ),
                ),
                OperationSpec(OperationType.TRANSFER_PRODUCT),
                OperationSpec(
                    OperationType.TRANSFER_KV,
                    (
                        OperationStageSpec("vit", RouteRowKind.ENCODE, OperationStagePurpose.STATE),
                        OperationStageSpec("mot", RouteRowKind.TOKEN, OperationStagePurpose.STATE),
                    ),
                ),
            ),
            weights=self.weight_spec,
            inputs=InputSpec(
                requires_worker_tokenizer=True,
                flow_prompt=FlowPromptSpec(
                    user_prefix="<|im_start|>user\n",
                    user_suffix="<|im_end|>\n",
                    assistant_suffix="<|im_start|>assistant\n",
                    conditioned_append="<think>\n\n</think>\n\n<img>",
                    unconditional_append="<img>",
                    system_prefix="<|im_start|>system\n",
                    system_message=_FLOW_SYSTEM_MESSAGE,
                    system_suffix="<|im_end|>\n",
                ),
                images=ImageInputSpec(
                    vit=ImagePatchSpec(
                        patch_size=int(vision.patch_size),
                        downsample_ratio=float(vision.downsample_ratio),
                        min_pixels=512 * 512,
                        max_pixels=2048 * 2048,
                        multi_image_pixel_budget=4096 * 4096,
                        normalization="imagenet",
                    ),
                    staging_dtype="bfloat16",
                    feature_injection=FeatureInjectionSpec(
                        layout=FeatureLayout.DIRECT,
                        positions=PositionLayout.TEMPORAL_SPATIAL,
                        start_token="<img>",
                        end_token="</img>",
                    ),
                ),
                encoder_cache_budget=256,
                max_vit_grid_tokens=_MAX_VISION_TOKENS,
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
                position_layout=PositionLayout.TEMPORAL_SPATIAL,
            ),
            flow=flow,
        )

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        if batch.route == "mot":
            return self._forward_mot(batch)
        if batch.route == "vit":
            return self._forward_vision(batch)
        raise ValueError(f"SenseNova received unknown route {batch.route!s}")

    def _flow_embeddings(
        self,
        rows: tuple[FlowRow, ...],
        context: ForwardContext,
    ) -> dict[int, torch.Tensor]:
        patches = tuple(row.conditioning for row in rows)
        if any(not isinstance(value, FlowPatches) for value in patches):
            raise TypeError("SenseNova flow rows require patch conditioning")
        typed = cast(tuple[FlowPatches, ...], patches)
        target = rows[0].latent.device
        pixels = torch.cat(tuple(value.pixels for value in typed), dim=0)
        grids = torch.cat(tuple(value.grid for value in typed), dim=0)
        local_pixels = context.mesh.dispatch(pixels, "tower", _FLOW_COORDINATE)
        local_grids = context.mesh.dispatch(grids, "tower", _FLOW_COORDINATE)
        tower = self.fm_modules["vision_model_mot_gen"]
        feature_dtype = next(tower.parameters()).dtype
        features = tower(local_pixels.to(dtype=feature_dtype), local_grids)
        if not isinstance(features, torch.Tensor):
            raise TypeError("SenseNova flow vision tower must return a tensor")
        features = context.mesh.combine(
            features,
            "tower",
            _FLOW_COORDINATE,
            target,
        )
        expected = sum(int(row.image_tokens) for row in rows)
        if int(features.shape[0]) != expected:
            raise ValueError("SenseNova flow vision features do not match row geometry")

        timesteps = torch.cat(
            tuple(row.timestep.reshape(1).expand(int(row.image_tokens)) for row in rows),
            dim=0,
        )
        time_features = _module_tensor(
            self.fm_modules["timestep_embedder"],
            timesteps,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=_plain_call,
        )
        features = features + time_features
        if self._add_noise_embedding:
            noise = torch.cat(
                tuple(
                    value.noise_scale.reshape(1).expand(int(row.image_tokens))
                    for row, value in zip(rows, typed, strict=True)
                ),
                dim=0,
            )
            noise = noise / self._noise_scale_max
            features = features + _module_tensor(
                self.fm_modules["noise_scale_embedder"],
                noise,
                context=context,
                coordinate=_FLOW_COORDINATE,
                target=target,
                call=_plain_call,
            )

        result: dict[int, torch.Tensor] = {}
        offset = 0
        for row in rows:
            end = offset + int(row.image_tokens)
            result[row.row_id] = features[offset:end]
            offset = end
        return result

    def _forward_mot(self, batch: ForwardBatch) -> ForwardOutput:
        rows: list[TokenRow | FlowRow] = []
        for row in batch.rows:
            if not isinstance(row, (TokenRow, FlowRow)):
                raise TypeError("SenseNova mot route accepts token and flow rows")
            rows.append(row)
        flow_rows = tuple(row for row in rows if isinstance(row, FlowRow))
        flow_embeddings = self._flow_embeddings(flow_rows, batch.context) if flow_rows else {}
        chunks: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for row in rows:
            if isinstance(row, TokenRow):
                chunk = self._token_embeddings(row, batch.context)
            else:
                chunk = flow_embeddings[row.row_id]
            chunk = chunk.reshape(-1, chunk.shape[-1])
            chunks.append(chunk)
            spans.append((offset, offset + int(chunk.shape[0])))
            offset += int(chunk.shape[0])
        hidden = self.language_model.model(
            torch.cat(chunks, dim=0),
            batch.context,
        )

        outputs: list[TokenOutput | FlowOutput] = []
        for row, (begin, end) in zip(rows, spans, strict=True):
            row_hidden = hidden[begin:end]
            if isinstance(row, TokenRow):
                value: TokenHidden | TokenLogits
                if row.selection is TokenSelection.HIDDEN:
                    value = TokenHidden(row_hidden)
                elif row.selection is TokenSelection.ALL_LOGITS:
                    value = TokenLogits(self.language_model.lm_head(row_hidden, batch.context.mesh))
                else:
                    value = TokenLogits(
                        self.language_model.lm_head(row_hidden[-1:], batch.context.mesh)
                    )
                outputs.append(TokenOutput(row.row_id, row.output_slot, value))
            else:
                prediction = self._velocity(row_hidden, row, batch.context)
                outputs.append(FlowOutput(row.row_id, row.output_slot, prediction))
        return ForwardOutput(tuple(outputs))

    def _token_embeddings(
        self,
        row: TokenRow,
        context: ForwardContext,
    ) -> torch.Tensor:
        embed = self.language_model.model.embed_tokens
        if isinstance(row.inputs, TokenIds):
            return embed(row.inputs.values.reshape(-1), context.mesh)
        if isinstance(row.inputs, TokenEmbeddings):
            return row.inputs.values.reshape(-1, row.inputs.values.shape[-1])
        if not isinstance(row.inputs, TokenSegments):
            raise TypeError("SenseNova token row has an unknown input variant")
        return torch.cat(
            tuple(
                embed(segment.values.reshape(-1), context.mesh)
                if isinstance(segment, TokenIds)
                else segment.values.reshape(-1, segment.values.shape[-1])
                for segment in row.inputs.values
            ),
            dim=0,
        )

    def _velocity(
        self,
        hidden: torch.Tensor,
        row: FlowRow,
        context: ForwardContext,
    ) -> torch.Tensor:
        target = row.latent.device
        latent = row.latent
        was_flat = latent.ndim == 2
        latent_batch = latent.unsqueeze(0) if was_flat else latent
        hidden_batch = hidden.unsqueeze(0)
        local_hidden = context.mesh.dispatch(
            hidden_batch,
            "tower",
            _FLOW_COORDINATE,
        )
        local_latent = context.mesh.dispatch(
            latent_batch,
            "tower",
            _FLOW_COORDINATE,
        )
        local_timestep = context.mesh.dispatch(
            row.timestep.reshape(1),
            "tower",
            _FLOW_COORDINATE,
        )
        batch, latent_tokens = int(local_latent.shape[0]), int(local_latent.shape[1])
        if self._use_pixel_head:
            merge = int(1 / self._downsample_ratio)
            token_height = row.image_height // (self._patch_size * merge)
            token_width = row.image_width // (self._patch_size * merge)
            image = local_hidden[:, -row.image_tokens :].view(
                batch,
                token_height,
                token_width,
                -1,
            )
            image = torch.einsum("b h w c -> b c h w", image).contiguous()
            predicted = self.fm_modules["fm_head"](image)
            predicted = predicted.view(
                batch,
                3,
                token_height,
                self._patch_size * merge,
                token_width,
                self._patch_size * merge,
            )
            predicted = torch.einsum("b c h p w q -> b h w p q c", predicted)
            predicted = predicted.contiguous().view(
                batch,
                latent_tokens,
                self._patch_size * merge * self._patch_size * merge * 3,
            )
        elif self._use_deep_head:
            predicted = self.fm_modules["fm_head"](
                local_hidden[:, -row.image_tokens :].reshape(batch * latent_tokens, -1),
                local_timestep.repeat(batch * latent_tokens),
            ).view(batch, latent_tokens, -1)
        else:
            predicted = self.fm_modules["fm_head"](
                local_hidden[:, -row.image_tokens :].view(batch, latent_tokens, -1)
            ).view(batch, latent_tokens, -1)
        velocity = (predicted - local_latent) / (1 - local_timestep).clamp_min(_GENERATION_EPSILON)
        velocity = context.mesh.combine(
            velocity,
            "tower",
            _FLOW_COORDINATE,
            target,
        )
        return velocity[0] if was_flat else velocity

    def _forward_vision(self, batch: ForwardBatch) -> ForwardOutput:
        rows = tuple(row for row in batch.rows if isinstance(row, EncodeRow))
        if len(rows) != len(batch.rows) or any(
            row.kind is not EncodeKind.VISION or not isinstance(row.inputs, PatchInput)
            for row in rows
        ):
            raise TypeError("SenseNova vit route requires vision patch rows")
        pixels = torch.cat(
            tuple(cast(PatchInput, row.inputs).pixels for row in rows),
            dim=0,
        )
        grids = torch.cat(
            tuple(cast(PatchInput, row.inputs).grid for row in rows),
            dim=0,
        )
        features = self.vision_model(pixels, grids)
        factor = max(
            1,
            int(round(1 / self._downsample_ratio)),
        )
        counts = tuple(
            int(cast(PatchInput, row.inputs).pixels.shape[0]) // (factor * factor) for row in rows
        )
        if sum(counts) != int(features.shape[0]):
            raise ValueError("SenseNova vision output does not align with encode rows")
        outputs: list[EncodeOutput] = []
        offset = 0
        for row, count in zip(rows, counts, strict=True):
            outputs.append(
                EncodeOutput(
                    row.row_id,
                    row.output_slot,
                    features[offset : offset + count],
                )
            )
            offset += count
        return ForwardOutput(tuple(outputs))
