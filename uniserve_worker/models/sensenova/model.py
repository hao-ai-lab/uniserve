"""SenseNova-U1 execution behavior and neural equations."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...execution.batch import ForwardMode
from ...execution.forward_batch import (
    AttentionMode,
    ExpertRoute,
    ForwardBatch,
    ForwardOutput,
    RouteSpan,
    TokenSelection,
)
from ...loader.handles import WeightHandle
from ...loader.mapping import LoadReport, WeightNameMap, stacked_weight_name
from ...loader.weight_loaders import load_parameter_weight
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
from ...nn.expert_routing import RoutedTensor
from ...nn.layer import LayerConfig
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
from ..generation import (
    BranchSource,
    FlowPrompt,
    GenerationPipeline,
    LatentLayout,
    Materialization,
    NoiseScaleMode,
)
from ..inputs import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
)
from ..runtime import (
    CacheGeometry,
    ExecutionModel,
    PositionLayout,
    ResourceGeometry,
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

_STACKED_WEIGHTS: WeightNameMap = (
    ("qkv_proj_mot_gen", "q_proj_mot_gen", "q"),
    ("qkv_proj_mot_gen", "k_proj_mot_gen", "k"),
    ("qkv_proj_mot_gen", "v_proj_mot_gen", "v"),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _scope_includes(name: str, scope: str) -> bool:
    if scope == "whole":
        return True
    generation = name.startswith("fm_modules.") or "_mot_gen." in name
    if scope == "generation":
        return generation
    if scope == "understanding":
        return not generation
    raise ValueError(f"unknown SenseNova model scope {scope!r}")


def _check_checkpoint_code_version(config: Any) -> None:
    from packaging.version import Version

    raw = config.to_dict() if hasattr(config, "to_dict") else config
    if not isinstance(raw, dict):
        return
    required = raw.get("uniserve_sensenova_min_version")
    if required and Version(_MODEL_CODE_VERSION) < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")


def _module_tensor(
    module: nn.Module,
    value: torch.Tensor,
    *,
    context: ForwardBatch,
    coordinate: int,
    target: torch.device,
    call: Callable[[nn.Module, torch.Tensor, ForwardBatch], torch.Tensor],
) -> torch.Tensor:
    staged = context.mesh.dispatch(value, "tower", coordinate)
    result = call(module, staged, context)
    if not isinstance(result, torch.Tensor):
        raise TypeError("SenseNova neural sublayer must return a tensor")
    return context.mesh.combine(result, "tower", coordinate, target)


def _route_modules(
    value: RoutedTensor,
    *,
    text_module: nn.Module,
    flow_module: nn.Module,
    context: ForwardBatch,
    call: Callable[[nn.Module, torch.Tensor, ForwardBatch], torch.Tensor],
) -> RoutedTensor:
    def apply_text(item: torch.Tensor) -> torch.Tensor:
        return _module_tensor(
            text_module,
            item,
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=item.device,
            call=call,
        )

    def apply_flow(item: torch.Tensor) -> torch.Tensor:
        return _module_tensor(
            flow_module,
            item,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=item.device,
            call=call,
        )

    return value.map(apply_text, apply_flow)


def _plain_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardBatch,
) -> torch.Tensor:
    del context
    return cast(torch.Tensor, module(value))


def _parallel_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardBatch,
) -> torch.Tensor:
    return cast(torch.Tensor, module(value, context.mesh))


@dataclass(frozen=True, slots=True)
class _PackedRope:
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _RoutedRope:
    text: _PackedRope | None
    flow: _PackedRope | None


def _route_rope(rope: _PackedRope, spans: tuple[RouteSpan, ...]) -> _RoutedRope:
    cosine = tuple(RoutedTensor.from_packed(value, spans) for value in rope.cos)
    sine = tuple(RoutedTensor.from_packed(value, spans) for value in rope.sin)
    text = (
        None
        if cosine[0].text is None
        else _PackedRope(
            (
                cosine[0].text,
                cast(torch.Tensor, cosine[1].text),
                cast(torch.Tensor, cosine[2].text),
            ),
            (
                cast(torch.Tensor, sine[0].text),
                cast(torch.Tensor, sine[1].text),
                cast(torch.Tensor, sine[2].text),
            ),
        )
    )
    flow = (
        None
        if cosine[0].flow is None
        else _PackedRope(
            (
                cosine[0].flow,
                cast(torch.Tensor, cosine[1].flow),
                cast(torch.Tensor, cosine[2].flow),
            ),
            (
                cast(torch.Tensor, sine[0].flow),
                cast(torch.Tensor, sine[1].flow),
                cast(torch.Tensor, sine[2].flow),
            ),
        )
    )
    return _RoutedRope(text, flow)


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

    def forward(
        self,
        pixels: torch.Tensor,
        grid: torch.Tensor,
        *,
        grid_shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        return self.embeddings(pixels, grid, grid_shapes=grid_shapes)


class _SenseAttention(nn.Module):
    """SenseNova dual-expert QKV projection over one explicit attention plan."""

    def __init__(self, config: NeoLlmConfig, layer: int, *, spec: LayerConfig) -> None:
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
        context: ForwardBatch,
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
        hidden: RoutedTensor,
        *,
        context: ForwardBatch,
        spans: tuple[RouteSpan, ...],
        rope: _RoutedRope,
        causal: bool,
    ) -> RoutedTensor:
        text_projection = (
            None
            if hidden.text is None or rope.text is None
            else self._project(hidden.text, rope.text, generation=False, context=context)
        )
        flow_projection = (
            None
            if hidden.flow is None or rope.flow is None
            else self._project(hidden.flow, rope.flow, generation=True, context=context)
        )
        query = RoutedTensor(
            None if text_projection is None else text_projection[0],
            None if flow_projection is None else flow_projection[0],
        ).packed(spans)
        key = RoutedTensor(
            None if text_projection is None else text_projection[1],
            None if flow_projection is None else flow_projection[1],
        ).packed(spans)
        value = RoutedTensor(
            None if text_projection is None else text_projection[2],
            None if flow_projection is None else flow_projection[2],
        ).packed(spans)
        attended = self.attention(
            query,
            key,
            value,
            context,
            causal=causal,
            scale=self.scaling,
        ).reshape(query.shape[0], -1)
        return _route_modules(
            RoutedTensor.from_packed(attended, spans),
            text_module=self.o_proj,
            flow_module=self.o_proj_mot_gen,
            context=context,
            call=_parallel_call,
        )


class _SenseLayer(nn.Module):
    def __init__(self, config: NeoLlmConfig, layer: int, *, spec: LayerConfig) -> None:
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
        hidden: RoutedTensor,
        *,
        context: ForwardBatch,
        spans: tuple[RouteSpan, ...],
        rope: _RoutedRope,
        causal: bool,
    ) -> RoutedTensor:
        normalized = _route_modules(
            hidden,
            text_module=self.input_layernorm,
            flow_module=self.input_layernorm_mot_gen,
            context=context,
            call=_plain_call,
        )
        hidden = hidden.add(
            self.self_attn(
                normalized,
                context=context,
                spans=spans,
                rope=rope,
                causal=causal,
            )
        )
        normalized = _route_modules(
            hidden,
            text_module=self.post_attention_layernorm,
            flow_module=self.post_attention_layernorm_mot_gen,
            context=context,
            call=_plain_call,
        )
        feed_forward = _route_modules(
            normalized,
            text_module=self.mlp,
            flow_module=self.mlp_mot_gen,
            context=context,
            call=_parallel_call,
        )
        return hidden.add(feed_forward)


class _SenseDecoder(nn.Module):
    """One packed text/flow decoder with no serving state."""

    def __init__(self, config: NeoLlmConfig, *, spec: LayerConfig) -> None:
        super().__init__()
        hidden = int(getattr(config, "hidden_size"))
        self.embed_tokens = VocabParallelEmbedding(
            int(getattr(config, "vocab_size")),
            hidden,
            int(getattr(config, "pad_token_id")),
            spec=spec,
            init_weights=False,
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
        context: ForwardBatch,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("SenseNova decoder inputs must have shape [tokens, hidden]")
        token_count = int(inputs.shape[0])
        spans: tuple[RouteSpan, ...]
        indexes: torch.Tensor
        causal: bool
        if context.forward_mode is AttentionMode.PACKED:
            if context.attention_indexes is None or tuple(context.attention_indexes.shape) != (
                3,
                token_count,
            ):
                raise ValueError("SenseNova positions must have shape [3, tokens]")
            spans = context.route_spans
            indexes = context.attention_indexes
            causal = False
        elif context.forward_mode is AttentionMode.PAGED_DECODE:
            if positions is None or tuple(positions.shape) != (token_count,):
                raise ValueError("SenseNova paged decode positions must align with text tokens")
            spans = (RouteSpan(ExpertRoute.TEXT, 0, token_count),)
            indexes = torch.stack(
                (
                    positions,
                    torch.zeros_like(positions),
                    torch.zeros_like(positions),
                )
            )
            causal = True
        else:
            raise ValueError("SenseNova decoder requires packed attention or paged decode")
        if not self.layers:
            raise ValueError("SenseNova decoder requires at least one layer")

        first = cast(_SenseLayer, self.layers[0])
        rope = _route_rope(first.self_attn.rope(indexes), spans)
        hidden = RoutedTensor.from_packed(inputs, spans)
        for layer_module in self.layers:
            layer = cast(_SenseLayer, layer_module)
            hidden = layer(
                hidden,
                context=context,
                spans=spans,
                rope=rope,
                causal=causal,
            )
        return _route_modules(
            hidden,
            text_module=self.norm,
            flow_module=self.norm_mot_gen,
            context=context,
            call=_plain_call,
        ).packed(spans)


class _LanguageModel(nn.Module):
    def __init__(self, config: NeoLlmConfig, *, spec: LayerConfig) -> None:
        super().__init__()
        self.model = _SenseDecoder(config, spec=spec)
        self.lm_head = ParallelLMHead(
            int(getattr(config, "hidden_size")),
            int(getattr(config, "vocab_size")),
            spec=spec,
            bias=False,
        )


class NEOChatModel(ExecutionModel):
    """Concrete stateless SenseNova model for mixed text, flow, and vision rows."""

    def load_weights(
        self,
        weights: Iterable[WeightHandle],
    ) -> LoadReport:
        """Stream checkpoint tensors into the selected SenseNova tower scope."""

        parameter_names = set(dict(self.named_parameters()))
        included = {name for name in parameter_names if _scope_includes(name, self._load_scope)}
        report = LoadReport()
        for handle in weights:
            source_name = handle.name
            target_name, shard_id = stacked_weight_name(source_name, _STACKED_WEIGHTS)
            if target_name not in parameter_names:
                if source_name in parameter_names:
                    target_name, shard_id = source_name, None
                else:
                    report.unexpected.append(source_name)
                    continue
            if target_name not in included:
                continue
            parameter = dict(self.named_parameters())[target_name]
            load_parameter_weight(parameter, handle, shard_id)
            report.loaded.add(target_name)
        return report

    def checkpoint_parameter_names(self) -> set[str]:
        return {
            name for name, _ in self.named_parameters() if _scope_includes(name, self._load_scope)
        }

    def __init__(
        self,
        config: NeoChatConfig,
        *,
        layer_config: LayerConfig,
        scope: str = "whole",
    ) -> None:
        super().__init__()
        if scope not in {"whole", "understanding", "generation"}:
            raise ValueError(f"unknown SenseNova model scope {scope!r}")
        self._load_scope = scope
        vision = config.vision_config
        hidden = int(config.llm_config.hidden_size)
        self.vision_model = _VisionModel(vision)
        self._parallel = layer_config.parallel
        self.language_model = _LanguageModel(config.llm_config, spec=layer_config)
        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": _VisionModel(vision),
                "timestep_embedder": TimestepEmbedder(hidden),
                "fm_head": self._flow_head(config, hidden, layer_config),
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
        self._configure_runtime(config)

    @staticmethod
    def _flow_head(
        config: NeoChatConfig,
        hidden: int,
        layer_config: LayerConfig,
    ) -> nn.Module:
        merge = int(1 / float(config.downsample_ratio))
        output_dim = 3 * (int(config.vision_config.patch_size) * merge) ** 2
        if int(getattr(config, "fm_head_layers", 2)) > 2:
            return FlowMatchingHead(
                hidden,
                output_dim,
                spec=layer_config,
                dim=int(getattr(config, "fm_head_dim")),
                layers=int(getattr(config, "fm_head_layers")),
                mlp_ratio=float(getattr(config, "fm_head_mlp_ratio")),
            )
        return nn.Sequential(
            LinearBase(hidden, 4096, spec=layer_config, bias=True),
            nn.GELU(),
            LinearBase(4096, output_dim, spec=layer_config, bias=True),
        )

    def _configure_runtime(self, config: NeoChatConfig) -> None:
        llm = config.llm_config
        vision = config.vision_config
        max_text = int(getattr(llm, "max_position_embeddings", 32768))
        max_image = max(1, int(getattr(config, "max_image_seq_len", 4096)))
        latent_downsample = int(int(vision.patch_size) * round(1 / float(config.downsample_ratio)))
        self.architecture = "NEOChatModel"
        self.generation = GenerationPipeline(
            latent_downsample=latent_downsample,
            prediction="velocity",
            prediction_dtype="float32",
            schedule_direction=ScheduleDirection.ASCENDING,
            schedule_shift_domain=ScheduleShiftDomain.SIGMA,
            max_latent_tokens=max_image,
            max_vae_grid_tokens=max_image,
            commit_marker_tokens=2,
            rope_advance=2,
            max_cfg_branches=_MAX_CFG_BRANCHES,
            latent_layout=LatentLayout.IMAGE_NCHW,
            latent_channels=3,
            latent_patch_size=latent_downsample,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            materialization=Materialization.RGB_LATENT,
            noise_scale=float(getattr(config, "noise_scale", 1.0)),
            noise_scale_mode=NoiseScaleMode(
                str(
                    getattr(
                        getattr(config, "noise_scale_mode", "constant"),
                        "value",
                        getattr(config, "noise_scale_mode", "constant"),
                    )
                )
            ),
            noise_scale_base_tokens=float(getattr(config, "noise_scale_base_image_seq_len", 1.0)),
            noise_scale_maximum=float(getattr(config, "noise_scale_max_value", 1.0)),
            text_unconditional=BranchSource.NEGATIVE_OR_START,
            image_unconditional=BranchSource.START,
            cfg_recipe=CfgRecipe.ADDITIVE_DELTAS,
            prompt=FlowPrompt(
                user_prefix="<|im_start|>user\n",
                user_suffix="<|im_end|>\n",
                assistant_suffix="<|im_start|>assistant\n",
                conditioned_append="<think>\n\n</think>\n\n<img>",
                unconditional_append="<img>",
                system_prefix="<|im_start|>system\n",
                system_message=_FLOW_SYSTEM_MESSAGE,
                system_suffix="<|im_end|>\n",
            ),
        )
        self.image_processor = ImageProcessor(
            vit=PatchTransform(
                patch_size=int(vision.patch_size),
                downsample_ratio=float(vision.downsample_ratio),
                min_pixels=512 * 512,
                max_pixels=2048 * 2048,
            ),
            staging_dtype="bfloat16",
            feature_injection=FeatureInjection(
                layout=FeatureLayout.DIRECT,
                positions=PositionLayout.TEMPORAL_SPATIAL,
                start_token="<img>",
                end_token="</img>",
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
            latent_downsample=latent_downsample,
        )
        self.supported_work = frozenset(
            {
                ForwardMode.TOKEN_EXTEND,
                ForwardMode.TOKEN_DECODE,
                ForwardMode.TOKEN_VERIFY,
                ForwardMode.GEN_TRANSITION,
                ForwardMode.GEN_FLOW,
                ForwardMode.ENCODE_VISION,
                ForwardMode.MATERIALIZE,
                ForwardMode.TRANSFER_PRODUCT,
                ForwardMode.TRANSFER_KV_PUBLISH,
                ForwardMode.TRANSFER_KV_INSTALL,
            }
        )
        self.max_vit_grid_tokens = _MAX_VISION_TOKENS
        self.vocab_size = int(llm.vocab_size)
        self.hidden_size = int(llm.hidden_size)
        self.text_max_tokens = max(max_text, max_image)
        self.text_topology = ("tp", "tower")
        self.tensorized_mixed = True

    def _flow_embeddings(
        self,
        batch: ForwardBatch,
    ) -> dict[int, torch.Tensor]:
        patches = batch.flow_conditioning
        if any(value is None for value in patches):
            raise TypeError("SenseNova flow rows require patch conditioning")
        typed = tuple(value for value in patches if value is not None)
        target = batch.flow_latents[0].device
        pixels = torch.cat(tuple(value.pixels for value in typed), dim=0)
        grids = torch.cat(tuple(value.grid for value in typed), dim=0)
        local_pixels = batch.mesh.dispatch(pixels, "tower", _FLOW_COORDINATE)
        local_grids = batch.mesh.dispatch(grids, "tower", _FLOW_COORDINATE)
        tower = self.fm_modules["vision_model_mot_gen"]
        feature_dtype = next(tower.parameters()).dtype
        # Each flow row's image patch grid (height/patch, width/patch) is known
        # on the host from its registered image size. Passing the per-row grids
        # lets the tower resolve the conv geometry without reading the grid
        # tensor back, keeping the flow forward capturable in a CUDA graph.
        patch = self._patch_size
        grid_shapes = tuple(
            (height // patch, width // patch)
            for height, width in zip(batch.flow_heights, batch.flow_widths, strict=True)
        )
        features = tower(local_pixels.to(dtype=feature_dtype), local_grids, grid_shapes=grid_shapes)
        if not isinstance(features, torch.Tensor):
            raise TypeError("SenseNova flow vision tower must return a tensor")
        features = batch.mesh.combine(
            features,
            "tower",
            _FLOW_COORDINATE,
            target,
        )
        expected = sum(batch.flow_image_tokens)
        if int(features.shape[0]) != expected:
            raise ValueError("SenseNova flow vision features do not match row geometry")

        timesteps = torch.cat(
            tuple(
                timestep.reshape(1).expand(image_tokens)
                for timestep, image_tokens in zip(
                    batch.flow_timesteps, batch.flow_image_tokens, strict=True
                )
            ),
            dim=0,
        )
        time_features = _module_tensor(
            self.fm_modules["timestep_embedder"],
            timesteps,
            context=batch,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=_plain_call,
        )
        features = features + time_features
        if self._add_noise_embedding:
            noise = torch.cat(
                tuple(
                    value.noise_scale.reshape(1).expand(image_tokens)
                    for image_tokens, value in zip(batch.flow_image_tokens, typed, strict=True)
                ),
                dim=0,
            )
            noise = noise / self._noise_scale_max
            features = features + _module_tensor(
                self.fm_modules["noise_scale_embedder"],
                noise,
                context=batch,
                coordinate=_FLOW_COORDINATE,
                target=target,
                call=_plain_call,
            )

        result: dict[int, torch.Tensor] = {}
        offset = 0
        for row_index, image_tokens in zip(
            batch.flow_row_indices, batch.flow_image_tokens, strict=True
        ):
            end = offset + image_tokens
            result[row_index] = features[offset:end]
            offset = end
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        decode_positions: torch.Tensor | None = None
        if batch.forward_mode is AttentionMode.PAGED_DECODE:
            if batch.flow_row_indices:
                raise TypeError("SenseNova paged decode accepts token rows only")
            decode_positions = positions
        token_embeds = self.language_model.model.embed_tokens(input_ids.reshape(-1), batch.mesh)
        if batch.input_embeddings is not None:
            if batch.embedding_mask is None:
                raise RuntimeError("SenseNova embedding input lost its selection mask")
            token_embeds = torch.where(
                batch.embedding_mask.reshape(-1, 1),
                batch.input_embeddings.to(dtype=token_embeds.dtype),
                token_embeds,
            )
        flow_embeddings = self._flow_embeddings(batch) if batch.flow_row_indices else {}
        chunks: list[torch.Tensor | None] = [None] * batch.row_count
        token_offset = 0
        for row_index, count in zip(
            batch.token_row_indices,
            tuple(batch.query_lens_cpu[index] for index in batch.token_row_indices),
            strict=True,
        ):
            chunks[row_index] = token_embeds[token_offset : token_offset + count]
            token_offset += count
        for row_index in batch.flow_row_indices:
            chunks[row_index] = flow_embeddings[row_index]
        if any(value is None for value in chunks):
            raise RuntimeError("SenseNova forward batch contains an unbound row")
        return self.language_model.model(
            torch.cat(
                tuple(value.reshape(-1, value.shape[-1]) for value in chunks if value is not None),
                dim=0,
            ),
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
        for row_index, count in zip(batch.flow_row_indices, batch.flow_image_tokens, strict=True):
            row_lengths[row_index] = count
        rows: list[torch.Tensor] = []
        offset = 0
        for count in row_lengths:
            rows.append(hidden[offset : offset + count])
            offset += count
        row_hidden = tuple(rows)
        selection_by_row = dict(zip(batch.token_row_indices, batch.token_selections, strict=True))
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
                and all(int(value.shape[0]) == 1 for value in row_hidden)
            ):
                selected = hidden
            else:
                selected_rows = tuple(
                    row_hidden[index]
                    if selection_by_row[index] is TokenSelection.ALL_LOGITS
                    else row_hidden[index][-1:]
                    for index in projected_rows
                )
                selected = (
                    selected_rows[0] if len(selected_rows) == 1 else torch.cat(selected_rows, dim=0)
                )
            projected = self.language_model.lm_head(selected, batch.mesh)

        outputs: list[torch.Tensor] = []
        projected_offset = 0
        flow_by_row = {row_index: index for index, row_index in enumerate(batch.flow_row_indices)}
        for index in range(batch.row_count):
            value_hidden = row_hidden[index]
            selection = selection_by_row.get(index)
            if selection is not None:
                if selection is TokenSelection.HIDDEN:
                    value = value_hidden
                else:
                    if projected is None:
                        raise RuntimeError("SenseNova projected output buffer is missing")
                    count = (
                        int(value_hidden.shape[0]) if selection is TokenSelection.ALL_LOGITS else 1
                    )
                    value = projected[projected_offset : projected_offset + count]
                    projected_offset += count
                outputs.append(value)
            else:
                flow_index = flow_by_row[index]
                outputs.append(self._velocity(value_hidden, flow_index, batch))
        return ForwardOutput(tuple(outputs))

    def _velocity(
        self,
        hidden: torch.Tensor,
        flow_index: int,
        context: ForwardBatch,
    ) -> torch.Tensor:
        target = context.flow_latents[flow_index].device
        latent = context.flow_latents[flow_index]
        image_tokens = context.flow_image_tokens[flow_index]
        image_height = context.flow_heights[flow_index]
        image_width = context.flow_widths[flow_index]
        timestep = context.flow_timesteps[flow_index]
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
            timestep.reshape(1),
            "tower",
            _FLOW_COORDINATE,
        )
        batch, latent_tokens = int(local_latent.shape[0]), int(local_latent.shape[1])
        if self._use_pixel_head:
            merge = int(1 / self._downsample_ratio)
            token_height = image_height // (self._patch_size * merge)
            token_width = image_width // (self._patch_size * merge)
            image = local_hidden[:, -image_tokens:].view(
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
                local_hidden[:, -image_tokens:].reshape(batch * latent_tokens, -1),
                local_timestep.repeat(batch * latent_tokens),
            ).view(batch, latent_tokens, -1)
        else:
            predicted = self.fm_modules["fm_head"](
                local_hidden[:, -image_tokens:].view(batch, latent_tokens, -1)
            ).view(batch, latent_tokens, -1)
        velocity = (predicted - local_latent) / (1 - local_timestep).clamp_min(_GENERATION_EPSILON)
        velocity = context.mesh.combine(
            velocity,
            "tower",
            _FLOW_COORDINATE,
            target,
        )
        return velocity[0] if was_flat else velocity

    def encode(self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch) -> ForwardOutput:
        if any(grid is None for grid in batch.encode_grids) or any(
            shape is None for shape in batch.encode_grid_shapes
        ):
            raise TypeError("SenseNova vision encode requires patch grids")
        grids = torch.cat(tuple(grid for grid in batch.encode_grids if grid is not None), dim=0)
        grid_shapes = tuple(shape for shape in batch.encode_grid_shapes if shape is not None)
        packed_pixels = torch.cat(pixels, dim=0)
        features = self.vision_model(packed_pixels, grids, grid_shapes=grid_shapes)
        factor = max(
            1,
            int(round(1 / self._downsample_ratio)),
        )
        counts = tuple(int(value.shape[0]) // (factor * factor) for value in pixels)
        if sum(counts) != int(features.shape[0]):
            raise ValueError("SenseNova vision output does not align with encode rows")
        outputs: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            outputs.append(features[offset : offset + count])
            offset += count
        return ForwardOutput(tuple(outputs))
