"""SenseNova-U1 execution behavior and neural equations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from uniserve_worker.protocol.batch import ForwardMode, PipelineStage, TransferMode

from ...execution.device_transfer import call_on_device, tensor_to_device
from ...execution.forward_batch import (
    AttentionMode,
    ExpertRoute,
    ForwardBatch,
    ForwardOutput,
    RouteSpan,
)
from ...loader.component import construct_owned_module
from ...loader.mapping import WeightNameMap
from ...nn.attention import RadixAttention
from ...nn.diffusion import (
    ConvDecoder,
    FlowMatchingHead,
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from ...nn.diffusion.cfg import CfgRecipe
from ...nn.expert_routing import RoutedTensor, slice_route_spans
from ...nn.layer import LayerConfig
from ...nn.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
    local_attention_head_count,
    local_kv_head_count,
    local_kv_head_offset,
)
from ...nn.logits import project_outputs
from ...nn.mlp import GatedMLP
from ...nn.norm import RMSNorm
from ...nn.parallel_pipeline import LayerPipeline
from ...nn.parallel_sequence import SequencePartition
from ...nn.rope import HFRotaryEmbedding, RotaryEmbedding, get_rope, qk_norm_rope
from ...nn.row_pipeline import (
    RowStage,
    RowTensors,
    RowTensorSegments,
    independent_linear_rows,
    packed_row_stage,
    run_row_pipeline,
)
from ...nn.shard import WeightMode
from ...nn.vision import NeoVitConfig, NeoVitEncoder
from ...nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    vocabulary_partition,
)
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

if TYPE_CHECKING:
    from ...loader.component import CheckpointComponent, ModelBuildContext, ModelConstruction


__all__ = ["NEOChatModel"]

_MODEL_CODE_VERSION = "0.1.0"
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


def _check_checkpoint_code_version(config: Any) -> None:
    """Validate the checkpoint-declared custom-code version when present."""

    from packaging.version import Version

    raw = config.to_dict() if hasattr(config, "to_dict") else config
    if not isinstance(raw, dict):
        return
    required = raw.get("uniserve_sensenova_min_version")
    if required and Version(_MODEL_CODE_VERSION) < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")


@dataclass(frozen=True, slots=True)
class _PackedRope:
    """Holds cosine and sine tables aligned to one packed token stream."""

    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _RoutedRope:
    """Independent rotary tables for text and diffusion-flow routes."""

    text: _PackedRope | None
    flow: _PackedRope | None

    def narrow(self, interval: slice, spans: tuple[RouteSpan, ...]) -> _RoutedRope:
        """Retain the temporal/spatial factors for one packed numerical interval."""

        def select(attribute: str) -> tuple[RoutedTensor, ...]:
            return tuple(
                RoutedTensor(
                    None if self.text is None else getattr(self.text, attribute)[axis],
                    None if self.flow is None else getattr(self.flow, attribute)[axis],
                ).narrow(interval, spans)
                for axis in range(3)
            )

        cos, sin = select("cos"), select("sin")

        def expert(route: str) -> _PackedRope:
            cos_t, cos_h, cos_w = (getattr(item, route) for item in cos)
            sin_t, sin_h, sin_w = (getattr(item, route) for item in sin)
            return _PackedRope((cos_t, cos_h, cos_w), (sin_t, sin_h, sin_w))

        return _RoutedRope(
            None if self.text is None else expert("text"),
            None if self.flow is None else expert("flow"),
        )


def _route_rope(
    rope: _PackedRope, spans: tuple[RouteSpan, ...], routes: frozenset[ExpertRoute] = frozenset()
) -> _RoutedRope:
    """Split packed rotary tables into independent text and flow route tables."""

    cosine = tuple(RoutedTensor.from_packed(value, spans, routes=routes) for value in rope.cos)
    sine = tuple(RoutedTensor.from_packed(value, spans, routes=routes) for value in rope.sin)
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
        """Build the vision encoder that projects patches directly to language width."""

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
        """Embed flattened vision patches and add their grid-derived positions."""

        return self.embeddings(pixels, grid, grid_shapes=grid_shapes)


class _SenseAttention(nn.Module):
    """SenseNova dual-expert QKV projection over one explicit attention plan."""

    def __init__(
        self,
        config: NeoLlmConfig,
        layer: int,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Build text and flow projection towers around one shared attention backend."""

        super().__init__()
        self.generation_device = generation_device
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
            layer_config=layer_config,
            prefix="qkv_proj",
            bias=bias,
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config,
            prefix="qkv_proj_mot_gen",
            packed_names=("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen"),
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
            sequence=layer_config.sequence,
        )
        self.o_proj = RowParallelLinear(
            query_width, hidden_size, layer_config=layer_config, prefix="o_proj", bias=bias
        )
        self.o_proj_mot_gen = RowParallelLinear(
            query_width,
            hidden_size,
            layer_config=layer_config,
            prefix="o_proj_mot_gen",
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

    def rope(self, indexes: torch.Tensor) -> _PackedRope:
        """Build temporal, height, and width rotary tables for ``[3, tokens]`` indexes."""

        if indexes.ndim != 2 or tuple(indexes.shape[:1]) != (3,):
            raise ValueError("SenseNova positions must have shape [3, tokens]")
        target = indexes.device

        def frequencies(
            module: RotaryEmbedding | HFRotaryEmbedding,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Evaluate one rotary axis on its module device and restore the target device."""

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
        """Project and rotate one route's QKV rows on its declared component device."""

        device = self.generation_device if generation else None
        target = hidden.device
        staged = tensor_to_device(hidden, device)
        qkv_module = self.qkv_proj_mot_gen if generation else self.qkv_proj
        split_sizes = tuple(int(size) for size in qkv_module.output_sizes)
        if generation:
            query_flat, key_flat, value_flat = qkv_module.forward_branches(staged)
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
        local_cos = tuple(tensor_to_device(item, device) for item in rope.cos)
        local_sin = tuple(tensor_to_device(item, device) for item in rope.sin)
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
            tensor_to_device(query, target),
            tensor_to_device(key, target),
            tensor_to_device(value, target),
        )

    def project_rows(
        self,
        hidden: RoutedTensor,
        *,
        context: ForwardBatch,
        spans: tuple[RouteSpan, ...],
        rope: _RoutedRope,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare routed normalized QKV without consuming global attention."""

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
        return query, key, value


class _SenseLayer(nn.Module):
    """Routes packed text and flow tokens through shared attention and modality-specific feed-forward experts."""

    def __init__(
        self,
        config: NeoLlmConfig,
        layer: int,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Assemble shared attention with route-specific norms and feed-forward towers."""

        super().__init__()
        self.generation_device = generation_device
        hidden = int(getattr(config, "hidden_size"))
        epsilon = float(getattr(config, "rms_norm_eps"))
        self.self_attn = _SenseAttention(
            config,
            layer,
            layer_config=layer_config.child("self_attn"),
            generation_device=generation_device,
        )
        self.mlp = GatedMLP(
            hidden,
            int(config.intermediate_size),
            hidden_act=getattr(config, "hidden_act", "silu"),
            layer_config=layer_config.child("mlp"),
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.mlp_mot_gen = GatedMLP(
            hidden,
            int(config.intermediate_size),
            hidden_act=getattr(config, "hidden_act", "silu"),
            layer_config=layer_config.child("mlp_mot_gen"),
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.input_layernorm = RMSNorm(hidden, eps=epsilon)
        self.input_layernorm_mot_gen = RMSNorm(hidden, eps=epsilon)
        self.post_attention_layernorm = RMSNorm(hidden, eps=epsilon)
        self.post_attention_layernorm_mot_gen = RMSNorm(hidden, eps=epsilon)

    def row_stage(
        self,
        *,
        context: ForwardBatch,
        spans: tuple[RouteSpan, ...],
        routes: frozenset[ExpertRoute],
        rope: _RoutedRope,
        causal: bool,
        partition: SequencePartition,
    ) -> RowStage[RowTensorSegments]:
        """Declare routed input and output equations around shared row transport."""

        independent_output = independent_linear_rows(
            self.self_attn.o_proj, self.self_attn.o_proj_mot_gen, self.mlp, self.mlp_mot_gen
        )

        def project(interval: slice, values: RowTensors) -> RowTensors:
            local_spans = slice_route_spans(spans, interval)
            hidden = RoutedTensor.from_packed(values[0], local_spans, routes=routes)
            normalized = hidden.apply(
                text=self.input_layernorm,
                flow=self.input_layernorm_mot_gen,
                generation_device=self.generation_device,
            )
            projected = self.self_attn.project_rows(
                normalized, context=context, spans=local_spans, rope=rope.narrow(interval, spans)
            )
            return (*projected, values[0])

        def finish(interval: slice, attended: torch.Tensor, state: RowTensors) -> RowTensors:
            local_spans = slice_route_spans(spans, interval)
            hidden = RoutedTensor.from_packed(state[0], local_spans, routes=routes)
            attended = attended.reshape(
                attended.shape[0], self.self_attn.num_heads * self.self_attn.head_dim
            )
            projected = RoutedTensor.from_packed(attended, local_spans, routes=routes).apply(
                text=self.self_attn.o_proj,
                flow=self.self_attn.o_proj_mot_gen,
                generation_device=self.generation_device,
            )
            hidden = hidden.add(projected)
            normalized = hidden.apply(
                text=self.post_attention_layernorm,
                flow=self.post_attention_layernorm_mot_gen,
                generation_device=self.generation_device,
            )
            feed_forward = normalized.apply(
                text=self.mlp, flow=self.mlp_mot_gen, generation_device=self.generation_device
            )
            return (hidden.add(feed_forward).packed(local_spans),)

        return packed_row_stage(
            project,
            self.self_attn.attention,
            finish,
            context=context.attention,
            partition=partition,
            causal=causal,
            scale=self.self_attn.scaling,
            independent_input=independent_linear_rows(
                self.self_attn.qkv_proj, self.self_attn.qkv_proj_mot_gen
            ),
            independent_output=independent_output,
        )


class _SenseDecoder(nn.Module):
    """One packed text/flow decoder with no serving state."""

    def __init__(
        self,
        config: NeoLlmConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Build the packed decoder with the declared device for its flow components."""

        super().__init__()
        self.generation_device = generation_device
        hidden = int(getattr(config, "hidden_size"))
        self.hidden_size = hidden
        self.pipeline = LayerPipeline(layer_config.pipeline, int(config.num_hidden_layers))
        self.sequence = layer_config.sequence
        self.embed_tokens = (
            VocabParallelEmbedding(
                int(getattr(config, "vocab_size")),
                hidden,
                int(getattr(config, "pad_token_id")),
                layer_config=layer_config,
                init_weights=False,
            )
            if self.pipeline.first
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(index): _SenseLayer(
                    config,
                    index - self.pipeline.layers.start,
                    layer_config=layer_config.child(f"layers.{index}"),
                    generation_device=generation_device,
                )
                for index in self.pipeline.layers
            }
        )
        epsilon = float(getattr(config, "rms_norm_eps"))
        self.norm = RMSNorm(hidden, eps=epsilon) if self.pipeline.last else None
        self.norm_mot_gen = RMSNorm(hidden, eps=epsilon) if self.pipeline.last else None
        parameters = tuple(dict(next(iter(self.layers.values())).named_parameters()))
        nonresident = self.pipeline.nonresident_layer_names("layers", parameters)
        if not self.pipeline.first:
            nonresident |= {"embed_tokens.weight"}
        if not self.pipeline.last:
            nonresident |= {"norm.weight", "norm_mot_gen.weight"}
        self.nonresident_parameters = nonresident

    def forward(
        self,
        inputs: torch.Tensor | None,
        context: ForwardBatch,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode packed routed rows and return final-normalized states in input order."""

        first = cast(_SenseLayer, next(iter(self.layers.values())))
        if inputs is not None:
            token_count = int(inputs.shape[0])
        elif (
            context.attention.attention_mode is AttentionMode.PACKED
            and context.attention.attention_indexes is not None
        ):
            token_count = int(context.attention.attention_indexes.shape[1])
        elif (
            context.attention.attention_mode is AttentionMode.PAGED_DECODE and positions is not None
        ):
            token_count = positions.numel()
        else:
            raise ValueError("pipeline input requires packed or decode row geometry")
        partition = SequencePartition(token_count, self.sequence)
        if inputs is None:
            if self.pipeline.first:
                raise ValueError("the first decoder stage requires input embeddings")
            inputs = first.input_layernorm.weight.new_empty((partition.count, self.hidden_size))
        else:
            if inputs.ndim != 2:
                raise ValueError("SenseNova decoder inputs must have shape [tokens, hidden]")
            inputs = partition.local(inputs)
        self.pipeline.receive_activation(inputs)
        spans: tuple[RouteSpan, ...]
        indexes: torch.Tensor
        causal: bool
        if context.attention.attention_mode is AttentionMode.PACKED:
            if context.attention.attention_indexes is None or tuple(
                context.attention.attention_indexes.shape
            ) != (
                3,
                token_count,
            ):
                raise ValueError("SenseNova positions must have shape [3, tokens]")
            spans = context.attention.route_spans
            indexes = context.attention.attention_indexes
            causal = False
        elif context.attention.attention_mode is AttentionMode.PAGED_DECODE:
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
        routes = frozenset(span.route for span in spans)
        spans = partition.routes(spans)
        rope = _route_rope(first.self_attn.rope(partition.local(indexes, axis=1)), spans, routes)
        (packed,) = run_row_pipeline(
            RowTensorSegments.complete((inputs,)),
            tuple(
                cast(_SenseLayer, layer).row_stage(
                    context=context,
                    spans=spans,
                    routes=routes,
                    rope=rope,
                    causal=causal,
                    partition=partition,
                )
                for layer in self.layers.values()
            ),
        ).materialize()
        if not self.pipeline.last:
            self.pipeline.send_activation(packed)
            return packed
        hidden = RoutedTensor.from_packed(packed, spans, routes=routes)
        assert self.norm is not None and self.norm_mot_gen is not None
        return partition.gather(
            hidden.apply(
                text=self.norm, flow=self.norm_mot_gen, generation_device=self.generation_device
            ).packed(spans)
        )


class _LanguageModel(nn.Module):
    """Owns the SenseNova token embedding, decoder, normalization, and vocabulary projection stack."""

    def __init__(
        self,
        config: NeoLlmConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Build the decoder and tensor-parallel vocabulary projection."""

        super().__init__()
        self.generation_device = generation_device
        self.model = _SenseDecoder(
            config, layer_config=layer_config.child("model"), generation_device=generation_device
        )
        self.lm_head = (
            ParallelLMHead(
                int(getattr(config, "hidden_size")),
                int(getattr(config, "vocab_size")),
                layer_config=layer_config,
                prefix="lm_head",
                bias=False,
            )
            if self.model.pipeline.last
            else None
        )
        self.nonresident_parameters = frozenset(
            f"model.{name}" for name in self.model.nonresident_parameters
        )
        if not self.model.pipeline.last:
            self.nonresident_parameters |= {"lm_head.weight"}


class NEOChatModel(ExecutionModel):
    """Concrete stateless SenseNova model for mixed text, flow, and vision rows."""

    ordered_collective_execution = True

    @classmethod
    def build_checkpoint(
        cls, config: dict[str, Any], context: ModelBuildContext
    ) -> ModelConstruction:
        """Interpret SenseNova configuration and declare its mapped checkpoint tensors."""

        from transformers import AutoTokenizer

        from ...loader.component import ModelConstruction

        prepared = NeoChatConfig.from_dict(config)
        _check_checkpoint_code_version(prepared)
        model = cls(
            prepared,
            layer_config=context.packed_decoder_layers("model"),
            generation_device=(
                None
                if context.request.execution.generation_device is None
                else torch.device(context.request.execution.generation_device)
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            context.root,
            use_fast=False,
            trust_remote_code=False,
            local_files_only=True,
        )
        return ModelConstruction(model.checkpoint_components(), lambda: model, prepared, tokenizer)

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare checkpoint projection mapping and branch-specific tensor devices."""

        from ...loader.component import CheckpointComponent

        return (
            CheckpointComponent(
                self,
                weight_name_map=_STACKED_WEIGHTS,
                nonresident=self.nonresident_parameters,
                module_devices=(
                    ()
                    if self.generation_device is None
                    else tuple(
                        (name, self.generation_device)
                        for name, _ in self.named_modules()
                        if name == "fm_modules" or name.endswith("_mot_gen")
                    )
                ),
            ),
        )

    def __init__(
        self,
        config: NeoChatConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Construct SenseNova neural components and publish their serving geometry."""

        super().__init__()
        self.generation_device = generation_device
        vision = config.vision_config
        hidden = int(config.llm_config.hidden_size)
        self.vision_model = _VisionModel(vision)
        self._parallel = layer_config.communicator
        self.language_model = _LanguageModel(
            config.llm_config,
            layer_config=layer_config.child("language_model"),
            generation_device=generation_device,
        )
        self.nonresident_parameters = frozenset(
            f"language_model.{name}" for name in self.language_model.nonresident_parameters
        )
        self._patch_size = int(vision.patch_size)
        self._downsample_ratio = float(config.downsample_ratio)
        self._use_deep_head = bool(getattr(config, "fm_head_layers", 2) > 2)
        self._use_pixel_head = bool(getattr(config, "use_pixel_head", False))
        self._add_noise_embedding = bool(getattr(config, "add_noise_scale_embedding", False))
        self._noise_scale_max = float(getattr(config, "noise_scale_max_value", 1.0))
        pipeline = self.language_model.model.pipeline
        declarations = [
            ("vision_model_mot_gen", pipeline.first, lambda: _VisionModel(vision)),
            ("timestep_embedder", pipeline.first, lambda: TimestepEmbedder(hidden)),
            (
                "fm_head",
                pipeline.last,
                lambda: (
                    ConvDecoder(hidden)
                    if self._use_pixel_head
                    else self._flow_head(config, hidden, layer_config.child("fm_modules.fm_head"))
                ),
            ),
        ]
        if self._add_noise_embedding:
            declarations.append(
                ("noise_scale_embedder", pipeline.first, lambda: TimestepEmbedder(hidden))
            )
        modules = {}
        for name, resident, factory in declarations:
            module, nonresident = construct_owned_module(factory, resident=resident)
            if module is not None:
                modules[name] = module
            self.nonresident_parameters |= {
                f"fm_modules.{name}.{parameter}" for parameter in nonresident
            }
        self.fm_modules = nn.ModuleDict(modules)
        self._configure_runtime(config)

    @staticmethod
    def _flow_head(
        config: NeoChatConfig,
        hidden: int,
        layer_config: LayerConfig,
    ) -> nn.Module:
        """Build the configured shallow or deep patch-space flow prediction head."""

        merge = int(1 / float(config.downsample_ratio))
        output_dim = 3 * (int(config.vision_config.patch_size) * merge) ** 2
        if int(getattr(config, "fm_head_layers", 2)) > 2:
            return FlowMatchingHead(
                hidden,
                output_dim,
                layer_config=layer_config,
                dim=int(getattr(config, "fm_head_dim")),
                layers=int(getattr(config, "fm_head_layers")),
                mlp_ratio=float(getattr(config, "fm_head_mlp_ratio")),
            )
        return nn.Sequential(
            LinearBase(hidden, 4096, layer_config=layer_config, prefix="0", bias=True),
            nn.GELU(),
            LinearBase(4096, output_dim, layer_config=layer_config, prefix="2", bias=True),
        )

    def _configure_runtime(self, config: NeoChatConfig) -> None:
        """Derive serving geometry and multimodal protocol capabilities from checkpoint config."""

        llm = config.llm_config
        vision = config.vision_config
        max_text = int(getattr(llm, "max_position_embeddings", 32768))
        max_image = max(1, int(getattr(config, "max_image_seq_len", 4096)))
        latent_downsample = int(int(vision.patch_size) * round(1 / float(config.downsample_ratio)))

        # Flow-head patch expansion determines latent params, positional
        # coordinates, and scheduler-visible capacity units.
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

        # Attention pages store rank-local heads while token and latent bounds
        # remain global scheduler-visible quantities.
        self.cache_geometry = CacheGeometry(
            num_layers=len(self.language_model.model.pipeline.layers),
            total_layers=int(llm.num_hidden_layers),
            layer_offset=self.language_model.model.pipeline.layers.start,
            num_attention_heads=local_attention_head_count(
                int(llm.num_attention_heads),
                parallel=self._parallel,
                sequence=self.language_model.model.sequence,
            ),
            num_kv_heads=local_kv_head_count(
                int(llm.num_key_value_heads),
                parallel=self._parallel,
                sequence=self.language_model.model.sequence,
            ),
            total_kv_heads=int(llm.num_key_value_heads),
            kv_head_offset=local_kv_head_offset(
                int(llm.num_key_value_heads),
                parallel=self._parallel,
                sequence=self.language_model.model.sequence,
            ),
            head_dim=int(llm.head_dim),
            dtype="bfloat16",
            store_dtype="bfloat16",
        )
        self.resource_geometry = ResourceGeometry(
            encoder_cache_entries=256,
            latent_downsample=latent_downsample,
        )

        # The advertised operation set exactly matches the state and transfer
        # transitions implemented by this runner.
        self.supported_work = frozenset(
            {
                ForwardMode.PREFILL,
                ForwardMode.DECODE,
                ForwardMode.VERIFY,
                PipelineStage.LATENT_PREPARATION,
                PipelineStage.DENOISING,
                PipelineStage.VISION_ENCODING,
                PipelineStage.IMAGE_DECODING,
                TransferMode.TENSOR,
                TransferMode.KV_PUBLISH,
                TransferMode.KV_INSTALL,
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
        """Assemble noisy image, text-conditioning, timestep, and route embeddings for flow."""

        patches = batch.flow_conditioning
        if any(value is None for value in patches):
            raise TypeError("SenseNova flow rows require patch conditioning")
        typed = tuple(value for value in patches if value is not None)
        target = batch.flow_latents[0].device
        pixels = torch.cat(tuple(value.pixels for value in typed), dim=0)
        grids = torch.cat(tuple(value.grid for value in typed), dim=0)
        local_pixels = tensor_to_device(pixels, self.generation_device)
        local_grids = tensor_to_device(grids, self.generation_device)
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
        features = tensor_to_device(features, target)
        expected = sum(batch.flow_image_tokens)
        if int(features.shape[0]) != expected:
            raise ValueError("SenseNova flow vision features do not match row geometry")

        # Expand each request timestep across its image-token span before adding
        # diffusion and optional noise-scale conditioning.
        timesteps = torch.cat(
            tuple(
                timestep.reshape(1).expand(image_tokens)
                for timestep, image_tokens in zip(
                    batch.flow_timesteps, batch.flow_image_tokens, strict=True
                )
            ),
            dim=0,
        )
        time_features = call_on_device(
            self.fm_modules["timestep_embedder"],
            timesteps,
            device=self.generation_device,
            target=target,
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
            features = features + call_on_device(
                self.fm_modules["noise_scale_embedder"],
                noise,
                device=self.generation_device,
                target=target,
            )

        # Restore the packed tower output to scheduler row identities.
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
        """Interleave text and flow embeddings in row order and run the routed decoder."""

        decode_positions: torch.Tensor | None = None
        if batch.attention.attention_mode is AttentionMode.PAGED_DECODE:
            if batch.flow_row_indices:
                raise TypeError("SenseNova paged decode accepts token rows only")
            decode_positions = positions
        if not self.language_model.model.pipeline.first:
            return self.language_model.model(None, batch, positions=decode_positions)
        assert self.language_model.model.embed_tokens is not None
        token_embeds = self.language_model.model.embed_tokens(input_ids.reshape(-1))
        if batch.input_embeddings is not None:
            if batch.embedding_mask is None:
                raise RuntimeError("SenseNova embedding input lost its selection mask")
            token_embeds = torch.where(
                batch.embedding_mask.reshape(-1, 1),
                batch.input_embeddings.to(dtype=token_embeds.dtype),
                token_embeds,
            )
        flow_embeddings = self._flow_embeddings(batch) if batch.flow_row_indices else {}

        # The decoder consumes a single packed stream; scheduler row indices retain
        # enough information to restore each modality during routing and projection.
        chunks: list[torch.Tensor | None] = [None] * batch.row_count
        token_offset = 0
        for row_index, count in zip(
            batch.token_row_indices,
            tuple(batch.attention.query_lens_cpu[index] for index in batch.token_row_indices),
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
        """Project text through the shared head and preserve flow prediction math."""

        return project_outputs(
            hidden,
            batch,
            self.language_model.lm_head,
            project_flow=lambda rows, index: self._velocity(rows, index, batch),
            pipeline=self.language_model.model.pipeline,
            vocabulary=vocabulary_partition(self.vocab_size, self._parallel),
            flow_dtype=torch.float32,
        )

    def _velocity(
        self,
        hidden: torch.Tensor,
        flow_index: int,
        context: ForwardBatch,
    ) -> torch.Tensor:
        """Select flow rows, predict patch velocity, and restore the requested image geometry."""

        target = context.flow_latents[flow_index].device
        latent = context.flow_latents[flow_index]
        image_tokens = context.flow_image_tokens[flow_index]
        image_height = context.flow_heights[flow_index]
        image_width = context.flow_widths[flow_index]
        timestep = context.flow_timesteps[flow_index]
        was_flat = latent.ndim == 2
        latent_batch = latent.unsqueeze(0) if was_flat else latent
        hidden_batch = hidden.unsqueeze(0)
        local_hidden = tensor_to_device(hidden_batch, self.generation_device)
        local_latent = tensor_to_device(latent_batch, self.generation_device)
        local_timestep = tensor_to_device(timestep.reshape(1), self.generation_device)
        batch, latent_tokens = int(local_latent.shape[0]), int(local_latent.shape[1])
        # Checkpoint metadata selects one of three equivalent prediction heads.
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
        # Convert the predicted clean sample to the flow-matching velocity used
        # by the scheduler, with a bounded denominator at the terminal endpoint.
        velocity = (predicted - local_latent) / (1 - local_timestep).clamp_min(_GENERATION_EPSILON)
        velocity = tensor_to_device(velocity, target)
        return velocity[0] if was_flat else velocity

    def encode(self, pixels: tuple[torch.Tensor, ...], batch: ForwardBatch) -> ForwardOutput:
        """Encode packed image patches and split language-width features by request."""

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
