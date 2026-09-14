"""SenseNova-U1 numerical composition and neural equations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from uniserve.attention.metadata import AttentionMetadata, AttentionMode, ExpertRoute, RouteSpan
from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.loading.component import construct_owned_module
from uniserve.loading.mapping import WeightNameMap
from uniserve.model.batch import DiffusionBatch, TensorOutput
from uniserve.model.components import ComponentCall
from uniserve.model.decoder import DecoderMixin
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.encoder import EncodeKind, EncoderMixin
from uniserve.model.image_diffusion import (
    BranchSource,
    ImageDiffusion,
    LatentLayout,
    NoiseScaleMode,
)
from uniserve.model.limits import ModelLimits
from uniserve.model.media import ImageSize
from uniserve.model.model import Model
from uniserve.model.tensors import FlowPatches, PositionLayout, TensorViews
from uniserve.model.text import TextMixin
from uniserve.nn.attention import RadixAttention
from uniserve.nn.branch import branch
from uniserve.nn.decoder.base import Decoder
from uniserve.nn.diffusion import (
    ConvDecoder,
    FlowMatchingHead,
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from uniserve.nn.diffusion.cfg import CfgRecipe
from uniserve.nn.diffusion.fm_modules import FlowHeadConfig
from uniserve.nn.diffusion.integrator import EulerSolver
from uniserve.nn.diffusion.prediction import ImageVelocity
from uniserve.nn.expert_routing import RoutedTensor, slice_route_spans
from uniserve.nn.layer import LayerConfig
from uniserve.nn.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
    local_kv_head_count,
    local_kv_head_offset,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.parallel_sequence import SequencePartition
from uniserve.nn.qkv import QKV
from uniserve.nn.rope import HFRotaryEmbedding, RotaryEmbedding, get_rope, qk_norm_rope
from uniserve.nn.row_pipeline import (
    RowStage,
    RowTensors,
    RowTensorSegments,
    independent_linear_rows,
    packed_row_stage,
)
from uniserve.nn.shard import WeightMode
from uniserve.nn.vae.patch import RgbDecoder
from uniserve.nn.vision import NeoVitConfig, NeoVitEncoder
from uniserve.nn.vocab_parallel_embedding import ParallelLMHead, vocabulary_partition
from uniserve.runtime.kv_cache import KVCacheConfig
from uniserve_models.sensenova.config import FlowConfig, NeoChatConfig, NeoLlmConfig

if TYPE_CHECKING:
    from uniserve.loading.component import CheckpointComponent


__all__ = ["NEOChatModel"]

_MAX_VISION_TOKENS = 70 * 70
_MAX_CFG_BRANCHES = 3
_GENERATION_EPSILON = 0.02


_STACKED_WEIGHTS: WeightNameMap = (
    ("flow_qkv.projection", "q_proj_mot_gen", "q"),
    ("flow_qkv.projection", "k_proj_mot_gen", "k"),
    ("flow_qkv.projection", "v_proj_mot_gen", "v"),
    ("flow_qkv.projection", "qkv_proj_mot_gen", None),
    ("flow_qkv.query_norm", "q_norm_mot_gen", None),
    ("flow_qkv.key_norm", "k_norm_mot_gen", None),
    ("flow_qkv.query_norm_hw", "q_norm_hw_mot_gen", None),
    ("flow_qkv.key_norm_hw", "k_norm_hw_mot_gen", None),
    ("text_qkv.projection", "q_proj", "q"),
    ("text_qkv.projection", "k_proj", "k"),
    ("text_qkv.projection", "v_proj", "v"),
    ("fm_modules.velocity.head", "fm_modules.fm_head", None),
    ("text_qkv.projection", "qkv_proj", None),
    ("text_qkv.query_norm", "q_norm", None),
    ("text_qkv.key_norm", "k_norm", None),
    ("text_qkv.query_norm_hw", "q_norm_hw", None),
    ("text_qkv.key_norm_hw", "k_norm_hw", None),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


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

    def __init__(self, config: NeoVitConfig) -> None:
        """Build the vision encoder that projects patches directly to language width."""

        super().__init__()
        self.embeddings = NeoVitEncoder(config)

    def forward(
        self,
        pixels: torch.Tensor,
        grid: torch.Tensor,
        *,
        grid_shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        """Embed flattened vision patches and add their grid-derived positions."""

        return self.embeddings(pixels, grid, grid_shapes=grid_shapes)


class _AxisQKV(QKV):
    """Normalize temporal and spatial head partitions with their checkpoint scales."""

    def __init__(self, projection: QKVParallelLinear, epsilon: float, *, separate: bool) -> None:
        super().__init__(projection, separate=separate)
        half = self.head_dim // 2
        self.query_norm = RMSNorm(half, eps=epsilon)
        self.query_norm_hw = RMSNorm(half, eps=epsilon)
        self.key_norm = RMSNorm(half, eps=epsilon)
        self.key_norm_hw = RMSNorm(half, eps=epsilon)

    def normalize(self, query, key, cos, sin):
        return qk_norm_rope(
            query,
            key,
            (self.query_norm.weight, self.query_norm_hw.weight, self.query_norm_hw.weight),
            (self.key_norm.weight, self.key_norm_hw.weight, self.key_norm_hw.weight),
            cos,
            sin,
            self.query_norm.eps,
            axis_dims=(self.head_dim // 2, self.head_dim // 4, self.head_dim // 4),
        )


class _SenseAttention(nn.Module):
    """SenseNova dual-expert QKV projection over one explicit attention plan."""

    def __init__(
        self,
        config: NeoLlmConfig,
        layer: int,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Build text and flow projection towers around one shared attention backend."""

        super().__init__()
        hidden_size = int(config.hidden_size)
        total_heads = int(config.num_attention_heads)
        total_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim**-0.5
        bias = bool(config.attention_bias)
        epsilon = float(config.rms_norm_eps)
        query_width = total_heads * self.head_dim

        text_projection = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config,
            prefix="qkv_proj",
            bias=bias,
        )
        flow_projection = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config,
            prefix="qkv_proj_mot_gen",
            packed_names=("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen"),
            bias=bias,
        )
        self.num_heads = int(text_projection.output_sizes[0]) // self.head_dim
        self.num_kv_heads = int(text_projection.output_sizes[1]) // self.head_dim
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
        self.o_proj_mot_gen = branch(
            RowParallelLinear(
                query_width,
                hidden_size,
                layer_config=layer_config,
                prefix="o_proj_mot_gen",
                bias=bias,
            ),
            ExpertRoute.FLOW,
        )

        half = self.head_dim // 2
        self.text_qkv = _AxisQKV(text_projection, epsilon, separate=False)
        self.flow_qkv = branch(_AxisQKV(flow_projection, epsilon, separate=True), ExpertRoute.FLOW)

        self.rotary_emb = get_rope(
            half,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )
        self.rotary_emb_hw = get_rope(
            self.head_dim // 4,
            theta=config.rope_theta_hw,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings_hw,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )

    def rope(self, indexes: torch.Tensor) -> _PackedRope:
        """Build temporal, height, and width rotary tables for ``[3, tokens]`` indexes."""

        if indexes.ndim != 2 or tuple(indexes.shape[:1]) != (3,):
            raise ValueError("SenseNova positions must have shape [3, tokens]")

        def frequencies(
            module: RotaryEmbedding | HFRotaryEmbedding,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return module.cos_sin_1d(positions)

        cos_t, sin_t = frequencies(self.rotary_emb, indexes[0])
        cos_h, sin_h = frequencies(self.rotary_emb_hw, indexes[1])
        cos_w, sin_w = frequencies(self.rotary_emb_hw, indexes[2])
        return _PackedRope((cos_t, cos_h, cos_w), (sin_t, sin_h, sin_w))

    def project_rows(
        self,
        hidden: RoutedTensor,
        *,
        context: AttentionMetadata,
        spans: tuple[RouteSpan, ...],
        rope: _RoutedRope,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare routed normalized QKV without consuming global attention."""

        text_projection = (
            None
            if hidden.text is None or rope.text is None
            else self.text_qkv(hidden.text, rope.text.cos, rope.text.sin)
        )
        flow_projection = (
            None
            if hidden.flow is None or rope.flow is None
            else self.flow_qkv(hidden.flow, rope.flow.cos, rope.flow.sin)
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
    ) -> None:
        """Assemble shared attention with route-specific norms and feed-forward towers."""

        super().__init__()
        hidden = int(config.hidden_size)
        epsilon = float(config.rms_norm_eps)
        self.self_attn = _SenseAttention(
            config,
            layer,
            layer_config=layer_config.child("self_attn"),
        )
        self.mlp = GatedMLP(
            hidden,
            int(config.intermediate_size),
            hidden_act=config.hidden_act,
            layer_config=layer_config.child("mlp"),
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.mlp_mot_gen = branch(
            GatedMLP(
                hidden,
                int(config.intermediate_size),
                hidden_act=config.hidden_act,
                layer_config=layer_config.child("mlp_mot_gen"),
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
            ),
            ExpertRoute.FLOW,
        )
        self.input_layernorm = RMSNorm(hidden, eps=epsilon)
        self.input_layernorm_mot_gen = branch(RMSNorm(hidden, eps=epsilon), ExpertRoute.FLOW)
        self.post_attention_layernorm = RMSNorm(hidden, eps=epsilon)
        self.post_attention_layernorm_mot_gen = branch(
            RMSNorm(hidden, eps=epsilon), ExpertRoute.FLOW
        )

    def row_stage(
        self,
        *,
        context: AttentionMetadata,
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
            )
            hidden = hidden.add(projected)
            normalized = hidden.apply(
                text=self.post_attention_layernorm,
                flow=self.post_attention_layernorm_mot_gen,
            )
            feed_forward = normalized.apply(text=self.mlp, flow=self.mlp_mot_gen)
            return (hidden.add(feed_forward).packed(local_spans),)

        return packed_row_stage(
            project,
            self.self_attn.attention,
            finish,
            context=context,
            partition=partition,
            causal=causal,
            scale=self.self_attn.scaling,
            independent_input=independent_linear_rows(
                self.self_attn.text_qkv.projection, self.self_attn.flow_qkv.projection
            ),
            independent_output=independent_output,
        )


class _SenseDecoder(Decoder):
    """One packed text/flow decoder with no serving state."""

    def __init__(
        self,
        config: NeoLlmConfig,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Compose resident text and flow layers with their numerical partitioning."""

        hidden = int(config.hidden_size)
        super().__init__(
            hidden,
            int(config.vocab_size),
            int(config.num_hidden_layers),
            layer_config=layer_config,
            max_tokens=config.max_position_embeddings,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleDict(
            {
                str(index): _SenseLayer(
                    config,
                    index - self.pipeline.layers.start,
                    layer_config=layer_config.child(f"layers.{index}"),
                )
                for index in self.pipeline.layers
            }
        )
        epsilon = float(config.rms_norm_eps)
        self.norm = RMSNorm(hidden, eps=epsilon) if self.pipeline.last else None
        self.norm_mot_gen = (
            branch(RMSNorm(hidden, eps=epsilon), ExpertRoute.FLOW) if self.pipeline.last else None
        )
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
        context: AttentionMetadata,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode packed routed rows and return final-normalized states in input order."""

        first = cast(_SenseLayer, next(iter(self.layers.values())))
        if inputs is not None:
            token_count = int(inputs.shape[0])
        elif (
            context.attention_mode is AttentionMode.PACKED and context.attention_indexes is not None
        ):
            token_count = int(context.attention_indexes.shape[1])
        elif context.attention_mode is AttentionMode.PAGED_DECODE and positions is not None:
            token_count = positions.numel()
        else:
            raise ValueError("pipeline input requires packed or decode row geometry")
        partition = SequencePartition(token_count, self.sequence)
        if inputs is not None and inputs.ndim != 2:
            raise ValueError("decoder inputs must have shape [tokens, hidden]")
        values = self.receive(
            inputs,
            token_count,
            reference=first.input_layernorm.weight,
            partition=partition,
        )
        spans: tuple[RouteSpan, ...]
        indexes: torch.Tensor
        causal: bool
        if context.attention_mode is AttentionMode.PACKED:
            if context.attention_indexes is None or tuple(context.attention_indexes.shape) != (
                3,
                token_count,
            ):
                raise ValueError("SenseNova positions must have shape [3, tokens]")
            spans = context.route_spans
            indexes = context.attention_indexes
            causal = False
        elif context.attention_mode is AttentionMode.PAGED_DECODE:
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
        (packed,) = self.run_layers(
            values,
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
        )
        if not self.pipeline.last:
            return packed
        hidden = RoutedTensor.from_packed(packed, spans, routes=routes)
        assert self.norm is not None and self.norm_mot_gen is not None
        return partition.gather(hidden.apply(text=self.norm, flow=self.norm_mot_gen).packed(spans))


class _LanguageModel(nn.Module):
    """Owns the SenseNova token embedding, decoder, normalization, and vocabulary projection stack."""

    def __init__(
        self,
        config: NeoLlmConfig,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Build the decoder and tensor-parallel vocabulary projection."""

        super().__init__()
        self.model = _SenseDecoder(config, layer_config=layer_config.child("model"))
        self.lm_head = (
            ParallelLMHead(
                int(config.hidden_size),
                int(config.vocab_size),
                layer_config=layer_config,
                prefix="lm_head",
                bias=False,
            )
            if self.model.pipeline.last
            else None
        )
        if config.tie_word_embeddings and self.model.pipeline.first and self.model.pipeline.last:
            assert self.lm_head is not None and self.model.embed_tokens is not None
            self.lm_head.weight = self.model.embed_tokens.weight
        self.nonresident_parameters = frozenset(
            f"model.{name}" for name in self.model.nonresident_parameters
        )
        if not self.model.pipeline.last:
            self.nonresident_parameters |= {"lm_head.weight"}


class _ImageHead(ImageVelocity):
    """Convert the configured clean-sample head to image-patch velocity."""

    def __init__(
        self, head: nn.Module, config: FlowConfig, *, patch_size: int, downsample_ratio: float
    ) -> None:
        super().__init__(head, epsilon=_GENERATION_EPSILON)
        self._use_pixel_head = bool(config.use_pixel_head)
        self._use_deep_head = bool(config.head.layers > 2)
        self._downsample_ratio = downsample_ratio
        self._patch_size = patch_size

    def predict(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_tokens: int,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        batch, latent_tokens = int(latent.shape[0]), int(latent.shape[1])
        # Checkpoint metadata selects the prediction head's mathematical layout.
        if self._use_pixel_head:
            merge = int(1 / self._downsample_ratio)
            token_height = image_height // (self._patch_size * merge)
            token_width = image_width // (self._patch_size * merge)
            image = hidden[:, -image_tokens:].view(
                batch,
                token_height,
                token_width,
                -1,
            )
            image = torch.einsum("b h w c -> b c h w", image).contiguous()
            predicted = self.head(image)
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
            predicted = self.head(
                hidden[:, -image_tokens:].reshape(batch * latent_tokens, -1),
                timestep.repeat(batch * latent_tokens),
            ).view(batch, latent_tokens, -1)
        else:
            predicted = self.head(hidden[:, -image_tokens:].view(batch, latent_tokens, -1)).view(
                batch, latent_tokens, -1
            )
        return predicted


class NEOChatModel(TextMixin, EncoderMixin, DiffusionMixin[ImageSize], DecoderMixin, Model):
    """Concrete stateless SenseNova model for mixed text, flow, and vision rows."""

    config: NeoChatConfig

    @classmethod
    def component_calls(cls, config: object) -> tuple[ComponentCall, ...]:
        """Declare actual numerical methods and their mathematical participation."""

        return (
            ComponentCall("", "forward", groups=("tp", "sp", "pp")),
            ComponentCall("", "forward_diffusion", groups=("tp", "sp", "pp")),
            ComponentCall("", "encode:vision"),
            ComponentCall("", "decode:image"),
        )

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare checkpoint projection mappings and nonresident parameters."""

        from uniserve.loading.component import CheckpointComponent

        mapping = _STACKED_WEIGHTS
        if self.config.text.tie_word_embeddings:
            pipeline = self.language_model.model.pipeline
            embedding = "language_model.model.embed_tokens.weight"
            projection = "language_model.lm_head.weight"
            # PP endpoints load the same checkpoint embedding onto their own
            # numerical partition; a local embedding/head shares one Parameter.
            if pipeline.first:
                mapping += ((embedding, projection, None),)
            elif pipeline.last:
                mapping += ((projection, embedding, None),)
        return (
            CheckpointComponent(
                self,
                weight_name_map=mapping,
                nonresident=self.nonresident_parameters,
            ),
        )

    def __init__(
        self,
        config: NeoChatConfig,
        *,
        parallel: Mapping[str, ParallelConfig],
        meshes: Mapping[str, DeviceMesh],
        layers: Mapping[str, LayerConfig],
        limits: ModelLimits,
    ) -> None:
        """Compose SenseNova numerical modules with their bound mathematical layers."""

        super().__init__(config)
        layer_config = layers[""]
        vision = config.vision
        hidden = int(config.text.hidden_size)
        self.vision_model = _VisionModel(vision)
        self._parallel = layer_config.communicator
        self.language_model = _LanguageModel(
            config.text,
            layer_config=layer_config.child("language_model"),
        )
        self.nonresident_parameters = frozenset(
            f"language_model.{name}" for name in self.language_model.nonresident_parameters
        )
        self._patch_size = int(vision.patch_size)
        self._downsample_ratio = float(config.vision.downsample_ratio)
        self._use_pixel_head = bool(config.flow.use_pixel_head)
        self._add_noise_embedding = bool(config.flow.add_noise_scale_embedding)
        self._noise_scale_max = float(config.flow.noise_scale_max_value)
        pipeline = self.language_model.model.pipeline
        declarations = [
            ("vision_model_mot_gen", pipeline.first, lambda: _VisionModel(vision)),
            ("timestep_embedder", pipeline.first, lambda: TimestepEmbedder(hidden)),
            (
                "velocity",
                pipeline.last,
                lambda: _ImageHead(
                    (
                        ConvDecoder(hidden)
                        if self._use_pixel_head
                        else self._flow_head(
                            config.flow.head,
                            hidden,
                            3 * (vision.patch_size * round(1 / vision.downsample_ratio)) ** 2,
                            layer_config.child("fm_modules.fm_head"),
                        )
                    ),
                    config.flow,
                    patch_size=vision.patch_size,
                    downsample_ratio=vision.downsample_ratio,
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
                modules[name] = branch(module, ExpertRoute.FLOW)
            self.nonresident_parameters |= {
                f"fm_modules.{name}.{parameter}" for parameter in nonresident
            }
        self.fm_modules = nn.ModuleDict(modules)
        llm = config.text
        vision = config.vision
        max_text = llm.max_position_embeddings
        max_image = config.max_image_seq_len
        latent_downsample = int(
            int(vision.patch_size) * round(1 / float(config.vision.downsample_ratio))
        )

        # Flow-head patch expansion determines latent params, positional
        # coordinates, and per-image sequence bounds.
        self.architecture = "NEOChatModel"
        self.solver: EulerSolver = EulerSolver()
        self.generation = ImageDiffusion(
            latent_downsample=latent_downsample,
            prediction_dtype=torch.float32,
            schedule_direction=ScheduleDirection.ASCENDING,
            schedule_shift_domain=ScheduleShiftDomain.SIGMA,
            max_latent_tokens=max_image,
            max_vae_grid_tokens=max_image,
            marker_tokens=2,
            rope_advance=2,
            max_cfg_branches=_MAX_CFG_BRANCHES,
            latent_layout=LatentLayout.IMAGE_NCHW,
            latent_channels=3,
            latent_patch_size=latent_downsample,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            noise_scale=float(config.flow.noise_scale),
            noise_scale_mode=NoiseScaleMode(config.flow.noise_scale_mode),
            noise_scale_base_tokens=float(config.flow.noise_scale_base_image_seq_len),
            noise_scale_maximum=float(config.flow.noise_scale_max_value),
            text_unconditional=BranchSource.NEGATIVE_OR_START,
            image_unconditional=BranchSource.START,
            cfg_recipe=CfgRecipe.ADDITIVE_DELTAS,
        )
        self.image_decoder = RgbDecoder(self.generation.latent_patch_size)

        # Attention pages store rank-local heads while token and latent bounds
        # remain global scheduler-visible quantities.
        self.text_backbone.cache_config = KVCacheConfig(
            num_layers=len(self.language_model.model.pipeline.layers),
            total_layers=int(llm.num_hidden_layers),
            layer_offset=self.language_model.model.pipeline.layers.start,
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
            dtype=torch.bfloat16,
            store_dtype=torch.bfloat16,
        )

        self.max_vit_grid_tokens = _MAX_VISION_TOKENS
        self.text_backbone.max_tokens = max(max_text, max_image)
        self.text_backbone.attention_mode = AttentionMode.PACKED

    @staticmethod
    def _flow_head(
        config: FlowHeadConfig,
        hidden: int,
        output_dim: int,
        layer_config: LayerConfig,
    ) -> nn.Module:
        """Build the configured shallow or deep patch-space flow prediction head."""

        if int(config.layers) > 2:
            return FlowMatchingHead(
                hidden,
                output_dim,
                layer_config=layer_config,
                config=config,
            )
        return nn.Sequential(
            LinearBase(hidden, config.dim, layer_config=layer_config, prefix="0", bias=True),
            nn.GELU(),
            LinearBase(config.dim, output_dim, layer_config=layer_config, prefix="2", bias=True),
        )

    def _flow_embeddings(
        self,
        batch: DiffusionBatch[ImageSize],
    ) -> torch.Tensor:
        """Assemble noisy image, text-conditioning, timestep, and route embeddings for flow."""

        patches = batch.conditioning["image"]
        if any(not isinstance(value, FlowPatches) for value in patches):
            raise TypeError("SenseNova flow rows require patch conditioning")
        typed = tuple(value for value in patches if isinstance(value, FlowPatches))
        pixels = torch.cat(tuple(value.pixels for value in typed), dim=0)
        grids = torch.cat(tuple(value.grid for value in typed), dim=0)
        tower = self.fm_modules["vision_model_mot_gen"]
        feature_dtype = next(tower.parameters()).dtype
        # Each flow row's image patch grid (height/patch, width/patch) is known
        # on the host from its registered image size. Passing the per-row grids
        # lets the tower resolve the conv geometry without reading the grid
        # tensor back, keeping the flow forward capturable in a CUDA graph.
        patch = self._patch_size
        grid_shapes = tuple((shape.height // patch, shape.width // patch) for shape in batch.sizes)
        features = tower(pixels.to(dtype=feature_dtype), grids, grid_shapes=grid_shapes)
        if not isinstance(features, torch.Tensor):
            raise TypeError("SenseNova flow vision tower must return a tensor")
        expected = sum(batch.sequence_lengths)
        if int(features.shape[0]) != expected:
            raise ValueError("SenseNova flow vision features do not match row geometry")

        # Expand each request timestep across its image-token span before adding
        # diffusion and optional noise-scale conditioning.
        timesteps = torch.cat(
            tuple(
                timestep.reshape(1).expand(image_tokens)
                for timestep, image_tokens in zip(
                    batch.timesteps["image"], batch.sequence_lengths, strict=True
                )
            ),
            dim=0,
        )
        time_features = self.fm_modules["timestep_embedder"](timesteps)
        features = features + time_features
        if self._add_noise_embedding:
            noise = torch.cat(
                tuple(
                    value.noise_scale.reshape(1).expand(image_tokens)
                    for image_tokens, value in zip(batch.sequence_lengths, typed, strict=True)
                ),
                dim=0,
            )
            noise = noise / self._noise_scale_max
            features = features + self.fm_modules["noise_scale_embedder"](noise)

        return features

    @property
    def text_backbone(self):
        return self.language_model.model

    @property
    def vocabulary(self):
        return vocabulary_partition(self.config.text.vocab_size, self._parallel)

    @property
    def lm_head(self):
        return self.language_model.lm_head

    @property
    def diffusion_pipeline(self):
        return self.language_model.model.pipeline

    def forward_diffusion(
        self,
        batch: DiffusionBatch[ImageSize],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Predict image velocity for homogeneous latent and CFG branch rows."""

        pipeline = self.diffusion_pipeline
        embeddings = self._flow_embeddings(batch) if pipeline.first else None
        hidden = self.language_model.model(embeddings, batch.attention)
        if not pipeline.last:
            return TensorOutput({"image": (None,) * batch.row_count})
        rows = hidden[: sum(batch.sequence_lengths)].split(batch.sequence_lengths)
        return TensorOutput(
            {"image": tuple(self._velocity(row, index, batch) for index, row in enumerate(rows))}
        )

    def _velocity(
        self,
        hidden: torch.Tensor,
        flow_index: int,
        context: DiffusionBatch[ImageSize],
    ) -> torch.Tensor:
        """Select flow rows, predict patch velocity, and restore the requested image geometry."""

        latent = context.latents["image"][flow_index]
        was_flat = latent.ndim == 2
        velocity = self.fm_modules["velocity"](
            latent.unsqueeze(0) if was_flat else latent,
            hidden.unsqueeze(0),
            context.timesteps["image"][flow_index].reshape(1),
            image_tokens=context.sequence_lengths[flow_index],
            image_height=context.sizes[flow_index].height,
            image_width=context.sizes[flow_index].width,
        )
        return velocity[0] if was_flat else velocity

    encoder_kinds: frozenset[EncodeKind] = frozenset({"vision"})

    @property
    def vision_encoder(self) -> NeoVitEncoder:
        """Return the packed-patch encoder shared by numerical vision calls."""

        return self.vision_model.embeddings
