"""DiffusionGemma text layers: Gemma-4 attention, parallel dense and MoE FFNs.

One ``Backbone`` serves both numerical roles of the model. As the causal
prompt pass it writes the K/V cache of a prompt, whose image soft tokens
attend to each other within their image; as the canvas denoiser it reads that
cache without writing it. The layer equations are identical in both roles;
only the attention inputs differ.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import PhasedLayer, TransformerDecoder
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import AttentionBatch
from uniserve.nn.functional import qk_norm_rope, sandwich_rms_norm
from uniserve.nn.linear import (
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.moe import FusedMoE, Routes, TopK
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding

from .config import LayerAttention, TextConfig


class Attention(nn.Module):
    """Gemma-4 attention with weighted Q/K norms and an unweighted V norm.

    Scores are unscaled: the Q/K norms fix their magnitude. Sliding layers
    project values; full layers have no value projection and attend with the
    normalized key projection itself, taken before the key norm and
    rotation. Both kinds rotate split halves of each head; proportional
    rotation turns only the leading fraction of the channel pairs.
    """

    def __init__(self, config: TextConfig, layer: LayerAttention, index: int):
        super().__init__()
        self.head_dim = layer.head_dim
        self.eps = config.rms_norm_eps
        heads, kv_heads = config.num_attention_heads, layer.num_kv_heads
        # Without a value branch the Q/K projection partitions key columns
        # contiguously, which keeps whole KV heads only when tensor
        # parallelism divides the full layers' KV heads.
        self.qkv = (
            QKVParallelLinear(
                config.hidden_size, heads, kv_heads, layer.head_dim, bias=False
            )
            if layer.value_projection
            else MergedColumnParallelLinear(
                config.hidden_size,
                {"q": heads * layer.head_dim, "k": kv_heads * layer.head_dim},
                bias=False,
            )
        )
        self.query_norm = RMSNorm(layer.head_dim, config.rms_norm_eps)
        self.key_norm = RMSNorm(layer.head_dim, config.rms_norm_eps)
        self.value_norm = RMSNorm(
            layer.head_dim, config.rms_norm_eps, elementwise_affine=False
        )
        self.attention = ScaledAttention(
            heads,
            kv_heads,
            layer.head_dim,
            scale=1.0,
            cache_name=f"text.backbone.layers.{index}.attention.attention",
            window=layer.window,
        )
        self.output = RowParallelLinear(
            heads * layer.head_dim, config.hidden_size, bias=False
        )
        self.rotary = RotaryEmbedding(
            layer.head_dim,
            theta=layer.rotary.theta,
            scaling=layer.rotary.scaling,
            partial_rotary_factor=layer.rotary.partial_rotary_factor,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden, positions)
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))

    def write_cache(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> None:
        """Write the K/V ``forward`` writes, without attending."""
        _, key, value = self._project(hidden, positions)
        self.attention.update_cache(key, value, attention)

    def _project(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalized, rotated ``[tokens, local heads, head_dim]`` Q, K, V."""
        # Compact [tokens, head_dim / 2] factors; proportional recipes give
        # the unrotated pairs a zero angle.
        cos, sin = self.rotary(
            positions.reshape(-1),
            dtype=torch.float32,
            sequence_length=positions.numel(),
        )

        # [tokens, local heads, head_dim] projections of this rank's heads.
        tokens = hidden.shape[0]
        projections = self.qkv(hidden)
        query = projections["q"].reshape(tokens, -1, self.head_dim)
        key = projections["k"].reshape(tokens, -1, self.head_dim)
        value = self.value_norm(
            projections["v"].reshape(tokens, -1, self.head_dim)
            if "v" in projections
            else key
        )
        query, key = qk_norm_rope(
            query,
            key,
            (self.query_norm.weight,),
            (self.key_norm.weight,),
            (cos,),
            (sin,),
            eps=self.eps,
            axis_dims=(self.head_dim,),
        )
        return query, key, value


class Router(nn.Module):
    """Select experts from the unweighted-normalized, rescaled layer stream.

    Scores are ``projection(rms(x) * scale * hidden^-1/2)``; the layer
    computes that input with its sandwich normalization (see
    :meth:`input_factors`). The top-k of their full softmax renormalize to
    one and then multiply by each selected expert's ``per_expert_scale``.
    Returns int32 expert ids and FP32 weights, both ``[tokens, top_k]``.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        size = config.hidden_size
        self.scale = nn.Parameter(torch.ones(size), requires_grad=False)
        self.projection = Linear(size, config.num_experts, bias=False)
        self.topk = TopK(config.top_k_experts, renormalize=True)
        self.per_expert_scale = nn.Parameter(
            torch.ones(config.num_experts), requires_grad=False
        )
        self.root_size = size**-0.5

    def input_factors(self) -> tuple:
        """The router input as a sandwich normalization entry: the stream
        normalized without weight, times ``scale``, times ``hidden^-1/2``.
        """  # noqa: D205
        return (None, self.scale, self.root_size)

    def forward(
        self, scaled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.topk(self.projection(scaled), scale=self.per_expert_scale)


class MoE(nn.Module):
    """Routed expert branch of a layer, with its own input and output norms.

    The router and the experts read two normalizations of the layer's
    stream, and the output norm scales the experts' sum; the layer applies
    all three in its sandwich normalizations. Every expert is a tanh-GELU
    gated MLP.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        size, eps = config.hidden_size, config.rms_norm_eps
        self.router = Router(config)
        self.input_norm = RMSNorm(size, eps)
        self.experts = FusedMoE(
            config.num_experts,
            size,
            config.moe_intermediate_size,
            top_k=config.top_k_experts,
            activation="gelu_tanh",
        )
        self.output_norm = RMSNorm(size, eps)

    def forward(self, routed: torch.Tensor, hidden: torch.Tensor) -> Routes:
        """Return the experts' unnormalized sum for the router input
        ``routed`` and the expert input ``hidden``, as the routes whose
        value it is: the layer's second sandwich normalization evaluates
        the sum while it reads them.
        """  # noqa: D205
        ids, weights = self.router(routed)
        return self.experts(hidden, ids, weights, combine=False)


class Layer(PhasedLayer[torch.Tensor, torch.Tensor]):
    """One Gemma-4 layer with sandwich norms and a trailing layer scalar.

    With ``h = x + post_attention_norm(attention(input_norm(x)))`` the layer
    returns ``(h + post_feedforward_norm(dense(h) + moe(h))) * layer_scalar``,
    where ``dense(h) = post_mlp_norm(mlp(pre_feedforward_norm(h)))`` and
    ``moe(h)`` is the MoE output norm of its experts, which read the MoE
    input norm and the router input of ``h``. Two sandwich normalizations
    evaluate everything between the attention, MLP and expert calls. The
    scalar rescales the whole stream, so the layer cannot defer its residual
    addition: it receives and returns ``residual=None`` and ``hidden`` is
    the complete stream, rounded where the reference rounds it. ``attend``
    returns ``(x, attention(input_norm(x)))``, from which ``feed_forward``
    evaluates the rest token by token, starting with the first sandwich
    normalization.
    """

    def __init__(self, config: TextConfig, index: int):
        super().__init__()
        size, eps = config.hidden_size, config.rms_norm_eps
        self.eps = eps
        self.input_norm = RMSNorm(size, eps)
        self.attention = Attention(config, config.layers[index], index)
        self.post_attention_norm = RMSNorm(size, eps)
        self.pre_feedforward_norm = RMSNorm(size, eps)
        self.mlp = GatedMLP(
            size, config.intermediate_size, activation="gelu_pytorch_tanh"
        )
        self.post_mlp_norm = RMSNorm(size, eps)
        self.moe = MoE(config)
        self.post_feedforward_norm = RMSNorm(size, eps)
        # A checkpoint value per layer, in the model's weight dtype.
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def attend(
        self,
        hidden: torch.Tensor,
        residual: None,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            raise ValueError("DiffusionGemma layers carry the complete stream")
        attended = self.attention(self.input_norm(hidden), positions, attention)
        return hidden, attended

    def feed_forward(
        self, hidden: torch.Tensor, attended: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        stream, (dense_input, expert_input, routed) = sandwich_rms_norm(
            hidden,
            ((attended, None),),
            self.post_attention_norm.weight,
            eps=self.eps,
            norms=(
                (self.pre_feedforward_norm.weight,),
                (self.moe.input_norm.weight,),
                self.moe.router.input_factors(),
            ),
            # The experts' input is stored in the encoding their first
            # projection reads (None for dense experts).
            encodings=(None, self.moe.experts.up_gate.input_quantizer),
        )

        dense = self.mlp(dense_input)
        experts = self.moe(routed, expert_input)
        stream, _ = sandwich_rms_norm(
            stream,
            (
                (dense, self.post_mlp_norm.weight),
                (experts, self.moe.output_norm.weight),
            ),
            self.post_feedforward_norm.weight,
            eps=self.eps,
            scale=self.layer_scalar,
        )
        return stream, None

    def write_cache(
        self,
        hidden: torch.Tensor,
        residual: None,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> None:
        if residual is not None:
            raise ValueError("DiffusionGemma layers carry the complete stream")
        self.attention.write_cache(
            self.input_norm(hidden), positions, attention
        )


class Backbone(TransformerDecoder):
    """Scaled token embedding, the Gemma-4 layers and the final norm.

    Layer keys are global layer indices; pipeline binding keeps the keys of
    its resident layers, so parameter paths stay aligned with checkpoint
    names.
    """

    def __init__(self, config: TextConfig):
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
            nn.ModuleDict(
                {
                    str(index): Layer(config, index)
                    for index in range(config.num_hidden_layers)
                }
            ),
            RMSNorm(config.hidden_size, config.rms_norm_eps),
            separate_residual=False,
        )
        self.embedding_scale = config.embed_scale

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and multiply by sqrt(hidden_size).

        The factor is rounded to the embedding dtype first, as the reference
        does: BF16 embeddings scale by 53.0 rather than sqrt(2816). Callers
        substitute image features after this scaling.
        """
        embedded = super().embed_input_ids(input_ids)
        scale = torch.tensor(self.embedding_scale, dtype=embedded.dtype).item()
        return embedded * scale
