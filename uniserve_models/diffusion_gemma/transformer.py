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

from uniserve.model import TransformerDecoder
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import AttentionBatch
from uniserve.nn.functional import qk_norm_rope
from uniserve.nn.linear import (
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.moe import FusedMoE, TopK
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
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))


class Router(nn.Module):
    """Select experts from the unweighted-normalized, rescaled layer stream.

    Scores are ``projection(rms(x) * scale * hidden^-1/2)``. The top-k of
    their full softmax renormalize to one and then multiply by each
    selected expert's ``per_expert_scale``. Returns int32 expert ids and FP32
    weights, both ``[tokens, top_k]``.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        size = config.hidden_size
        self.norm = RMSNorm(size, config.rms_norm_eps, elementwise_affine=False)
        self.scale = nn.Parameter(torch.ones(size), requires_grad=False)
        self.projection = Linear(size, config.num_experts, bias=False)
        self.topk = TopK(config.top_k_experts, renormalize=True)
        self.per_expert_scale = nn.Parameter(
            torch.ones(config.num_experts), requires_grad=False
        )
        self.root_size = size**-0.5

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = self.norm(hidden) * self.scale * self.root_size
        return self.topk(self.projection(scaled), scale=self.per_expert_scale)


class MoE(nn.Module):
    """Routed expert branch of a layer, with its own input and output norms.

    The router reads the layer's stream as it is; the experts read its
    normalization. Every expert is a tanh-GELU gated MLP.
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

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        ids, weights = self.router(stream)
        return self.output_norm(
            self.experts(self.input_norm(stream), ids, weights)
        )


class Layer(nn.Module):
    """One Gemma-4 layer with sandwich norms and a trailing layer scalar.

    With ``h = x + post_attention_norm(attention(input_norm(x)))`` the layer
    returns ``(h + post_feedforward_norm(dense(h) + moe(h))) * layer_scalar``,
    where ``dense(h) = post_mlp_norm(mlp(pre_feedforward_norm(h)))``. The
    scalar rescales the whole stream, so the layer cannot defer its residual
    addition: it receives and returns ``residual=None`` and ``hidden`` is
    the complete stream, rounded where the reference rounds it.
    """

    def __init__(self, config: TextConfig, index: int):
        super().__init__()
        size, eps = config.hidden_size, config.rms_norm_eps
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

    def forward(
        self,
        hidden: torch.Tensor,
        residual: None,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None:
            raise ValueError("DiffusionGemma layers carry the complete stream")

        attended = self.attention(self.input_norm(hidden), positions, attention)
        stream = hidden + self.post_attention_norm(attended)

        dense = self.post_mlp_norm(self.mlp(self.pre_feedforward_norm(stream)))
        stream = stream + self.post_feedforward_norm(dense + self.moe(stream))
        return stream * self.layer_scalar, None


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
