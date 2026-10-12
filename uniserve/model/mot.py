"""Mixture-of-transformers decoder layers and their checkpoint naming.

A mixture-of-transformers (MoT) backbone gives every route (``text`` and
``flow``) its own input norm, attention projections, output projection,
post-attention norm and gated MLP, while one attention call covers the packed
tokens of all routes, so every route reads and writes the same K/V cache
layer. ``RouteSpan`` values assign packed token ranges to routes; calls
without spans run every token through the first route.

``MoTLayer`` owns that composition and the residual flow. A subclass states
what is model-specific: how its projections are built and rotated
(``position_embeddings``) and the precision its MLPs consume. ``MoTDecoder``
stacks the layers under per-route output norms. ``backbone_names`` and
``absent_layers`` translate a pruned backbone to the Qwen-style checkpoint
tensor names such backbones are published under.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from uniserve.model.transformer import TransformerDecoder
from uniserve.nn.attention import Attention, AttentionBatch
from uniserve.nn.norm import RMSNorm
from uniserve.nn.routing import RoutedTensor, RouteSpan

__all__ = [
    "MoTDecoder",
    "MoTLayer",
    "ROUTES",
    "absent_layers",
    "backbone_names",
]

# The first route is the one calls without route spans take.
ROUTES = ("text", "flow")


class MoTLayer(nn.Module):
    """Run per-route experts around one shared attention call.

    ``projection``, ``output`` and ``mlp`` each build one route's module; the
    layer calls them once per route. ``attention`` is the shared module, so
    its ``cache_name`` addresses one K/V layer for every route. Experts run
    on their own tokens: Q/K/V are repacked into sequence order for the
    attention call and its result is split again for the output projections.
    With ``mlp_dtype`` the experts' MLPs consume the post-attention norm
    results cast to that dtype, as a checkpoint whose experts were trained in
    that precision expects even when the surrounding accumulation is wider.

    A subclass implements ``position_embeddings``.
    """

    def __init__(
        self,
        *,
        attention: Attention,
        hidden_size: int,
        eps: float,
        projection: Callable[[], nn.Module],
        output: Callable[[], nn.Module],
        mlp: Callable[[], nn.Module],
        mlp_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        for route in ROUTES:
            self.input_norms[route] = RMSNorm(hidden_size, eps)
            self.projections[route] = projection()
            self.outputs[route] = output()
            self.post_attention_norms[route] = RMSNorm(hidden_size, eps)
            self.mlps[route] = mlp()
        self.attention = attention
        self.mlp_dtype = mlp_dtype

    def position_embeddings(
        self, positions: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return the packed ``(cosine, sine)`` factors of every rotary axis.

        ``positions`` is ``[tokens]`` or ``[axes, tokens]`` for this rank's
        shard. Each axis contributes one ``[tokens, width]`` table, in the
        order the route projections rotate their partitions.
        """
        raise NotImplementedError

    def forward(
        self,
        hidden: RoutedTensor,
        residual: RoutedTensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: tuple[RouteSpan, ...],
    ) -> tuple[RoutedTensor, RoutedTensor]:
        """Apply one layer to this rank's routed token shard.

        ``routes`` are shard-local spans. Returns the MLP output and the
        residual stream it has not yet been added to, the pair
        ``TransformerDecoder`` carries between layers.
        """
        hidden = hidden if residual is None else hidden.add(residual)
        normalized = hidden.apply(self.input_norms)

        names = frozenset(hidden.values)
        cos, sin = (
            tuple(
                RoutedTensor.from_packed(table, routes, routes=names)
                for table in tables
            )
            for tables in self.position_embeddings(positions)
        )
        projected = {
            route: self.projections[route](
                value,
                tuple(axis.values[route] for axis in cos),
                tuple(axis.values[route] for axis in sin),
            )
            for route, value in normalized.values.items()
        }
        query, key, value = (
            RoutedTensor(
                {route: values[index] for route, values in projected.items()}
            ).packed(routes)
            for index in range(3)
        )

        attended = self.attention(query, key, value, attention).flatten(1)
        update = RoutedTensor.from_packed(attended, routes, routes=names).apply(
            self.outputs
        )
        residual = hidden.add(update)

        normalized = residual.apply(self.post_attention_norms)
        if self.mlp_dtype is not None:
            normalized = normalized.to(self.mlp_dtype)
        return normalized.apply(self.mlps), residual


class MoTDecoder(TransformerDecoder):
    """Stack ``MoTLayer`` modules over a shared embedding.

    Every route has its own output norm, and calls without route spans run
    every token through the first route.
    """

    def __init__(
        self,
        embedding: nn.Module,
        layers: nn.ModuleDict,
        *,
        hidden_size: int,
        eps: float,
    ):
        super().__init__(
            embedding,
            layers,
            nn.ModuleDict(
                {route: RMSNorm(hidden_size, eps) for route in ROUTES}
            ),
            default_route=ROUTES[0],
        )


def backbone_names(
    backbone: TransformerDecoder, *, flow_suffix: str
) -> dict[str, str]:
    """Translate backbone parameter paths to Qwen-style checkpoint names.

    Names are relative to the checkpoint's language model (``model.`` of a
    Qwen causal LM). The text route keeps the plain checkpoint names and the
    flow route's carry ``flow_suffix`` (``_moe_gen`` or ``_mot_gen``); a QK
    norm over the spatial half of the head (a second norm domain) takes
    ``_hw`` before the suffix. Only layers the backbone holds appear, so
    after pipeline partitioning that is the rank's share, and the embedding
    and output-norm entries are always present, even on a rank that does not
    hold those modules.

    Raises:
        ValueError: A layer parameter has no known checkpoint name.
    """
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": f"norm{flow_suffix}.weight",
    }

    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        # Layer paths read ``layers.<index>.<kind>.<route>.<parts...>``.
        _, index, kind, route, *parts = path.split(".")
        suffix = "" if route == ROUTES[0] else flow_suffix
        tail = ".".join(parts)

        if kind in ("input_norms", "post_attention_norms"):
            target = (
                (
                    "input_layernorm"
                    if kind == "input_norms"
                    else "post_attention_layernorm"
                )
                + suffix
                + "."
                + tail
            )
        elif kind == "outputs":
            target = f"self_attn.o_proj{suffix}.{tail}"
        elif kind == "projections":
            # ``projection.projections.<q|k|v>.<weight|bias>`` is the fused
            # QKV linear; ``<query|key>_norm[.<domain>].weight`` a QK norm.
            if parts[0] == "projection":
                target = f"self_attn.{parts[2]}_proj{suffix}.{parts[3]}"
            else:
                spatial = "_hw" if parts[1] == "1" else ""
                target = (
                    f"self_attn.{'q' if parts[0] == 'query_norm' else 'k'}"
                    f"_norm{spatial}{suffix}.{parts[-1]}"
                )
        elif kind == "mlps":
            tail = (
                tail.replace("gate_up.projections.gate", "gate_proj")
                .replace("gate_up.projections.up", "up_proj")
                .replace("down.", "down_proj.")
            )
            target = f"mlp{suffix}.{tail}"
        else:
            raise ValueError(f"unmapped MoT backbone parameter {path}")

        names[path] = f"layers.{index}.{target}"

    return names


def absent_layers(
    backbone: TransformerDecoder,
    names: Mapping[str, str],
    *,
    depth: int,
    prefix: str,
) -> set[str]:
    """Name the checkpoint tensors of the layers this backbone does not hold.

    ``names`` is ``backbone_names`` with ``prefix`` prepended to each entry
    and ``depth`` the checkpoint's layer count. Every layer has the same
    tensors, so the first resident layer's names, minus the layer index,
    enumerate those of each absent layer.
    """
    resident = f"layers.{next(iter(backbone.layers))}."
    template = tuple(
        names[target].removeprefix(prefix).split(".", 2)[-1]
        for target in names
        if target.startswith(resident)
    )
    return {
        f"{prefix}layers.{index}.{tail}"
        for index in range(depth)
        if str(index) not in backbone.layers
        for tail in template
    }
