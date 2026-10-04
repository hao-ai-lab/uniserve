"""Shared transformer traversal and mathematical activation communication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from uniserve import cache
from uniserve.cache import mha
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention import (
    Attention,
    AttentionBatch,
    AttentionParallelConfig,
)
from uniserve.nn.functional import add_rms_norm
from uniserve.nn.norm import RMSNorm
from uniserve.nn.routing import RoutedTensor, RouteSpan


@dataclass(frozen=True, slots=True)
class CacheLayer:
    """Logical K/V geometry of one cache layer, before any partitioning.

    ``window`` is the layer's attention history bound (``None`` reads the
    whole history), and ``num_kv_heads`` counts every KV head of the
    architecture. Cache layers that agree on all three share one KV cache
    group of the unit pool.
    """

    window: int | None
    num_kv_heads: int
    head_dim: int


class TransformerDecoder(nn.Module):
    """Traverse resident layers, carrying a separate residual stream.

    Layers return (hidden, residual). Pipeline peers exchange those two values
    with their original dtypes; only the final stage normalizes and gathers the
    token shards. Placement binding selects resident modules before loading.
    """

    # Pipeline binding drops the embedding outside the first stage and the
    # output norm outside the last one.
    embedding: nn.Module | None
    norm: nn.Module | None

    # Recorded by parallelize_ once the decoder is bound to its partition.
    _parallel_mesh: DeviceMesh
    _attention_parallel: AttentionParallelConfig

    def __init__(
        self,
        embedding,
        layers: nn.ModuleDict,
        norm: nn.Module,
        *,
        default_route: str | None = None,
    ):
        super().__init__()
        if not layers:
            raise ValueError(
                "a transformer decoder requires at least one layer"
            )
        self.embedding, self.layers, self.norm = embedding, layers, norm
        self.mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
        self._pipeline = Communicator()
        self._tokens = Communicator()
        self.hidden_size = embedding.embedding_dim
        self.vocab_size = embedding.num_embeddings
        # Logical cache identities and geometry survive pipeline pruning.
        # Resource owners use this order to describe global K/V transfers,
        # including each cache group's complete layer axis, without retaining
        # parameters or modules belonging to another pipeline stage.
        cached = tuple(
            child
            for layer in layers.values()
            for child in layer.modules()
            if isinstance(child, Attention) and child.cache_name is not None
        )
        self.cache_names = tuple(child.cache_name for child in cached)
        self.cache_layers = tuple(
            CacheLayer(child.window, child.num_kv_heads, child.head_dim)
            for child in cached
        )
        if default_route is not None and (
            not isinstance(norm, nn.ModuleDict) or default_route not in norm
        ):
            raise ValueError(
                "a default expert route must name one of the decoder's "
                "output norms"
            )
        self._default_route = default_route

    @property
    def cache_config(self) -> cache.Config:
        result = {}
        for layer in self.layers.values():
            # Parameters expose their logical dtype even for encoded weights.
            dtype = next(layer.parameters()).dtype
            for child in layer.modules():
                if (
                    isinstance(child, Attention)
                    and child.cache_name is not None
                ):
                    result[child.cache_name] = mha.Config(
                        child.num_kv_heads,
                        child.head_dim,
                        child.head_indices,
                        dtype,
                        window=child.window,
                    )
        return cache.Config(result)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._pipeline.rank != 0 or self.embedding is None:
            raise ValueError(
                "token embeddings belong to the first pipeline stage"
            )
        return self.embedding(input_ids)

    def forward(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: tuple[RouteSpan, ...] = (),
        deepstack: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run this stage's resident layers over the packed tokens.

        ``positions`` is ``[tokens]`` or ``[axes, tokens]``. ``deepstack``
        maps layer keys to complete ``[tokens, hidden]`` features added to
        the residual stream after that layer (DeepStack); rows that receive
        nothing hold zeros. Keys of layers on other pipeline stages are
        ignored.
        """
        count = positions.shape[-1]
        if not routes and self._default_route is not None:
            routes = (RouteSpan(self._default_route, 0, count),)
        if deepstack and routes:
            raise ValueError("DeepStack features require one packed stream")
        partition = TokenShard(count, self._tokens)
        positions = partition.local(positions, dim=positions.ndim - 1)

        # Routes replace each packed stream with its per-route tensors; the
        # first stage starts without a residual stream.
        hidden: torch.Tensor | RoutedTensor
        residual: torch.Tensor | RoutedTensor | None
        if self._pipeline.rank == 0:
            if embeddings is None or embeddings.shape != (
                count,
                self.hidden_size,
            ):
                raise ValueError(
                    "the first stage requires complete packed token embeddings"
                )
            hidden = partition.local(embeddings)
            residual = None
        else:
            reference = next(self.layers.parameters())
            hidden = reference.new_empty((partition.count, self.hidden_size))
            residual = torch.empty_like(hidden)
            for value in (hidden, residual):
                self._pipeline.recv(src=self._pipeline.rank - 1, out=value)

        # Clip route spans to this rank's token shard, in shard-local offsets.
        local_routes: tuple[RouteSpan, ...] = ()
        if routes:
            local_routes = tuple(
                RouteSpan(
                    span.route,
                    max(span.start, partition.token_slice.start)
                    - partition.token_slice.start,
                    min(span.stop, partition.token_slice.stop)
                    - max(span.start, partition.token_slice.start),
                )
                for span in routes
                if span.stop > partition.token_slice.start
                and span.start < partition.token_slice.stop
            )
            keys = frozenset(span.route for span in routes)
            hidden = RoutedTensor.from_packed(hidden, local_routes, routes=keys)
            if residual is not None:
                residual = RoutedTensor.from_packed(
                    residual, local_routes, routes=keys
                )

        for key, layer in self.layers.items():
            if routes:
                hidden, residual = layer(
                    hidden, residual, positions, attention, routes=local_routes
                )
            else:
                hidden, residual = layer(hidden, residual, positions, attention)

            features = None if deepstack is None else deepstack.get(key)
            if features is not None:
                # The reference model rounds the layer output into the
                # residual stream before adding the features, so the sum is
                # materialized first. The next normalization then reads that
                # stream against a zero update, which adds nothing.
                assert isinstance(hidden, torch.Tensor)
                assert isinstance(residual, torch.Tensor)
                residual = residual + hidden
                residual = residual + partition.local(features).to(
                    residual.dtype
                )
                hidden = torch.zeros_like(residual)
        # Every layer returns the residual stream it carries forward.
        assert residual is not None

        last = self._pipeline.rank == self._pipeline.size - 1
        if not last:
            for stream in (hidden, residual):
                packed = (
                    stream.packed(local_routes)
                    if isinstance(stream, RoutedTensor)
                    else stream
                )
                self._pipeline.send(packed, dst=self._pipeline.rank + 1)
            return (
                hidden.packed(local_routes)
                if isinstance(hidden, RoutedTensor)
                else hidden
            )

        # Fold the residual into the output norm only on the final stage,
        # which retains the norm. Layers keep both streams in one form.
        norm = self.norm
        assert norm is not None
        if isinstance(hidden, RoutedTensor):
            assert isinstance(residual, RoutedTensor)
            assert isinstance(norm, nn.ModuleDict)
            result = hidden.add(residual).apply(norm).packed(local_routes)
        else:
            assert isinstance(residual, torch.Tensor)
            if isinstance(norm, RMSNorm):
                result, _ = add_rms_norm(
                    hidden, residual, norm.weight, norm.eps
                )
            else:
                result = norm(hidden + residual)
        return partition.gather(result)


class TransformerEncoder(nn.Module):
    """Evaluate ordinary encoder layers over the same packed sequence domain."""

    def __init__(self, layers: nn.ModuleList, norm: nn.Module):
        super().__init__()
        self.layers, self.norm = layers, norm

    def forward(
        self, features: torch.Tensor, attention: AttentionBatch
    ) -> torch.Tensor:
        """Encode packed features over one uncached variable-length batch."""
        if (
            attention.queries is not None
            and attention.queries.num_tokens is not None
            and features.shape[0] != attention.queries.num_tokens
        ):
            raise ValueError(
                "encoder features must cover the declared query sequences"
            )
        for layer in self.layers:
            features = layer(features, attention)
        return self.norm(features)
