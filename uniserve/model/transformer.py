"""Shared transformer traversal and mathematical activation communication."""

from __future__ import annotations

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
        # Logical cache identities survive pipeline pruning. Resource owners
        # use this order to describe global K/V transfers without retaining
        # parameters or modules belonging to another pipeline stage.
        self.cache_names = tuple(
            child.cache_name
            for layer in layers.values()
            for child in layer.modules()
            if isinstance(child, Attention) and child.cache_name is not None
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
    ) -> torch.Tensor:
        """Run this stage's resident layers over the packed tokens.

        ``positions`` is ``[tokens]``, or ``[axes, tokens]`` for layers whose
        rotary embedding reads several coordinates per token. Returns the
        final stage's normalized ``[tokens, hidden]`` rows, or the hidden
        stream an earlier stage forwards.
        """
        stream = self._enter(embeddings, positions, routes)
        for layer in self.layers.values():
            stream.apply(layer, attention)
        return self._exit(stream)

    def _enter(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        routes: tuple[RouteSpan, ...],
    ) -> _Stream:
        """Take this stage's token shard of the streams entering its layers.

        The first stage starts from the packed embeddings without a residual
        stream; later stages receive both streams from the previous stage.
        """
        count = positions.shape[-1]
        if not routes and self._default_route is not None:
            routes = (RouteSpan(self._default_route, 0, count),)
        partition = TokenShard(count, self._tokens)
        positions = partition.local(positions, dim=positions.ndim - 1)

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

        # Routes replace each packed stream with its per-route tensors.
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
        return _Stream(hidden, residual, positions, partition, local_routes)

    def _exit(self, stream: _Stream) -> torch.Tensor:
        """Forward an earlier stage's streams, or normalize the final ones."""
        # Every layer returns the residual stream it carries forward.
        hidden, residual = stream.hidden, stream.residual
        assert residual is not None

        if self._pipeline.rank != self._pipeline.size - 1:
            for value in (hidden, residual):
                packed = (
                    value.packed(stream.routes)
                    if isinstance(value, RoutedTensor)
                    else value
                )
                self._pipeline.send(packed, dst=self._pipeline.rank + 1)
            return (
                hidden.packed(stream.routes)
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
            result = hidden.add(residual).apply(norm).packed(stream.routes)
        else:
            assert isinstance(residual, torch.Tensor)
            if isinstance(norm, RMSNorm):
                result, _ = add_rms_norm(
                    hidden, residual, norm.weight, norm.eps
                )
            else:
                result = norm(hidden + residual)
        return stream.partition.gather(result)


@dataclass(slots=True)
class _Stream:
    """One stage's token shard of the streams a decoder's layers carry.

    ``hidden`` and ``residual`` are packed tensors, or per-route tensors in
    a routed pass, whose shard-local spans ``routes`` holds; ``positions``
    are the shard's rotary coordinates.
    """

    hidden: torch.Tensor | RoutedTensor
    residual: torch.Tensor | RoutedTensor | None
    positions: torch.Tensor
    partition: TokenShard
    routes: tuple[RouteSpan, ...]

    def apply(self, layer: nn.Module, attention: AttentionBatch) -> None:
        """Advance both streams through one decoder layer."""
        # A routed pass carries per-route tensors even on a shard that no
        # route span reaches.
        if isinstance(self.hidden, RoutedTensor):
            self.hidden, self.residual = layer(
                self.hidden,
                self.residual,
                self.positions,
                attention,
                routes=self.routes,
            )
        else:
            self.hidden, self.residual = layer(
                self.hidden, self.residual, self.positions, attention
            )


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
