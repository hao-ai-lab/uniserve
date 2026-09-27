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


class PhasedLayer:
    """A decoder layer whose evaluation splits after its attention sublayer.

    ``forward(hidden, residual, positions, attention)`` equals
    ``feed_forward(*attend(hidden, residual, positions, attention))``.
    ``attend`` evaluates the attention sublayer, which mixes tokens, and
    returns the layer's state after it as a tuple of ``[tokens, ...]``
    tensors; ``feed_forward`` evaluates the rest of the layer from that
    state and returns its ``(hidden, residual)`` streams. The rest acts on
    each token alone, so ``feed_forward`` may evaluate any subset of the
    state's rows. ``write_cache`` evaluates the layer only up to its K/V
    cache write and returns nothing, for a pass whose only product is the
    cache. ``TransformerDecoder`` uses these phases for its final layer; a
    final layer without them runs whole.
    """

    def attend(
        self, hidden, residual, positions, attention
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def feed_forward(self, *state: torch.Tensor):
        raise NotImplementedError

    def write_cache(self, hidden, residual, positions, attention) -> None:
        raise NotImplementedError


class TransformerDecoder(nn.Module):
    """Traverse resident layers, carrying a separate residual stream.

    Layers are called as ``layer(hidden, residual, positions, attention)`` and
    return ``(hidden, residual)``. With ``separate_residual`` (the default) a
    layer may defer its residual addition: ``hidden`` is its unsummed update
    and ``residual`` the stream it carries forward, and the first layer of
    the first stage receives ``residual=None``. Without it every layer
    receives and returns ``residual=None``, and ``hidden`` is the complete
    stream; architectures whose layer output rescales the whole stream
    cannot defer the addition. Pipeline peers exchange the carried values
    with their original dtypes; only the final stage normalizes and gathers
    the token shards. Placement binding selects resident modules before
    loading.

    Besides the complete pass (``forward``), ``fill_cache`` evaluates the
    model's final layer only up to its K/V cache write
    (``PhasedLayer.write_cache``), for prompts whose only product is the
    cache. Every other layer, and every layer of an earlier pipeline stage,
    runs whole.
    """

    # Pipeline binding drops the embedding outside the first stage and the
    # output norm outside the last one.
    embedding: nn.Module | None
    norm: nn.Module | None
    # Factor ``embed_input_ids`` multiplies embedding-table rows by; an
    # architecture that scales its token embeddings overrides it.
    embedding_scale: float = 1.0

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
        separate_residual: bool = True,
    ):
        super().__init__()
        if not layers:
            raise ValueError(
                "a transformer decoder requires at least one layer"
            )
        self.embedding, self.layers, self.norm = embedding, layers, norm
        self.separate_residual = separate_residual
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
    ) -> torch.Tensor:
        """Evaluate every layer and, on the final stage, the output norm.

        Returns the final stage's normalized ``[tokens, hidden]`` rows, or
        the activations an earlier stage forwards.
        """
        traversal = self._traverse(
            embeddings, positions, attention, routes, split=False
        )
        hidden, residual, local_routes = (
            traversal.hidden,
            traversal.residual,
            traversal.routes,
        )
        if self._pipeline.rank != self._pipeline.size - 1:
            return self._forward_stage(hidden, residual, local_routes)
        return traversal.partition.gather(
            self._normalize(hidden, residual, local_routes)
        )

    def fill_cache(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: tuple[RouteSpan, ...] = (),
    ) -> None:
        """Write every layer's K/V cache without evaluating any output.

        The final layer stops at its K/V write (``PhasedLayer.write_cache``)
        and the output norm is skipped; earlier stages forward their
        activations as ``forward`` does. The cache holds exactly what
        ``forward`` writes.
        """
        traversal = self._traverse(
            embeddings, positions, attention, routes, split=True
        )
        if self._pipeline.rank != self._pipeline.size - 1:
            self._forward_stage(
                traversal.hidden, traversal.residual, traversal.routes
            )
        elif traversal.final is not None:
            traversal.final.write_cache(
                traversal.hidden,
                traversal.residual,
                traversal.positions,
                attention,
            )

    def _traverse(self, embeddings, positions, attention, routes, *, split):
        """Evaluate the resident layers, all but a split final layer.

        With ``split``, an unrouted pass leaves the model's final layer on
        the final pipeline stage unevaluated when it is a ``PhasedLayer``,
        returning it as ``final`` with the streams it would receive.
        """
        count = positions.shape[-1]
        if not routes and self._default_route is not None:
            routes = (RouteSpan(self._default_route, 0, count),)
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
            residual = (
                torch.empty_like(hidden) if self.separate_residual else None
            )
            for value in (hidden, residual):
                if value is not None:
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

        layers = tuple(self.layers.values())
        final = None
        if (
            split
            and not routes
            and self._pipeline.rank == self._pipeline.size - 1
            and isinstance(layers[-1], PhasedLayer)
        ):
            final, layers = layers[-1], layers[:-1]
        for layer in layers:
            if routes:
                hidden, residual = layer(
                    hidden, residual, positions, attention, routes=local_routes
                )
            else:
                hidden, residual = layer(hidden, residual, positions, attention)
        # Layers that defer their residual addition return the stream they
        # carry forward; complete-stream layers return none.
        if final is None and (residual is None) == self.separate_residual:
            raise ValueError(
                "decoder layers must return a residual stream exactly when "
                "the decoder carries one separately"
            )
        return _Traversal(
            hidden, residual, positions, partition, local_routes, final
        )

    def _forward_stage(self, hidden, residual, local_routes):
        """Send an earlier stage's streams on and return its hidden stream."""
        for stream in (hidden, residual):
            if stream is None:
                continue
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

    def _normalize(self, hidden, residual, local_routes):
        """Fold the residual into the final stage's output norm."""
        # Only the final stage retains the norm. Layers keep both streams in
        # one form.
        norm = self.norm
        assert norm is not None
        if residual is None:
            result = (
                hidden.apply(norm).packed(local_routes)
                if isinstance(hidden, RoutedTensor)
                else norm(hidden)
            )
        elif isinstance(hidden, RoutedTensor):
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
        return result


@dataclass(frozen=True, slots=True)
class _Traversal:
    """The streams one ``TransformerDecoder._traverse`` leaves, and how.

    ``positions`` and ``routes`` are local to this rank's token shard of
    ``partition``; ``final`` is the unevaluated final layer of a split pass.
    """

    hidden: torch.Tensor | RoutedTensor
    residual: torch.Tensor | RoutedTensor | None
    positions: torch.Tensor
    partition: TokenShard
    routes: tuple[RouteSpan, ...]
    final: PhasedLayer | None


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
