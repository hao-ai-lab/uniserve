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


class PhasedLayer(nn.Module):
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

    def forward(self, hidden, residual, positions, attention):
        """Compose token-mixing attention and the per-token remainder."""
        return self.feed_forward(
            *self.attend(hidden, residual, positions, attention)
        )

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

    Besides the complete pass (``forward``), two partial passes evaluate the
    model's final layer in its ``PhasedLayer`` phases: ``fill_cache`` stops
    at the final layer's K/V cache write, for prompts whose only product is
    the cache, and ``attend`` stops after the final layer's attention, so
    ``finish`` completes the layer and the output norm for selected rows
    alone. Every other layer, and every layer of an earlier pipeline stage,
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

        ``positions`` is ``[tokens]``, or ``[axes, tokens]`` for layers whose
        rotary embedding reads several coordinates per token. Returns the
        final stage's normalized ``[tokens, hidden]`` rows, or the
        activations an earlier stage forwards.
        """
        stream = self._enter(embeddings, positions, routes)
        for layer in self.layers.values():
            stream.apply(layer, attention)
        return self._exit(stream)

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
        stream = self._enter(embeddings, positions, routes)
        leading, final = self._phases(stream)
        for layer in leading:
            stream.apply(layer, attention)
        if self._pipeline.rank != self._pipeline.size - 1:
            self._check_streams(stream)
            self._forward_stage(stream.hidden, stream.residual, stream.routes)
        elif final is not None:
            final.write_cache(
                stream.hidden, stream.residual, stream.positions, attention
            )
        else:
            self._check_streams(stream)

    def attend(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate every layer up to the final layer's attention output.

        Returns the final stage's per-token state of every token for
        ``finish``: the final layer's ``PhasedLayer.attend`` state, or, when
        that layer does not split, its ``(hidden, residual)`` streams
        (``residual`` omitted for complete-stream layers). An earlier stage
        returns the activations it forwards. Routed passes are not split.
        """
        if self._default_route is not None:
            raise ValueError("routed decoders evaluate their layers whole")
        stream = self._enter(embeddings, positions, ())
        leading, final = self._phases(stream)
        for layer in leading:
            stream.apply(layer, attention)
        hidden, residual = stream.hidden, stream.residual
        if final is None:
            self._check_streams(stream)
        if self._pipeline.rank != self._pipeline.size - 1:
            return (self._forward_stage(hidden, residual, ()),)
        if final is not None:
            state = final.attend(hidden, residual, stream.positions, attention)
        else:
            state = (hidden,) if residual is None else (hidden, residual)
        return tuple(stream.partition.gather(value) for value in state)

    def finish(
        self, state: tuple[torch.Tensor, ...], rows: torch.Tensor
    ) -> torch.Tensor:
        """Complete the final layer and the output norm for ``rows`` alone.

        ``state`` is ``attend``'s per-token state on the final stage and
        ``rows`` the int64 token rows to evaluate. Returns their normalized
        ``[rows, hidden]`` outputs, which equal the same rows of ``forward``
        up to the rounding that fewer rows allow.
        """
        if self._pipeline.rank != self._pipeline.size - 1:
            raise ValueError("the output norm belongs to the final stage")
        selected = tuple(value.index_select(0, rows) for value in state)
        layer = next(reversed(self.layers.values()))
        if isinstance(layer, PhasedLayer):
            hidden, residual = layer.feed_forward(*selected)
        else:
            hidden, residual = selected[0], (selected[1:] or (None,))[0]
        return self._normalize(hidden, residual, ())

    def _enter(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        routes: tuple[RouteSpan, ...],
    ) -> _Stream:
        """Take this stage's token shard of the streams entering its layers.

        The first stage starts from the packed embeddings without a residual
        stream; later stages receive the streams the previous stage carries.
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
            residual = (
                torch.empty_like(hidden) if self.separate_residual else None
            )
            for value in (hidden, residual):
                if value is not None:
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

    def _phases(
        self, stream: _Stream
    ) -> tuple[tuple[nn.Module, ...], PhasedLayer | None]:
        """Split off the final layer a partial pass evaluates in phases.

        An unrouted pass on the final pipeline stage leaves the model's
        final layer to its ``PhasedLayer`` phases; every other layer, and
        every layer of a routed pass or an earlier stage, runs whole.
        """
        layers = tuple(self.layers.values())
        if (
            not isinstance(stream.hidden, RoutedTensor)
            and self._pipeline.rank == self._pipeline.size - 1
            and isinstance(layers[-1], PhasedLayer)
        ):
            return layers[:-1], layers[-1]
        return layers, None

    def _check_streams(self, stream: _Stream) -> None:
        """Require the residual stream exactly when it is carried separately.

        Layers that defer their residual addition return the stream they
        carry forward; complete-stream layers return none.
        """
        if (stream.residual is None) == self.separate_residual:
            raise ValueError(
                "decoder layers must return a residual stream exactly when "
                "the decoder carries one separately"
            )

    def _exit(self, stream: _Stream) -> torch.Tensor:
        """Forward an earlier stage's streams, or normalize the final ones."""
        self._check_streams(stream)
        if self._pipeline.rank != self._pipeline.size - 1:
            return self._forward_stage(
                stream.hidden, stream.residual, stream.routes
            )
        return stream.partition.gather(
            self._normalize(stream.hidden, stream.residual, stream.routes)
        )

    def _forward_stage(self, hidden, residual, local_routes):
        """Send an earlier stage's streams on and return its hidden stream."""
        for value in (hidden, residual):
            if value is None:
                continue
            packed = (
                value.packed(local_routes)
                if isinstance(value, RoutedTensor)
                else value
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


@dataclass(slots=True)
class _Stream:
    """One stage's token shard of the streams a decoder's layers carry.

    ``hidden`` and ``residual`` are packed tensors, or per-route tensors in
    a routed pass, whose shard-local spans ``routes`` holds; ``residual`` is
    None before the first layer and for complete-stream layers.
    ``positions`` are the shard's rotary coordinates.
    """

    hidden: torch.Tensor | RoutedTensor
    residual: torch.Tensor | RoutedTensor | None
    positions: torch.Tensor
    partition: TokenShard
    routes: tuple[RouteSpan, ...]

    def apply(self, layer: nn.Module, attention: AttentionBatch) -> None:
        """Advance the streams through one decoder layer."""
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
