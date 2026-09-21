"""Video tile pooling, top-k selection, fine attention and dense compression."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.nn import _binding
from uniserve.ops import video_sparse as ops
from uniserve.ops.video_sparse_rows import (
    pack_sparse_input_rows,
    prepare_sparse_input_rows,
)

from .inputs import BlockInput, Input, NormRope, Workspace


class BlockAttention(nn.Module):
    """Compute attention over an explicit per-head selected key-block domain."""

    def __init__(self, scale: float, tile_size: int = 64):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0 or tile_size != 64:
            raise ValueError("VSA requires a positive scale and tile size 64")
        self.scale, self.tile_size = scale, tile_size

    @contextmanager
    def _operator(self, q, batch):
        """Yield the bound operator, or build a transient one outside
        serving.
        """  # noqa: D205
        binding = _binding.vsa.get().get(id(self))
        if binding is not None:
            yield binding.prepare(batch.pattern, q)
            return

        from uniserve.runtime.backends.attention import vsa
        from uniserve.runtime.tensor_buffers import TensorBuffers

        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind and prepare VSA before CUDA graph capture")
        provider = vsa.resolve("auto", device=q.device)
        options = {
            "num_heads": q.shape[1],
            "head_dim": q.shape[2],
            "dtype": q.dtype,
        }
        requirements = provider.workspace_buffers(batch.pattern, **options)
        buffers = TensorBuffers.allocate(requirements, device=q.device)
        operator = provider.prepare(
            batch.pattern, **options, workspace=buffers.view(requirements)
        )
        try:
            operator.bind(batch)
            yield operator
        finally:
            operator.close()
            buffers.close()

    def forward(self, q, k, v, batch: BlockInput, *, out=None):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        with self._operator(q, batch) as operator:
            return operator(q, k, v, batch, scale=self.scale, out=out)


@dataclass
class _PreparedInput:
    """Borrow complete projected Q/K/V and publish tile means by interval."""

    shape: tuple[int, int, int]
    dtype: torch.dtype
    valid_sizes: torch.Tensor
    owners: int
    chunk_tokens: int
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    packed: torch.Tensor
    gate: torch.Tensor

    def append(
        self,
        interval: slice,
        q,
        k,
        v,
        g,
        norm_rope: NormRope | None = None,
        tokens: slice | None = None,
    ):
        """Publish one projected interval; ``tokens`` is its global span.

        With ``norm_rope``, Q and K are normalized and rotated as they are
        pooled and packed, in one pass over the projections that also copies
        the gate rows.
        """
        start, stop = interval.start, interval.stop
        if (
            start < 0
            or start % 64
            or stop % 64
            or stop > self.shape[0]
            or stop - start != q.shape[0]
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape != g.shape
        ):
            raise ValueError(
                "VSA projection chunks must cover complete in-range tiles"
            )

        if norm_rope is not None:
            assert tokens is not None
            prepare_sparse_input_rows(
                q,
                k,
                v,
                g,
                norm_rope.query_weight,
                norm_rope.key_weight,
                norm_rope.cos[tokens],
                norm_rope.sin[tokens],
                self.valid_sizes,
                eps=norm_rope.eps,
                packed=self.packed,
                packed_gate=self.gate,
                pooled_query=self.pooled_query,
                pooled_key=self.pooled_key,
                pooled_value=self.pooled_value,
                owners=self.owners,
                chunk_rows=self.chunk_tokens,
                row_start=start,
            )
            return

        # Pool one mean per 64-token tile, then pack the full-resolution rows
        # into the shared exchange layout at the same logical positions.
        self.gate[start:stop].copy_(g)
        tiles = slice(start // 64, stop // 64)
        ops.pool_qkv_means(
            q,
            k,
            v,
            self.valid_sizes,
            self.pooled_query[tiles],
            self.pooled_key[tiles],
            self.pooled_value[tiles],
            query_tile_offset=start // 64,
            key_tile_offset=start // 64,
        )
        pack_sparse_input_rows(
            q,
            k,
            v,
            self.valid_sizes,
            owners=self.owners,
            chunk_rows=self.chunk_tokens,
            packed=self.packed,
            row_start=start,
            row_major=True,
        )


class Attention(nn.Module):
    """Compose VSA's selected fine attention and learned dense tile compression.

    Direct select/forward calls consume explicit query and key domains.
    ``forward_chunks`` consumes tile-aligned projection intervals, pools each
    interval once, and supplies complete keys to each fine-query interval.
    """

    def __init__(
        self,
        attention: BlockAttention,
        *,
        tile_size: int = 64,
        mesh: DeviceMesh | None = None,
    ):
        super().__init__()
        if tile_size != attention.tile_size:
            raise ValueError(
                "VSA selection and block attention must use the same tile size"
            )
        self.attention, self.tile_size = attention, tile_size
        self.mesh = (
            DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
            if mesh is None
            else mesh
        )

    def _select_pooled(
        self, inputs, selected_tiles, workspace, query_tile_offset=0
    ):
        # Similarity between pooled queries and pooled keys:
        # [heads, query, key].
        scores = workspace.tile_scores
        query = workspace.pooled_query.permute(1, 0, 2)
        key = workspace.pooled_key.permute(1, 0, 2)
        # The softmax scale is applied in the GEMM epilogue; beta 0 leaves
        # the score storage unread.
        torch.baddbmm(
            scores,
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=self.attention.scale,
            out=scores,
        )

        heads, tiles, _ = scores.shape
        pattern = inputs.pattern(
            tiles,
            selected_tiles=selected_tiles,
            query_tile_offset=query_tile_offset,
        )
        prefix, valid = inputs.prefix_tiles, inputs.valid_tiles
        local_prefix = max(0, min(tiles, prefix - query_tile_offset))
        local_video = max(0, min(tiles, valid - query_tile_offset))

        # One pass writes every query tile's key-tile list: prefix tiles
        # attend densely, video tiles to the prefix plus their top-scoring
        # video tiles, padding tiles to one tile.
        indices, counts = workspace.block_indices, workspace.block_counts
        ops.write_block_map(
            scores[:, local_prefix:local_video, prefix:valid],
            inputs.prefix_key_indices,
            inputs.dense_key_indices,
            indices,
            counts,
            local_prefix=local_prefix,
            local_video=local_video,
            prefix_tiles=prefix,
            valid_tiles=valid,
            selected=selected_tiles,
        )

        return BlockInput(
            pattern, indices, counts, inputs.valid_sizes, query_tile_offset
        )

    def select(
        self, q, k, inputs: Input, *, selected_tiles: int, workspace: Workspace
    ) -> BlockInput:
        """Pool complete Q/K and return the selected key-block domain."""
        offset = self._query_offset(q.shape[0], inputs.padded_tokens)
        ops.pool_qkv_means(
            q,
            k,
            k,
            inputs.valid_sizes,
            workspace.pooled_query,
            workspace.pooled_key,
            workspace.pooled_value,
            query_tile_offset=offset,
        )
        return self._select_pooled(inputs, selected_tiles, workspace, offset)

    def _query_offset(self, query_tokens, key_tokens):
        if query_tokens == key_tokens:
            return 0
        parallel = getattr(self, "_parallel", None)
        if (
            parallel is None
            or query_tokens * parallel.context_group.size != key_tokens
        ):
            raise ValueError(
                "VSA query domain requires a matching bound context partition"
            )
        return parallel.context_group.rank * query_tokens // self.tile_size

    def _compress(self, batch, workspace):
        scores = workspace.tile_scores
        query = workspace.pooled_query.permute(1, 0, 2)
        key = workspace.pooled_key.permute(1, 0, 2)
        # The softmax scale is applied in the GEMM epilogue; beta 0 leaves
        # the score storage unread.
        torch.baddbmm(
            scores,
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=self.attention.scale,
            out=scores,
        )
        self._compress_scores(batch, workspace)

    @staticmethod
    def _compress_scores(batch, workspace):
        scores = workspace.tile_scores
        # One pass over the score storage masks empty key tiles and
        # normalizes each query tile's row in fp32.
        ops.tile_softmax(scores, batch.valid_sizes)
        torch.matmul(
            scores,
            workspace.pooled_value.permute(1, 0, 2),
            out=workspace.compressed_tiles,
        )
        start = batch.query_tile_offset
        query_valid = batch.valid_sizes[
            start : start + workspace.pooled_query.shape[0]
        ]
        workspace.compressed_tiles.masked_fill_(
            query_valid.view(1, -1, 1) == 0, 0
        )

    def forward(
        self,
        q,
        k,
        v,
        gate,
        batch: BlockInput,
        *,
        workspace: Workspace,
        out=None,
    ):
        """Fuse selected fine attention with the gated dense tile
        compression.
        """  # noqa: D205
        ops.pool_qkv_means(
            q,
            k,
            v,
            batch.valid_sizes,
            workspace.pooled_query,
            workspace.pooled_key,
            workspace.pooled_value,
            query_tile_offset=batch.query_tile_offset,
        )
        self._compress(batch, workspace)

        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        self.attention(q, k, v, batch, out=workspace.attention_output)
        ops.unpack_add_compression(
            workspace.attention_output.transpose(0, 1).unsqueeze(0),
            gate,
            workspace.compressed_tiles,
            out,
        )
        return out

    def forward_chunks(
        self,
        chunks: Iterator[
            tuple[
                slice,
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ],
        inputs: Input,
        *,
        selected_tiles: int,
        workspace: Workspace,
        norm_rope: NormRope | None = None,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Attend tile-aligned projection chunks of ``(q, k, v, gate)``.

        ``norm_rope`` applies the model's Q/K normalization and rotation to
        each chunk while it is prepared, so callers pass raw projections.
        """
        from uniserve.nn.attention._parallel import AttentionRowExchange

        parallel = getattr(self, "_parallel", None)
        context_size = 1 if parallel is None else parallel.context_group.size
        context_rank = 0 if parallel is None else parallel.context_group.rank
        owners = 1 if parallel is None else parallel.ulysses_group.size
        query_tokens = inputs.padded_tokens // context_size
        query_start = context_rank * query_tokens
        if inputs.padded_tokens % context_size or query_tokens % (64 * owners):
            raise ValueError(
                "VSA context and head partitions require complete tile "
                "intervals"
            )
        query_tiles = slice(
            query_start // 64, (query_start + query_tokens) // 64
        )

        # Keys must be complete before fine attention. Pool and pack each
        # projection while its producer can overlap subsequent communication.
        intervals = []
        prepared = gate = output_backing = None
        for interval, values in chunks:
            q, k, v, g = values
            if (
                interval.step not in (None, 1)
                or interval.start < query_start
                or interval.stop > query_start + query_tokens
                or interval.stop - interval.start != q.shape[0]
                or q.shape[0] == 0
            ):
                raise ValueError(
                    "VSA chunks must lie within their global query interval"
                )
            if prepared is None:
                shape = (query_tokens, *q.shape[1:])
                chunk_tokens = AttentionRowExchange.chunk_rows(
                    workspace.attention_output
                )
                binding = _binding.vsa.get().get(id(self.attention))
                requirements = {
                    "qkv": ((3, *shape), q.dtype),
                    "gate": (shape, g.dtype),
                    "output": (shape, q.dtype),
                }
                if binding is None:
                    if parallel is not None and (
                        context_size > 1 or owners > 1
                    ):
                        raise RuntimeError(
                            "distributed VSA requires an ExecutionContext"
                        )
                    buffers = {
                        name: torch.empty(shape, dtype=dtype, device=q.device)
                        for name, (shape, dtype) in requirements.items()
                    }
                else:
                    buffers = binding.buffers(requirements, q.device)
                    binding.exchange(
                        self, query_tokens, q.shape[1], q.shape[2], q.dtype
                    )
                gate, output_backing = buffers["gate"], buffers["output"]
                prepared = _PreparedInput(
                    shape,
                    q.dtype,
                    inputs.valid_sizes[query_tiles],
                    owners if context_size == 1 else 1,
                    chunk_tokens,
                    workspace.pooled_query,
                    workspace.pooled_key[query_tiles],
                    workspace.pooled_value[query_tiles],
                    buffers["qkv"],
                    gate,
                )
            local = slice(
                interval.start - query_start, interval.stop - query_start
            )
            prepared.append(local, q, k, v, g, norm_rope, interval)
            intervals.append((interval.start, interval.stop))

        # A gather may publish its local interval before remote intervals.
        # Pooling and packing address logical positions, so publication order
        # is independent of coverage and every tile still has exactly one
        # writer.
        cursor = query_start
        for start, stop in sorted(intervals):
            if start != cursor:
                raise ValueError(
                    "VSA projection chunks must cover each query token once"
                )
            cursor = stop
        if prepared is None or cursor != query_start + query_tokens:
            raise ValueError(
                "VSA chunks must publish the complete context query interval"
            )

        q, k, v = prepared.packed.unbind(0)
        if context_size > 1:
            # Complete pooled keys and values across the context partition.
            group = parallel.context_group
            group._all_gather_into_tensor(
                workspace.pooled_key, workspace.pooled_key[query_tiles].clone()
            )
            group._all_gather_into_tensor(
                workspace.pooled_value,
                workspace.pooled_value[query_tiles].clone(),
            )

        batch = self._select_pooled(
            inputs, selected_tiles, workspace, query_start // 64
        )
        self._compress_scores(batch, workspace)
        if context_size > 1:
            yield from self._context_chunks(
                q, k, v, gate, batch, parallel, workspace
            )
            return

        # Fine-query intervals share the full selected key domain. Mutable
        # query maps are prepared by the runtime operator before graph capture.
        with self.attention._operator(q, batch) as operator:
            producer = operator.rows(
                q,
                k,
                v,
                batch,
                gate=gate,
                compressed=workspace.compressed_tiles,
                out=workspace.attention_output,
                owners=owners,
                chunk_tokens=prepared.chunk_tokens,
                packed=prepared.packed,
                scale=self.attention.scale,
            )
            if owners > 1:
                # Registered destination tensors support the existing paired
                # owner production and copy-engine head-to-token exchange.
                outputs = parallel.output_views(q)
                local_output = outputs[parallel.ulysses_group.rank].view_as(q)
                exchange = AttentionRowExchange(
                    parallel,
                    local_output,
                    producer,
                    receive_workspace=parallel.output_buffers.receive,
                )
                start = parallel.ulysses_group.rank * (query_tokens // owners)
                for interval, output in exchange.chunks():
                    yield (
                        slice(start + interval.start, start + interval.stop),
                        output,
                    )
            else:
                for start in range(0, query_tokens, prepared.chunk_tokens):
                    stop = min(query_tokens, start + prepared.chunk_tokens)
                    # Fine-attention scratch can be reused by sequential dense
                    # and sparse pieces without overwriting completed outputs.
                    output = output_backing[start:stop]
                    producer(slice(start, stop), (output,))
                    yield slice(start, stop), output

    def _context_chunks(self, q, k, v, gate, batch, parallel, workspace):
        # The gather returns every owner's rows in one compact domain, which
        # is the domain the batch's tile identifiers already address.
        context_k, context_v = parallel.distribute_key_value(k, v)

        self.attention(
            q, context_k, context_v, batch, out=workspace.attention_output
        )

        outputs = parallel.output_views(q)
        attended = workspace.attention_output.transpose(0, 1).unsqueeze(0)
        if len(outputs) == 1:
            ops.unpack_add_compression(
                attended, gate, workspace.compressed_tiles, outputs[0]
            )
        else:
            ops.compose_to_head_shards(
                attended,
                gate,
                workspace.compressed_tiles,
                outputs,
                parallel.ulysses_group.rank,
            )

        result = parallel.finish_output(outputs)
        start = (
            batch.query_tile_offset * 64
            + parallel.ulysses_group.rank * result.shape[0]
        )
        yield slice(start, start + result.shape[0]), result
