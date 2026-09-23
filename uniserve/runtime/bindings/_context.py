"""Prepare packed context attention over borrowed head and K/V transports."""

from dataclasses import replace

import torch

from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention.inputs import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    VarlenInput,
    VisibleInput,
)


class _ContextPlan:
    """Map physical collective slots to global sequence coordinates.

    Token ownership follows topology order. A head exchange can join several
    disjoint token intervals; context publication can order them differently
    again. These maps preserve the logical sequences and exclude all transport
    padding, including entirely empty rank shards.
    """

    def __init__(self, layer, batch, *, cache, transport, allocate, dtype):
        if isinstance(batch, DenseInput):
            raise ValueError(
                "context attention requires explicit packed sequence lengths"
            )
        self.layer, self.cache = layer, cache
        table = getattr(batch, "block_table", None)
        if (
            cache is not None
            and table is not None
            and table.block_size != cache.block_size
        ):
            raise ValueError(
                "attention block table and cache block sizes differ"
            )

        self.parallel = layer.context_parallel
        self.mesh = self.parallel.mesh
        axes = self.parallel.parallel
        self.head_axes = () if axes.heads is None else (axes.heads.axis,)
        context_axes = tuple(
            axis for axis in self.mesh.axes if axis == axes.context.gather_axis
        )
        token_axes = tuple(
            axis
            for axis in self.mesh.axes
            if axis in (*context_axes, *self.head_axes)
        )
        self.tokens = self.mesh.get_group(token_axes)

        key_lengths = (
            batch.keys
            if isinstance(batch, (VarlenInput, VisibleInput))
            else batch.queries
        )
        query_tokens, key_tokens = (
            batch.queries.num_tokens,
            key_lengths.num_tokens,
        )
        if query_tokens is None or key_tokens is None:
            raise ValueError(
                "context attention requires exact host sequence lengths"
            )
        self.queries = TokenShard(query_tokens, self.tokens)
        self.keys = TokenShard(key_tokens, self.tokens)

        device = batch.queries.values.device
        self.query_members = layer.exchange.group.ranks
        # Physical exchange slots order each owner's shard contiguously;
        # query_indices gathers this rank's tokens in logical sequence order.
        query_slots = self._slots(self.query_members, self.queries)
        self.query_indices = torch.tensor(
            [slot for slot, _ in query_slots], dtype=torch.long, device=device
        )
        query_ids = [index for _, index in query_slots]
        # Per-sequence query columns, localized to each sequence's token start.
        self.query_counts, self.query_columns = [], []
        start = 0
        for count in batch.queries.host:
            indices = [
                index - start
                for index in query_ids
                if start <= index < start + count
            ]
            self.query_counts.append(len(indices))
            self.query_columns.append(
                torch.tensor(indices, dtype=torch.long, device=device)
            )
            start += count

        local = SequenceLengths.from_lengths(
            tuple(self.query_counts), device=device
        )
        self.visible = torch.empty(
            (local.batch_size, max(self.query_counts, default=0)),
            dtype=torch.int32,
            device=device,
        )
        self.key_values = torch.empty(
            local.batch_size, dtype=torch.int32, device=device
        )
        self.key_offsets = torch.empty(
            local.batch_size + 1, dtype=torch.int32, device=device
        )
        self.packed = None
        if isinstance(batch, SegmentedInput) and device.type == "cuda":
            from uniserve.tensors import BufferConfig

            capacity = (
                batch.block_table.indices.numel() * batch.block_table.block_size
                + query_tokens
            )
            shape = (capacity, layer.local_kv_heads, layer.head_dim)
            views = allocate(
                {name: BufferConfig(shape, dtype) for name in ("key", "value")},
                device,
            )
            self.packed = views["key"], views["value"]

        self._key_lengths = key_lengths
        self._transport = (
            None
            if isinstance(batch, VisibleInput) and batch.block_table is not None
            else transport(
                layer, self.keys.capacity * layer.exchange.group.size, dtype
            )
        )
        if self._transport is None:
            self.key_indices = None
        else:
            # The gather writes each owner's window in topology order, and
            # within a window the Ulysses members' tokens follow their own
            # order, so the map is built by walking owners then head fibers.
            members = self.parallel.key_group.ranks
            owner_capacity = self.keys.capacity * layer.exchange.group.size
            # Walk owners in topology order, pairing each physical slot with
            # the logical token index it carries.
            physical: list[tuple[int, int]] = []
            for owner_index, rank in enumerate(members):
                offset = owner_index * owner_capacity
                heads = next(
                    fiber
                    for fiber in self.mesh.members(self.head_axes)
                    if rank in fiber
                )
                physical.extend(
                    (offset + slot, index)
                    for slot, index in self._slots(heads, self.keys)
                )

            # Ordering by logical token index turns the slots into a gather map.
            physical.sort(key=lambda item: item[1])
            if [index for _, index in physical] != list(range(key_tokens)):
                raise ValueError(
                    "context publication must cover each logical key token once"
                )

            self.key_indices = torch.tensor(
                [slot for slot, _ in physical], dtype=torch.long, device=device
            )
        self.batch = VisibleInput(
            local, key_lengths, self.visible, None, False, False
        )
        self.refresh(batch)

    @staticmethod
    def signature(batch):
        """Identify a reusable context plan.

        Return the batch structure key identifying a reusable context plan.
        """
        if isinstance(batch, DenseInput):
            return (DenseInput,)
        lengths = (
            batch.keys
            if isinstance(batch, (VarlenInput, VisibleInput))
            else batch.queries
        )
        table = getattr(batch, "block_table", None)
        return (
            type(batch),
            batch.queries.host,
            lengths.host,
            None
            if table is None
            else (tuple(table.indices.shape), table.block_size),
        )

    def _slots(self, members, partition):
        # Each rank owns a contiguous capacity-sized window of the logical
        # token domain. Pairs are (physical exchange slot, logical token
        # index) in owner order, covering only tokens actually present.
        return [
            (owner * partition.capacity + offset, index)
            for owner, rank in enumerate(members)
            for offset, index in enumerate(
                range(
                    min(
                        partition.num_tokens,
                        self.tokens.ranks.index(rank) * partition.capacity,
                    ),
                    min(
                        partition.num_tokens,
                        (self.tokens.ranks.index(rank) + 1)
                        * partition.capacity,
                    ),
                )
            )
        ]

    def refresh(self, batch):
        """Read live lengths into localized metadata.

        Read live device lengths and endpoints into the localized metadata.
        """
        paged = isinstance(batch, (PagedInput, SegmentedInput))
        if paged:
            host = tuple(
                a + b
                for a, b in zip(
                    batch.prefixes.host, batch.queries.host, strict=True
                )
            )
            torch.add(
                batch.prefixes.values, batch.queries.values, out=self.key_values
            )
            self.key_offsets[0].zero_()
            torch.cumsum(
                self.key_values, 0, dtype=torch.int32, out=self.key_offsets[1:]
            )
            keys = SequenceLengths(
                host=host, values=self.key_values, offsets=self.key_offsets
            )
        else:
            keys = batch.keys

        # ends holds each local query token's exclusive visible key end.
        for row, columns in enumerate(self.query_columns):
            count = columns.numel()
            if isinstance(batch, (PagedInput, VarlenInput)):
                # Causal tokens see keys up to their own absolute position.
                ends = (
                    columns + keys.values[row] - batch.queries.values[row] + 1
                    if batch.causal[row]
                    else keys.values[row].expand(count)
                )
            elif isinstance(batch, SegmentedInput):
                ends = (
                    keys.values[row].expand(count)
                    if batch.fully_visible_current
                    else batch.prefixes.values[row]
                    + batch.visible_current_end[row].index_select(0, columns)
                )
            else:
                ends = (
                    keys.values[row].expand(count)
                    if batch.fully_visible
                    else batch.visible_end[row].index_select(0, columns)
                )
            self.visible[row, :count].copy_(ends)

        table = (
            batch.block_table
            if isinstance(batch, PagedInput) or isinstance(batch, VisibleInput)
            else None
        )
        self.batch = replace(self.batch, keys=keys, block_table=table)
        return self.batch

    def exchange(self, k, v, *, storage):
        """Publish local K/V shards into the global context key domain."""
        exchange = self.layer.exchange
        key, value = (
            exchange.heads(self.keys.pad(tensor), storage=storage, role=role)
            for role, tensor in (("key", k), ("value", v))
        )
        if self.keys.num_tokens == 0:
            return key[:0], value[:0]

        from uniserve.nn.attention._parallel import context_scope

        # Without context transport this layer binds no context buffers.
        bindings = (
            {} if self._transport is None else {self.parallel: self._transport}
        )
        with context_scope(bindings):
            key, value = self.parallel.distribute_key_value(key, value)
            if self.key_indices is not None:
                # Dense kernels consume compact sequences, and the gathered
                # rows arrive in owner order rather than logical token order.
                key = key.index_select(0, self.key_indices)
                value = value.index_select(0, self.key_indices)
            else:
                key, value = (
                    key[: self.keys.num_tokens],
                    value[: self.keys.num_tokens],
                )
        return key, value

    def inputs(self, q, k, v, batch, *, storage):
        """Gather this rank's attention inputs.

        Gather query rows and context K/V into this rank's attention inputs.
        """
        query = self.layer.exchange.heads(
            self.queries.pad(q), storage=storage, role="query"
        ).index_select(0, self.query_indices)

        cached = (
            isinstance(batch, VisibleInput) and batch.block_table is not None
        )
        if cached:
            if self.cache is not None:
                k, v = self.cache.key, self.cache.value
            else:
                heads = self.layer.exchange.head_slice(k.shape[2])
                k, v = k[:, :, heads], v[:, :, heads]
        else:
            k, v = self.exchange(k, v, storage=storage)

        if (
            isinstance(batch, (PagedInput, SegmentedInput))
            and batch.write_indices is not None
        ):
            if self.cache is None:
                raise RuntimeError(
                    "context cache writes require bound prefix state"
                )
            self.cache.update(k, v, indices=batch.write_indices)

        if isinstance(batch, SegmentedInput):
            if self.cache is None:
                raise RuntimeError(
                    "segmented context attention requires bound prefix state"
                )
            from ..backends.attention._segments import pack

            localized = self.refresh(batch)
            k, v = pack(
                self.cache, k, v, batch, localized.keys.offsets, out=self.packed
            )
            return query, k, v, localized

        return query, k, v, self.refresh(batch)

    def restore(self, result, *, storage, out):
        """Map output back to logical query order.

        Map compact attention output back to this owner's logical query order.
        """
        count = self.queries.capacity * self.layer.exchange.group.size
        # Scatter compact rows into physical exchange slots, exchange them to
        # their token owners, and keep this rank's logical prefix.
        physical = result.new_zeros((count, *result.shape[1:]))
        physical.index_copy_(0, self.query_indices, result)
        local = self.layer.exchange.tokens(physical, storage=storage)[
            : self.queries.count
        ]
        return out.copy_(local)
