"""Token exchange behind the public projection iterators."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from uniserve.distributed._chunks import _gather_chunks
from uniserve.distributed.mesh import _finish, _start_all_gather
from uniserve.quantization import QuantizedTensor, Quantizer, ScaleLayout

from . import _binding, functional


def tensor_statistics(quantizer: Quantizer | None) -> bool:
    """Return whether encoding derives scales from whole-tensor statistics.

    Dynamic NVFP4 and per-tensor FP8 scales span every source value, so
    encoded transport must wait for the complete input before quantization.
    A calibrated NVFP4 scale is already fixed by its checkpoint and can
    preserve that domain while encoding bounded row intervals.
    """
    return quantizer is not None and quantizer.requires_complete_source


def _projection_chunk_rows(width: int, dtype: torch.dtype) -> int:
    """Bound one projected source interval to an aligned 64 MiB payload."""
    return max(
        128,
        (64 * 1024 * 1024 // (width * dtype.itemsize) // 128) * 128,
    )


def _partition(module, x, token_slice, num_tokens):
    """Locate this rank's token interval within the logical projection input.

    Returns the gather group, the per-owner padded row capacity, and the
    logical token slice this context coordinate is responsible for.
    """
    axes = module.input_distribution.shard_axes(0)
    mesh = module.input_distribution.mesh
    owners = mesh.get_group(axes)
    capacity = (num_tokens + owners.size - 1) // owners.size
    start = min(num_tokens, capacity * owners.rank)
    stop = min(num_tokens, start + capacity)
    if (
        num_tokens < 0
        or token_slice.step not in (None, 1)
        or token_slice.start != start
        or token_slice.stop != stop
        or (x is not None and (x.ndim != 2 or x.shape[0] != stop - start))
    ):
        raise ValueError("token input must match its complete logical shard")

    gathered = getattr(module, "_gather_axes", axes)
    if not set(gathered).issubset(axes):
        raise ValueError("projection gather axes must partition input tokens")
    retained = tuple(axis for axis in axes if axis not in gathered)
    # A context coordinate must own a contiguous token interval. Token/head
    # axes inside it then publish consecutive subintervals in logical order.
    if (*retained, *gathered) != axes:
        raise ValueError("context token axes must precede gathered head axes")

    group = mesh.get_group(gathered)
    domain_rank = mesh.get_group(retained).rank
    begin = min(num_tokens, domain_rank * capacity * group.size)
    end = min(num_tokens, begin + capacity * group.size)
    return group, capacity, slice(begin, end)


def _pad(value: torch.Tensor, capacity: int) -> torch.Tensor:
    if value.shape[0] == capacity:
        return value.contiguous()
    result = torch.zeros(
        (capacity, *value.shape[1:]), dtype=value.dtype, device=value.device
    )
    result[: value.shape[0]].copy_(value)
    return result


@contextmanager
def _storage(module, size, device):
    binding = _binding.linear_chunks.get().get(id(module))
    if binding is None:
        # Standalone numerical calls use the same transport with an invocation
        # allocation. ExecutionContext supplies reusable, graph-stable backing.
        yield torch.empty(size, dtype=torch.uint8, device=device)
    else:
        with binding(size, device) as storage:
            yield storage


@contextmanager
def _encoded_gather(module, x, group, capacity, num_tokens):
    """All-gather every row-dependent encoded field into one quantized input.

    Row fields are packed contiguously into a single byte payload, one padded
    shard interval per group member, then narrowed back to the valid rows.
    """
    x = x.repack(scale_layout=ScaleLayout.LINEAR)
    row_fields = {
        name: value
        for name, value in x.buffers().items()
        if _row_field(name, x.quantizer)
    }
    # Empty local shards still have a known physical row width.
    size = (
        sum(
            capacity
            * torch.Size(value.shape[1:]).numel()
            * value.element_size()
            for value in row_fields.values()
        )
        * group.size
    )
    with _storage(module, size, x.device) as storage:
        fields = dict(x.buffers())
        cursor = 0
        for name, value in row_fields.items():
            source = _pad(value, capacity)
            byte_count = source.numel() * source.element_size() * group.size
            target = (
                storage[cursor : cursor + byte_count]
                .view(value.dtype)
                .view(capacity * group.size, *value.shape[1:])
            )
            # Byte transport preserves FP8 encodings on CPU as well as CUDA.
            group.all_gather(
                source.view(torch.uint8), out=target.view(torch.uint8)
            )
            fields[name] = target[:num_tokens]
            cursor += byte_count

        yield x.quantizer.from_tensors(
            fields,
            shape=(num_tokens, x.shape[1]),
            dtype=x.dtype,
            scale_layout=ScaleLayout.LINEAR,
        )


@contextmanager
def materialize_input(module, x, token_slice, num_tokens):
    """Yield the complete gathered input rows for this context coordinate."""
    x = functional._matrix(x)
    group, capacity, domain = _partition(module, x, token_slice, num_tokens)
    num_tokens = domain.stop - domain.start

    if group.size == 1 or num_tokens == 0:
        yield x
    elif isinstance(x, QuantizedTensor):
        with _encoded_gather(module, x, group, capacity, num_tokens) as values:
            yield values
    else:
        local = _pad(x, capacity)
        size = local.numel() * local.element_size() * group.size
        with _storage(module, size, x.device) as storage:
            output = (
                storage[:size]
                .view(x.dtype)
                .view(capacity * group.size, x.shape[1])
            )
            group.all_gather(local, out=output)
            yield output[:num_tokens]


def projection_inputs(
    module, x, token_slice, num_tokens
) -> Iterator[tuple[slice, torch.Tensor]]:
    """Yield ``(logical token slice, gathered rows)`` pairs for one projection.

    A plain tensor is exchanged in transport-sized chunks; a chunk iterator is
    streamed through ``_stream_inputs`` so projection can overlap production.
    """
    if not isinstance(x, torch.Tensor):
        yield from _stream_inputs(module, x, token_slice, num_tokens)
        return

    x = functional._matrix(x)
    group, capacity, domain = _partition(module, x, token_slice, num_tokens)
    num_tokens = domain.stop - domain.start
    if group.size == 1 or num_tokens == 0:
        yield token_slice, x
        return

    if isinstance(x, QuantizedTensor) or tensor_statistics(
        module.input_quantizer
    ):
        # Resolve full-source tensor statistics before transmitting encoded
        # fields. A scalar scale is replicated; row/block scales follow values.
        if not isinstance(x, QuantizedTensor):
            from .linear import _input

            x = _input(module, x)
        with _encoded_gather(module, x, group, capacity, num_tokens) as values:
            yield domain, values
        return

    local = _pad(x, capacity)
    with _storage(
        module, local.numel() * local.element_size() * group.size, x.device
    ) as storage:
        for interval, values in _gather_chunks(group, local, storage):
            stop = min(interval.stop, num_tokens)
            if interval.start < stop:
                yield (
                    slice(domain.start + interval.start, domain.start + stop),
                    values[: stop - interval.start],
                )


def _stream_inputs(module, chunks, token_slice, num_tokens):
    """Project ready intervals while the upstream layer produces later tokens.

    Inputs cover the local shard in logical order. Collective payload boundaries
    derive from the shard capacity, independent of the producer's chunk sizes.
    Complete-source quantizers wait for all source values before encoding.
    """
    group, capacity, domain = _partition(module, None, token_slice, num_tokens)
    width = module.weight.shape[1]
    rows = token_slice.stop - token_slice.start
    chunks = iter(chunks)

    if tensor_statistics(module.input_quantizer):
        values = _assemble(chunks, token_slice, width, module.weight)
        yield from projection_inputs(module, values, token_slice, num_tokens)
        return

    if group.size == 1:
        cursor = token_slice.start
        for interval, value in chunks:
            _check_chunk(interval, value, cursor, token_slice.stop, width)
            cursor = interval.stop
            # A singleton group has no transport to impose payload bounds.
            # Preserve the same bound locally so calibrated block-scaled
            # projections do not materialize full-sequence MLP intermediates.
            chunk_rows = _projection_chunk_rows(width, value.dtype)
            for start in range(0, value.shape[0], chunk_rows):
                stop = min(value.shape[0], start + chunk_rows)
                yield (
                    slice(interval.start + start, interval.start + stop),
                    value[start:stop],
                )
        if cursor != token_slice.stop:
            raise ValueError(
                "projection chunks must cover their complete token shard"
            )
        return

    # Gather scratch keeps every payload until its remote readers are enqueued.
    # Local projections can run immediately; remote projections wait only after
    # upstream production has launched all of its remaining numerical work.
    first = next(chunks, None)
    dtype = module.weight.dtype if first is None else first[1].dtype
    device = module.weight.device
    if first is not None and isinstance(first[1], QuantizedTensor):
        # Encoded fields have their own row layouts. Preserve those fields and
        # their original scale domains through the regular encoded transport.
        from itertools import chain

        values = _assemble(
            chain((first,), chunks), token_slice, width, module.weight
        )
        yield from projection_inputs(module, values, token_slice, num_tokens)
        return

    byte_count = capacity * width * dtype.itemsize * group.size
    with _storage(module, byte_count, device) as backing:
        storage = backing[:byte_count].view(dtype)
        # Use the same payload bound as complete-input row gathers. Producer
        # intervals need not force smaller GEMMs or additional peer
        # publications.
        chunk_rows = _projection_chunk_rows(width, dtype)
        backend_rank = group._backend_order.index(group.rank)
        pending = []
        staged = cursor = 0
        from itertools import chain

        def publish(start, count):
            # Storage lays each payload out as [group.size, count, width] so
            # every member contributes its staged rows at its own backend rank.
            target = storage[
                start * group.size * width : (start + count)
                * group.size
                * width
            ]
            target = target.view(group.size, count, width)
            work = _start_all_gather(
                target.flatten(0, 1), target[backend_rank], group._require()
            )
            pending.append((start, count, target, work))
            stop = min(start + count, rows)
            if start < stop:
                return slice(
                    token_slice.start + start, token_slice.start + stop
                ), target[backend_rank, : stop - start]
            return None

        consumed = 0
        try:
            # Stage produced rows into this member's payload slot and publish
            # each full payload as soon as its rows are complete.
            for interval, value in chain(
                () if first is None else (first,), chunks
            ):
                _check_chunk(
                    interval,
                    value,
                    token_slice.start + cursor,
                    token_slice.stop,
                    width,
                )
                offset = 0
                while offset < value.shape[0]:
                    count = min(chunk_rows, capacity - staged)
                    target = storage[
                        staged * group.size * width : (staged + count)
                        * group.size
                        * width
                    ].view(group.size, count, width)[backend_rank]
                    take = min(value.shape[0] - offset, staged + count - cursor)
                    target[cursor - staged : cursor - staged + take].copy_(
                        value[offset : offset + take]
                    )
                    cursor += take
                    offset += take
                    if cursor == staged + count:
                        ready = publish(staged, count)
                        staged += count
                        if ready is not None:
                            yield ready
            if cursor != rows:
                raise ValueError(
                    "projection chunks must cover their complete token shard"
                )

            # Pad and publish any remaining payload slots up to the shard
            # capacity so every peer's collective sees the same payload sizes.
            while staged < capacity:
                count = min(chunk_rows, capacity - staged)
                target = storage[
                    staged * group.size * width : (staged + count)
                    * group.size
                    * width
                ].view(group.size, count, width)[backend_rank]
                target[max(0, cursor - staged) :].zero_()
                ready = publish(staged, count)
                staged += count
                if ready is not None:
                    yield ready

            # Remote rows become readable once their owners' gathers complete.
            for start, count, target, work in pending:
                _finish(work, target)
                consumed += 1
                for physical, logical in enumerate(group._backend_order):
                    if logical == group.rank:
                        continue
                    begin = domain.start + logical * capacity + start
                    stop = min(begin + count, domain.stop)
                    if begin < stop:
                        yield (
                            slice(begin, stop),
                            target[physical, : stop - begin],
                        )
        finally:
            for _, _, target, work in pending[consumed:]:
                _finish(work, target)


def _check_chunk(interval, value, cursor, stop, width):
    if (
        interval.step not in (None, 1)
        or interval.start != cursor
        or not cursor < interval.stop <= stop
        or value.ndim != 2
        or value.shape != (interval.stop - cursor, width)
    ):
        raise ValueError(
            "projection chunks require ordered intervals covering their "
            "local shard"
        )


def _assemble(chunks, token_slice, width, reference):
    """Join an ordered chunk stream into one dense or encoded [rows, width]
    tensor.
    """  # noqa: D205
    cursor = token_slice.start
    parts = []
    for interval, value in chunks:
        _check_chunk(interval, value, cursor, token_slice.stop, width)
        cursor = interval.stop
        parts.append(value)
    if cursor != token_slice.stop:
        raise ValueError(
            "projection chunks must cover their complete token shard"
        )

    if not parts:
        return torch.empty(
            (0, width), dtype=reference.dtype, device=reference.device
        )
    if len(parts) == 1:
        return parts[0]
    if not any(isinstance(value, QuantizedTensor) for value in parts):
        return torch.cat(parts)

    first = parts[0]
    if not all(
        isinstance(value, QuantizedTensor)
        and value.quantizer == first.quantizer
        and value.dtype == first.dtype
        and value.device == first.device
        for value in parts
    ):
        raise ValueError(
            "encoded projection chunks must share one representation"
        )

    encoded = [value.repack(scale_layout=ScaleLayout.LINEAR) for value in parts]
    fields = {}
    for name in first.buffers():
        values = [value.buffers()[name] for value in encoded]
        if _row_field(name, first.quantizer):
            # Concatenate encoded bytes, never the tensor subclass's dense
            # numerical fallback. Re-encoding would change the source domain.
            fields[name] = torch.cat(
                [value.contiguous().view(torch.uint8) for value in values]
            ).view(values[0].dtype)
        else:
            # Chunks describe one logical tensor. Replicated scales therefore
            # agree even when the caller copied their backing separately.
            for value in values[1:]:
                if value.data_ptr() == values[0].data_ptr():
                    continue
                equal = (value == values[0]).all()
                if value.is_cuda:
                    torch._assert_async(
                        equal, "projection chunk scale domains disagree"
                    )
                elif not bool(equal):
                    raise ValueError("projection chunk scale domains disagree")
            fields[name] = values[0]

    return first.quantizer.from_tensors(
        fields,
        shape=(token_slice.stop - token_slice.start, width),
        dtype=first.dtype,
        scale_layout=ScaleLayout.LINEAR,
    )


def _row_field(name, quantizer):
    """Return whether an encoded buffer carries one value per tensor row."""
    return name in {"values", "block_scale"} or (
        name == "scale" and (quantizer.axis == 0 or quantizer.format == "mxfp8")
    )
