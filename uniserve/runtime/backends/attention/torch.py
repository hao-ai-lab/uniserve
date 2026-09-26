"""Portable SDPA over dense, variable-length, paged and segmented inputs."""

import torch
from torch.nn import functional as F

from uniserve.nn.attention.inputs import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor

from . import Backend as _Backend
from . import Operator as _Operator


def _host(lengths: SequenceLengths) -> tuple[int, ...]:
    """Return the exact host lengths that eager row evaluation indexes."""
    if lengths.host is None:
        raise ValueError(
            "this attention preparation requires exact host sequence lengths"
        )
    return lengths.host


def _paged(value, table, length):
    """Gather one sequence's live prefix pages.

    Gather one sequence's live prefix pages into compact [length, heads,
    dim] form.

    The cache backing is [blocks, tokens, heads, dim]; per-block FP8 state is
    dequantized during the gather.
    """
    if value.ndim != 4 or length < 0:
        raise ValueError(
            "paged attention requires [blocks, tokens, heads, dim] backing"
        )
    if length == 0:
        return torch.empty(
            (0, *value.shape[2:]), dtype=value.dtype, device=value.device
        )

    count = (length + value.shape[1] - 1) // value.shape[1]
    if table.numel() < count:
        raise ValueError(
            "block table does not cover the requested key sequence"
        )

    indices = table[:count].to(device=value.device, dtype=torch.int64)
    if isinstance(value, QuantizedTensor):
        if value.quantizer.format != "fp8" or value.quantizer.axis != 0:
            raise ValueError("paged SDPA requires dense or per-block FP8 state")
        fields = value.buffers()
        encoded = (
            fields["values"]
            .view(torch.uint8)
            .index_select(0, indices)
            .view(torch.float8_e4m3fn)
        )
        scales = fields["scale"].index_select(0, indices)
        gathered = (encoded.float() * scales).to(value.dtype)
    else:
        gathered = value.index_select(0, indices)
    return gathered.flatten(0, 1)[:length]


def _dense(q, k, v, *, causal, scale, mask=None, window=None):
    packed = q.ndim == 3
    if packed:
        q, k, v = (value.transpose(0, 1).unsqueeze(0) for value in (q, k, v))

    if k.shape[-2] == 0:
        result = torch.zeros_like(q)
    else:
        causal_flag = causal and mask is None and window is None

        if window is not None or (
            causal and (mask is not None or q.shape[-2] != k.shape[-2])
        ):
            # SDPA's is_causal assumes square Q/K alignment and no custom mask;
            # fold causality and the history window into an explicit
            # visibility mask otherwise. Queries align to the end of the keys.
            query_positions = (
                torch.arange(q.shape[-2], device=q.device)
                + k.shape[-2]
                - q.shape[-2]
            )
            key_positions = torch.arange(k.shape[-2], device=q.device)
            visible = torch.ones(
                (q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device
            )
            if causal:
                visible &= key_positions[None] <= query_positions[:, None]
            if window is not None:
                visible &= (
                    key_positions[None] >= query_positions[:, None] - window
                )
            if mask is None:
                mask = visible
            elif mask.dtype.is_floating_point:
                mask = mask.to(q.device).masked_fill(~visible, -torch.inf)
            else:
                mask = mask.to(device=q.device, dtype=torch.bool) & visible
            causal_flag = False

        if mask is not None:
            mask = mask.to(
                device=q.device,
                dtype=q.dtype if mask.dtype.is_floating_point else torch.bool,
            )

        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal_flag,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    return result.squeeze(0).transpose(0, 1) if packed else result


def _state(q, k, v, *, scale, allowed):
    """Evaluate one attention segment in float32, returning (output, logsumexp).

    q/k/v are [tokens, heads, dim]; `allowed` is an optional [queries, keys]
    visibility mask shared across heads. A fully masked row yields NaN
    probabilities, which nan_to_num resets to zero output with -inf LSE.
    """
    if k.shape[0] == 0:
        return torch.zeros_like(q), torch.full(
            q.shape[:2], -torch.inf, dtype=torch.float32, device=q.device
        )

    copies = q.shape[1] // k.shape[1]
    if copies > 1:
        k, v = (
            k.repeat_interleave(copies, dim=1),
            v.repeat_interleave(copies, dim=1),
        )

    scores = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
    if allowed is not None:
        scores.masked_fill_(~allowed.unsqueeze(1), -torch.inf)

    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1).nan_to_num(0).to(v.dtype)
    return torch.einsum("qhk,khd->qhd", probabilities, v).to(q.dtype), lse


def _merge(first, second):
    """Combine two partial attention states.

    Combine two (output, lse) partial attention states with online softmax.
    """
    a, alse = first
    b, blse = second
    maximum = torch.logaddexp(alse, blse)
    aweight = torch.exp(alse - maximum).nan_to_num(0)
    bweight = torch.exp(blse - maximum).nan_to_num(0)
    return (
        a.float() * aweight.unsqueeze(-1) + b.float() * bweight.unsqueeze(-1)
    ).to(a.dtype)


def _page_capacity(value, table, length):
    """Gather fixed backing while ignoring table entries beyond live lengths."""
    blocks = torch.arange(table.numel(), device=table.device)
    table = torch.where(blocks * value.shape[1] < length, table, 0)
    return _paged(value, table, table.numel() * value.shape[1])


def _captured(q, k, v, batch, cache, scale, window):
    """Express packed visibility with live device masks during CUDA capture.

    Python slices cannot follow changed sequence lengths on replay. The Torch
    provider evaluates fixed-capacity SDPA domains and masks each sequence's
    live query/key interval. Native providers handle large packed workloads.
    """
    query_indices = torch.arange(q.shape[0], device=q.device)
    key_indices = torch.arange(k.shape[0], device=k.device)
    output = torch.zeros_like(q)
    if not q.shape[0]:
        return output
    for row in range(batch.queries.batch_size):
        query_start = batch.queries.offsets[row]
        query_count = batch.queries.values[row]
        local_query = query_indices - query_start
        queries = (local_query >= 0) & (local_query < query_count)

        # Paged and segmented inputs always carry a block table; visible
        # inputs may instead hold their keys densely.
        if (
            isinstance(batch, (PagedInput, SegmentedInput, VisibleInput))
            and batch.block_table is not None
        ):
            key, value = (k, v) if cache is None else (cache.key, cache.value)
            key_count = (
                batch.keys.values[row]
                if isinstance(batch, VisibleInput)
                else batch.prefixes.values[row]
                + (query_count if isinstance(batch, PagedInput) else 0)
            )
            table = batch.block_table.indices[row]
            keys, values = (
                _page_capacity(tensor, table, key_count)
                for tensor in (key, value)
            )
            local_key = torch.arange(keys.shape[0], device=q.device)
        else:
            keys, values = k, v
            key_count = batch.keys.values[row]
            local_key = key_indices - batch.keys.offsets[row]

        valid_keys = (local_key >= 0) & (local_key < key_count)
        # Unused cache slots can contain arbitrary bytes, including NaNs.
        # Remove them before the matrix products rather than relying on a
        # later score mask to suppress invalid floating-point operands.
        keys = torch.where(valid_keys[:, None, None], keys, 0)
        values = torch.where(valid_keys[:, None, None], values, 0)

        allowed = queries[:, None] & valid_keys[None, :]
        # Paged and variable-length queries align to the end of their keys.
        position = local_query + key_count - query_count
        if isinstance(batch, (PagedInput, VarlenInput)) and batch.causal[row]:
            allowed &= local_key[None, :] <= position[:, None]
        if isinstance(batch, (PagedInput, VarlenInput)) and window is not None:
            allowed &= local_key[None, :] >= position[:, None] - window
        if isinstance(batch, SegmentedInput) and window is not None:
            # The segmented prefix keeps one fixed interval for all queries.
            allowed &= local_key[None, :] >= key_count - window
        if isinstance(batch, VisibleInput) and not batch.fully_visible:
            ends = batch.visible_end[row]
            visible = ends[local_query.clamp(0, ends.numel() - 1)]
            allowed &= local_key[None, :] < visible[:, None]

        if isinstance(batch, SegmentedInput):
            current_key = key_indices - query_start
            current_keys = (current_key >= 0) & (current_key < query_count)
            current = queries[:, None] & current_keys[None, :]
            if not batch.fully_visible_current:
                ends = batch.visible_current_end[row]
                visible = ends[local_query.clamp(0, ends.numel() - 1)]
                current &= current_key[None, :] < visible[:, None]
            # Current K/V spans every sequence. A zero attention probability
            # cannot suppress another sequence's NaN value in the final sum.
            result = _merge(
                _state(
                    q,
                    torch.where(current_keys[:, None, None], k, 0),
                    torch.where(current_keys[:, None, None], v, 0),
                    scale=scale,
                    allowed=current,
                ),
                _state(q, keys, values, scale=scale, allowed=allowed),
            )
        else:
            result = _dense(
                q, keys, values, causal=False, scale=scale, mask=allowed
            )
        # Different sequences have disjoint query intervals. Fully masked
        # rows are zero, so combining them preserves each sequence's result.
        output.add_(torch.where(queries[:, None, None], result, 0))
    return output


class _TorchOperator(_Operator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        if (
            isinstance(batch, (PagedInput, SegmentedInput))
            and batch.write_indices is not None
        ):
            self.update_cache(k, v, indices=batch.write_indices)

        if isinstance(batch, DenseInput):
            return out.copy_(
                _dense(
                    q,
                    k,
                    v,
                    causal=batch.causal,
                    scale=scale,
                    mask=batch.mask,
                    window=self.window,
                )
            )

        if q.ndim != 3 or (
            batch.queries.num_tokens is not None
            and q.shape[0] != batch.queries.num_tokens
        ):
            raise ValueError(
                "packed attention rows must match their declared token lengths"
            )

        if isinstance(batch, SegmentedInput) and self.cache is None:
            raise RuntimeError(
                "segmented attention requires bound prefix state"
            )

        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            return out.copy_(
                _captured(q, k, v, batch, self.cache, scale, self.window)
            )

        qstart = kstart = 0
        for row, count in enumerate(batch.queries.host):
            query = q[qstart : qstart + count]

            if isinstance(batch, VarlenInput):
                key_count = _host(batch.keys)[row]
                keys, values = (
                    k[kstart : kstart + key_count],
                    v[kstart : kstart + key_count],
                )
                result = _dense(
                    query,
                    keys,
                    values,
                    causal=batch.causal[row],
                    scale=scale,
                    window=self.window,
                )
                kstart += key_count
            elif isinstance(batch, PagedInput):
                key, value = (
                    (k, v)
                    if self.cache is None
                    else (self.cache.key, self.cache.value)
                )
                key_count = _host(batch.prefixes)[row] + count
                keys = _paged(key, batch.block_table.indices[row], key_count)
                values = _paged(
                    value, batch.block_table.indices[row], key_count
                )
                result = _dense(
                    query,
                    keys,
                    values,
                    causal=batch.causal[row],
                    scale=scale,
                    window=self.window,
                )
            elif isinstance(batch, VisibleInput):
                key_count = _host(batch.keys)[row]
                if batch.block_table is None:
                    keys, values = (
                        k[kstart : kstart + key_count],
                        v[kstart : kstart + key_count],
                    )
                else:
                    key, value = (
                        (k, v)
                        if self.cache is None
                        else (self.cache.key, self.cache.value)
                    )
                    keys = _paged(
                        key, batch.block_table.indices[row], key_count
                    )
                    values = _paged(
                        value, batch.block_table.indices[row], key_count
                    )
                allowed = (
                    None
                    if batch.fully_visible
                    else (
                        torch.arange(key_count, device=q.device).unsqueeze(0)
                        < batch.visible_end[row, :count].unsqueeze(1)
                    )
                )
                result = _dense(
                    query, keys, values, causal=False, scale=scale, mask=allowed
                )
                kstart += key_count
            elif isinstance(batch, SegmentedInput):
                prefix = _host(batch.prefixes)[row]
                keys = _paged(
                    self.cache.key, batch.block_table.indices[row], prefix
                )
                values = _paged(
                    self.cache.value, batch.block_table.indices[row], prefix
                )
                allowed = (
                    None
                    if batch.fully_visible_current
                    else (
                        torch.arange(count, device=q.device).unsqueeze(0)
                        < batch.visible_current_end[row, :count].unsqueeze(1)
                    )
                )
                # The window fixes one prefix interval [P - window, P) for
                # every query; the current keys stay as declared.
                history = (
                    None
                    if self.window is None
                    else (
                        torch.arange(prefix, device=q.device)
                        >= prefix - self.window
                    ).expand(count, prefix)
                )
                result = _merge(
                    _state(
                        query,
                        k[qstart : qstart + count],
                        v[qstart : qstart + count],
                        scale=scale,
                        allowed=allowed,
                    ),
                    _state(query, keys, values, scale=scale, allowed=history),
                )
            else:
                raise TypeError("unsupported numerical attention input")

            out[qstart : qstart + count].copy_(result)
            qstart += count

        return out


class Backend(_Backend):
    name = "torch"

    operator_class = _TorchOperator
