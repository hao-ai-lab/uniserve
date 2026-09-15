"""Compact prefix pages and current keys using live numerical lengths."""

import torch
import triton
import triton.language as tl

from uniserve.quantization import QuantizedTensor


@triton.jit
def _pack(
    KEYS,
    VALUES,
    KEY_SCALE,
    VALUE_SCALE,
    CURRENT_KEY,
    CURRENT_VALUE,
    TABLE,
    PREFIXES,
    QUERIES,
    CURRENT_OFFSETS,
    OFFSETS,
    OUTPUT_KEY,
    OUTPUT_VALUE,
    FEATURES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    TILE: tl.constexpr,
):
    # FEATURES is one token's flattened feature width (kv_heads * head_dim).
    sequence = tl.program_id(1)
    indices = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token, feature = indices // FEATURES, indices % FEATURES
    prefix, count = tl.load(PREFIXES + sequence), tl.load(QUERIES + sequence)
    valid = token < prefix + count
    from_prefix = token < prefix

    # Prefix tokens live in paged cache rows addressed through the block table.
    block = tl.load(TABLE + sequence * TABLE_STRIDE + token // BLOCK_SIZE, from_prefix, 0)
    cache_index = (block * BLOCK_SIZE + token % BLOCK_SIZE) * FEATURES + feature
    key = tl.load(KEYS + cache_index, from_prefix, 0)
    value = tl.load(VALUES + cache_index, from_prefix, 0)
    if QUANTIZED:
        key = key.to(tl.float32) * tl.load(KEY_SCALE + block, from_prefix, 0)
        value = value.to(tl.float32) * tl.load(VALUE_SCALE + block, from_prefix, 0)

    # Current tokens are packed per sequence, right after that sequence's prefix.
    current = (tl.load(CURRENT_OFFSETS + sequence) + token - prefix) * FEATURES + feature
    key = tl.where(from_prefix, key, tl.load(CURRENT_KEY + current, valid & ~from_prefix, 0))
    value = tl.where(from_prefix, value, tl.load(CURRENT_VALUE + current, valid & ~from_prefix, 0))

    destination = (tl.load(OFFSETS + sequence) + token) * FEATURES + feature
    tl.store(OUTPUT_KEY + destination, key, valid)
    tl.store(OUTPUT_VALUE + destination, value, valid)


def pack(cache, k, v, batch, offsets, *, out):
    """Return compact K/V within fixed capacity; offsets delimit its live rows."""

    if k.device.type != "cuda":
        # Reference path: gather each sequence's pages with plain indexing.
        from .torch import _paged

        keys, values, start = [], [], 0
        for row, count in enumerate(batch.queries.host):
            table, prefix = batch.block_table.indices[row], batch.prefixes.host[row]
            keys.extend((_paged(cache.key, table, prefix), k[start : start + count]))
            values.extend((_paged(cache.value, table, prefix), v[start : start + count]))
            start += count
        return torch.cat(keys), torch.cat(values)

    prefix_capacity = batch.block_table.indices.shape[1] * cache.block_size
    key, value = out
    encoded = isinstance(cache.key, QuantizedTensor)
    fields = tuple(
        tensor.buffers() if encoded else {"values": tensor, "scale": tensor}
        for tensor in (cache.key, cache.value)
    )
    features = k.shape[1] * k.shape[2]
    extent = (prefix_capacity + batch.queries.maximum) * features

    if extent and batch.queries.batch_size:
        _pack[(triton.cdiv(extent, 256), batch.queries.batch_size)](
            fields[0]["values"],
            fields[1]["values"],
            fields[0]["scale"],
            fields[1]["scale"],
            k.contiguous(),
            v.contiguous(),
            batch.block_table.indices,
            batch.prefixes.values,
            batch.queries.values,
            batch.queries.offsets,
            offsets,
            key,
            value,
            features,
            cache.block_size,
            batch.block_table.indices.stride(0),
            encoded,
            256,
        )
    return key, value
