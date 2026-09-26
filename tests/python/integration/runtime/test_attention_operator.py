"""Prepared attention consumes numerical batches and borrowed prefix state."""

import pytest
import torch
from torch.nn import functional as F

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn.attention import (
    BlockTable,
    DenseInput,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache
from uniserve.runtime.backends.attention.torch import Backend

pytestmark = pytest.mark.integration


def _operator(cache=None, dtype=torch.float32):
    return Backend().prepare(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=dtype,
        size=TextSize(5, 2),
        cache=cache,
        workspace={},
    )


def _expected(q, k, v, allowed=None):
    return (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=allowed,
            scale=0.5,
            enable_gqa=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )


def test_dense_varlen_and_visible_attention_use_declared_ranges():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(5, 2, 4, generator=generator)
    k, v = (torch.randn(5, 1, 4, generator=generator) for _ in range(2))
    out = torch.empty_like(q)
    operator = _operator()
    batch = DenseInput(causal=True, mask=None)
    assert operator(q, k, v, batch, scale=0.5, out=out) is out
    torch.testing.assert_close(
        out, _expected(q, k, v, torch.ones(5, 5, dtype=torch.bool).tril())
    )
    allowed = torch.ones(5, 5, dtype=torch.bool)
    allowed[:, 1] = False
    operator(q, k, v, DenseInput(causal=True, mask=allowed), scale=0.5, out=out)
    torch.testing.assert_close(out, _expected(q, k, v, allowed.tril()))
    queries = SequenceLengths.from_lengths((2, 3), device="cpu")
    keys = SequenceLengths.from_lengths((3, 2), device="cpu")
    batch = VarlenInput(queries, keys, (True, False))
    operator(q, k, v, batch, scale=0.5, out=out)
    allowed = torch.tensor([[True, True, False], [True, True, True]])
    expected = torch.cat(
        (
            _expected(q[:2], k[:3], v[:3], allowed),
            _expected(q[2:], k[3:], v[3:]),
        )
    )
    torch.testing.assert_close(out, expected)
    ends = torch.tensor([[1, 3, 0], [0, 1, 2]], dtype=torch.int32)
    batch = VisibleInput(queries, keys, ends, None, False, False)
    operator(q, k, v, batch, scale=0.5, out=out)
    expected = torch.cat(
        (
            _expected(
                q[:2],
                k[:3],
                v[:3],
                torch.arange(3)[None, :] < ends[0, :2, None],
            ),
            _expected(
                q[2:], k[3:], v[3:], torch.arange(2)[None, :] < ends[1, :, None]
            ),
        )
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("quantized", [False, True])
def test_paged_attention_observes_updates_and_mutated_block_tables(quantized):
    config = mha.Config(1, 4, (0,), torch.float32)
    with PrefixCache(
        Config({"attention": config}),
        num_blocks=3,
        block_size=2,
        device="cpu",
        quantization={"attention": Quantizer("fp8", axis=0)}
        if quantized
        else None,
    ) as cache:
        state = cache.state("attention")
        operator = _operator(state)
        key = torch.arange(12).reshape(3, 1, 4).float() / 4
        value = key + 1
        state.write((1,), start=0, key=key[:2], value=value[:2])
        q = torch.ones(1, 2, 4)
        batch = PagedInput.from_blocks(
            blocks=((1, 0),),
            query_lengths=(1,),
            prefix_lengths=(2,),
            block_size=2,
            causal=True,
            device="cpu",
        )
        out = torch.empty_like(q)
        operator.bind(batch)
        operator(q, key[2:], value[2:], batch, scale=0.5, out=out)
        keys, values = state.read((1, 0), start=0, length=3)
        torch.testing.assert_close(out, _expected(q, keys, values))
        state.write((2,), start=0, key=key[:2], value=value[:2] * 4)
        batch.block_table.indices[0, 0] = 2
        operator.bind(batch)
        operator(q, key[2:], value[2:], batch, scale=0.5, out=out)
        keys, values = state.read((2, 0), start=0, length=3)
        torch.testing.assert_close(out, _expected(q, keys, values))


@pytest.mark.parametrize("prefix_length", [0, 1, 2, 3])
def test_segmented_attention_merges_visible_current_tokens_with_prefix(
    prefix_length,
):
    with PrefixCache(
        Config({"attention": mha.Config(1, 4, (0,), torch.float32)}),
        num_blocks=2,
        block_size=2,
        device="cpu",
    ) as cache:
        state = cache.state("attention")
        key = torch.arange((prefix_length + 2) * 4).view(-1, 1, 4).float() / 8
        value = key + 1
        state.write(
            (0, 1),
            start=0,
            key=key[:prefix_length],
            value=value[:prefix_length],
        )
        batch = SegmentedInput(
            SequenceLengths.from_lengths((2,), device="cpu"),
            SequenceLengths.from_lengths((prefix_length,), device="cpu"),
            BlockTable(torch.tensor([[0, 1]], dtype=torch.int32), 2),
            None,
            torch.tensor([[0, 2]], dtype=torch.int32),
            False,
        )
        query = torch.ones(2, 2, 4)
        out = torch.empty_like(query)
        _operator(state)(
            query,
            key[prefix_length:],
            value[prefix_length:],
            batch,
            scale=0.5,
            out=out,
        )
        allowed = torch.arange(prefix_length + 2)[None] < torch.tensor(
            [[prefix_length], [prefix_length + 2]]
        )
        torch.testing.assert_close(out, _expected(query, key, value, allowed))


def _windowed(window, cache=None):
    return Backend().prepare(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        size=TextSize(12, 2),
        cache=cache,
        workspace={},
        window=window,
    )


def _history(queries, keys, *, window, causal):
    """Visible [query, key] pairs for queries aligned to the key end."""
    positions = torch.arange(queries) + keys - queries
    columns = torch.arange(keys)[None]
    allowed = columns >= positions[:, None] - window
    if causal:
        allowed &= columns <= positions[:, None]
    return allowed


@pytest.mark.parametrize("window", [0, 2, 9])
@pytest.mark.parametrize("causal", [True, False])
def test_paged_window_follows_each_query_position(window, causal):
    generator = torch.Generator().manual_seed(23)
    key = torch.randn(8, 1, 4, generator=generator)
    value = torch.randn(8, 1, 4, generator=generator)
    query = torch.randn(3, 2, 4, generator=generator)
    with PrefixCache(
        Config({"attention": mha.Config(1, 4, (0,), torch.float32)}),
        num_blocks=5,
        block_size=2,
        device="cpu",
    ) as cache:
        state = cache.state("attention")
        # A five-token prefix is resident; the three query tokens append.
        state.write((3, 1, 4), start=0, key=key[:5], value=value[:5])
        batch = PagedInput.from_blocks(
            blocks=((3, 1, 4, 2),),
            query_lengths=(3,),
            prefix_lengths=(5,),
            block_size=2,
            causal=causal,
            device="cpu",
        )
        out = torch.empty_like(query)
        _windowed(window, state)(
            query, key[5:], value[5:], batch, scale=0.5, out=out
        )

    allowed = _history(3, 8, window=window, causal=causal)
    torch.testing.assert_close(out, _expected(query, key, value, allowed))


@pytest.mark.parametrize("window", [0, 1, 4])
def test_variable_length_and_dense_windows_bound_history(window):
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(5, 2, 4, generator=generator)
    k, v = (torch.randn(6, 1, 4, generator=generator) for _ in range(2))
    operator = _windowed(window)

    batch = VarlenInput(
        SequenceLengths.from_lengths((2, 3), device="cpu"),
        SequenceLengths.from_lengths((4, 2), device="cpu"),
        (True, False),
    )
    out = torch.empty_like(q)
    operator(q, k, v, batch, scale=0.5, out=out)
    expected = torch.cat(
        (
            _expected(
                q[:2], k[:4], v[:4], _history(2, 4, window=window, causal=True)
            ),
            _expected(
                q[2:], k[4:], v[4:], _history(3, 2, window=window, causal=False)
            ),
        )
    )
    torch.testing.assert_close(out, expected)

    operator(q, k[:5], v[:5], DenseInput(True, None), scale=0.5, out=out)
    torch.testing.assert_close(
        out,
        _expected(q, k[:5], v[:5], _history(5, 5, window=window, causal=True)),
    )


@pytest.mark.parametrize(("prefix_length", "window"), [(5, 3), (2, 3), (5, 0)])
def test_segmented_window_reads_one_prefix_interval(prefix_length, window):
    generator = torch.Generator().manual_seed(31)
    key = torch.randn(prefix_length + 3, 1, 4, generator=generator)
    value = torch.randn(prefix_length + 3, 1, 4, generator=generator)
    query = torch.randn(3, 2, 4, generator=generator)
    with PrefixCache(
        Config({"attention": mha.Config(1, 4, (0,), torch.float32)}),
        num_blocks=3,
        block_size=2,
        device="cpu",
    ) as cache:
        state = cache.state("attention")
        state.write(
            (0, 1, 2),
            start=0,
            key=key[:prefix_length],
            value=value[:prefix_length],
        )
        batch = SegmentedInput(
            SequenceLengths.from_lengths((3,), device="cpu"),
            SequenceLengths.from_lengths((prefix_length,), device="cpu"),
            BlockTable(torch.tensor([[0, 1, 2]], dtype=torch.int32), 2),
            None,
            torch.tensor([[3, 3, 3]], dtype=torch.int32),
            True,
        )
        out = torch.empty_like(query)
        _windowed(window, state)(
            query,
            key[prefix_length:],
            value[prefix_length:],
            batch,
            scale=0.5,
            out=out,
        )

    # Every query reads the same final `window` prefix tokens plus all of the
    # current tokens, independent of its own position.
    columns = torch.arange(prefix_length + 3)
    allowed = (
        (columns >= prefix_length - window) | (columns >= prefix_length)
    ).expand(3, -1)
    torch.testing.assert_close(out, _expected(query, key, value, allowed))


def test_windowed_attention_rejects_inputs_without_query_positions():
    queries = SequenceLengths.from_lengths((1,), device="cpu")
    batch = VisibleInput(
        queries, queries, torch.ones(1, 1, dtype=torch.int32), None, True, True
    )
    with pytest.raises(ValueError, match="windowed attention"):
        _windowed(2).bind(batch)
