from types import SimpleNamespace

import pytest
import torch

from uniserve.attention.base import merge_attention_states
from uniserve.attention.flashinfer import FlashInferAttentionBackend
from uniserve.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve.attention.tuning import FlashInferTuningConfig


@pytest.mark.parametrize("write_current", [False, True])
def test_paged_decode_includes_current_token(write_current: bool) -> None:
    generator = torch.Generator().manual_seed(31)
    prefix, heads, width, page_size = 3, 2, 8, 4
    query = torch.randn((1, heads, 1, width), generator=generator)
    keys = torch.randn((prefix + 1, heads, width), generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    key_cache = torch.zeros((2, page_size, heads, width))
    value_cache = torch.zeros_like(key_cache)
    resident = prefix if write_current else prefix + 1
    key_cache[1, :resident].copy_(keys[:resident])
    value_cache[1, :resident].copy_(values[:resident])
    context = SimpleNamespace(prefix_lens_cpu=(prefix,), seq_lens_cpu=(prefix + 1,))

    actual = TorchSDPAAttentionBackend().forward_paged(
        query,
        key_cache,
        value_cache,
        block_table=torch.tensor([[1]], dtype=torch.int32),
        cache_seqlens=torch.tensor([prefix + 1], dtype=torch.int32),
        k=keys[-1:] if write_current else None,
        v=values[-1:] if write_current else None,
        causal=False,
        scale=width**-0.5,
        context=context,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_state_merge_matches_reference_during_cuda_graph_replay() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(5)
    shape = (7, 4, 64)
    first_output = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    second_output = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    first_lse = torch.randn(shape[:-1], device=device, dtype=torch.float32, generator=generator)
    second_lse = torch.randn(shape[:-1], device=device, dtype=torch.float32, generator=generator)

    with torch.inference_mode():
        merge_attention_states(first_output, first_lse, second_output, second_lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        actual_output, actual_lse = merge_attention_states(
            first_output, first_lse, second_output, second_lse
        )

    first_output.copy_(torch.randn(shape, device=device, dtype=torch.float16, generator=generator))
    second_output.copy_(torch.randn(shape, device=device, dtype=torch.float16, generator=generator))
    first_lse.copy_(
        torch.randn(shape[:-1], device=device, dtype=torch.float32, generator=generator)
    )
    second_lse.copy_(
        torch.randn(shape[:-1], device=device, dtype=torch.float32, generator=generator)
    )
    first_lse[0, 0] = -torch.inf
    second_lse[0, 0] = -torch.inf
    expected_lse = torch.logaddexp(first_lse, second_lse)
    first_weight = torch.exp(first_lse - expected_lse).nan_to_num(0.0)
    second_weight = torch.exp(second_lse - expected_lse).nan_to_num(0.0)
    expected_output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ).to(first_output.dtype)

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_paged_prefix_dense_current_matches_concatenated_attention(causal: bool) -> None:
    torch.manual_seed(7)
    page_size = 4
    prefix_lens = (0, page_size - 1, page_size, page_size + 1)
    query_len = 3
    query_heads = 4
    kv_heads = 2
    head_dim = 8
    rows = len(prefix_lens)
    pages_per_row = 2
    cache = torch.zeros((1 + rows * pages_per_row, page_size, kv_heads, head_dim))
    value_cache = torch.zeros_like(cache)
    page_table = torch.zeros((rows, pages_per_row), dtype=torch.int32)
    dense_prefixes: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row, prefix_len in enumerate(prefix_lens):
        pages = torch.arange(
            1 + row * pages_per_row,
            1 + (row + 1) * pages_per_row,
            dtype=torch.int32,
        )
        page_table[row] = pages
        key = torch.randn(prefix_len, kv_heads, head_dim)
        value = torch.randn_like(key)
        dense_prefixes.append((key, value))
        if prefix_len:
            cache.index_copy_(0, pages.to(torch.long), _padded_pages(key, pages_per_row, page_size))
            value_cache.index_copy_(
                0, pages.to(torch.long), _padded_pages(value, pages_per_row, page_size)
            )

    query = torch.randn(rows * query_len, query_heads, head_dim)
    current_key = torch.randn(rows * query_len, kv_heads, head_dim)
    current_value = torch.randn_like(current_key)
    offsets = torch.arange(0, (rows + 1) * query_len, query_len, dtype=torch.int32)
    visible = torch.full((rows, query_len), query_len, dtype=torch.int32)
    if causal:
        visible[:] = torch.arange(1, query_len + 1, dtype=torch.int32)
    context = SimpleNamespace(
        query_lens_cpu=(query_len,) * rows,
        prefix_lens_cpu=prefix_lens,
    )
    backend = TorchSDPAAttentionBackend()
    actual = backend.forward_segmented(
        query,
        current_key,
        current_value,
        cache,
        value_cache,
        page_table=page_table,
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        cu_seqlens_q=offsets,
        visible_current_end=visible,
        scale=head_dim**-0.5,
        fully_visible_current=not causal,
        context=context,
    )

    expected = []
    for row, (prefix_key, prefix_value) in enumerate(dense_prefixes):
        begin = row * query_len
        end = begin + query_len
        expected.append(
            backend.forward(
                query[begin:end],
                torch.cat((prefix_key, current_key[begin:end])),
                torch.cat((prefix_value, current_value[begin:end])),
                causal=causal,
                scale=head_dim**-0.5,
            )
        )
    torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("prefix_lens", [(2, 5), (0, 15)])
def test_flashinfer_segmented_attention_matches_concatenated_attention(prefix_lens) -> None:
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.float16
    page_size = 16
    query_lens = (1, 3)
    causal_rows = (True, False)
    query_heads = 4
    kv_heads = 2
    head_dim = 64
    cache = torch.zeros((2, page_size, kv_heads, head_dim), device=device, dtype=dtype)
    value_cache = torch.zeros_like(cache)
    dense_prefixes: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row, prefix_len in enumerate(prefix_lens):
        key = torch.randn(prefix_len, kv_heads, head_dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        cache[row, :prefix_len].copy_(key)
        value_cache[row, :prefix_len].copy_(value)
        dense_prefixes.append((key, value))

    total_query = sum(query_lens)
    query = torch.randn(total_query, query_heads, head_dim, device=device, dtype=dtype)
    current_key = torch.randn(total_query, kv_heads, head_dim, device=device, dtype=dtype)
    current_value = torch.randn_like(current_key)
    offsets = torch.tensor((0, query_lens[0], total_query), device=device, dtype=torch.int32)
    context = SimpleNamespace(
        query_lens_cpu=query_lens,
        query_lens=torch.tensor(query_lens, device=device, dtype=torch.int32),
        prefix_lens_cpu=prefix_lens,
        seq_lens_cpu=tuple(prefix + current for prefix, current in zip(prefix_lens, query_lens)),
        seq_lens=torch.tensor(
            [prefix + current for prefix, current in zip(prefix_lens, query_lens)],
            device=device,
            dtype=torch.int32,
        ),
        causal_rows_cpu=causal_rows,
        binding=17,
    )
    backend = FlashInferAttentionBackend(
        tuning=FlashInferTuningConfig(workspace_size=64 * 1024 * 1024)
    )
    actual = backend.forward_segmented(
        query,
        current_key,
        current_value,
        cache,
        value_cache,
        page_table=torch.tensor(((0,), (1,)), device=device, dtype=torch.int32),
        prefix_lens=torch.tensor(prefix_lens, device=device, dtype=torch.int32),
        cu_seqlens_q=offsets,
        visible_current_end=torch.tensor(((1, 0, 0), (3, 3, 3)), device=device, dtype=torch.int32),
        scale=head_dim**-0.5,
        fully_visible_current=False,
        context=context,
    )

    expected = []
    for row, (begin, end) in enumerate(
        zip(offsets.tolist()[:-1], offsets.tolist()[1:], strict=True)
    ):
        prefix_key, prefix_value = dense_prefixes[row]
        expected.append(
            flashinfer.single_prefill_with_kv_cache(
                query[begin:end],
                torch.cat((prefix_key, current_key[begin:end])),
                torch.cat((prefix_value, current_value[begin:end])),
                causal=causal_rows[row],
                sm_scale=head_dim**-0.5,
            )
        )
    torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-2, atol=2e-2)


def _padded_pages(value: torch.Tensor, page_count: int, page_size: int) -> torch.Tensor:
    padded = value.new_zeros((page_count * page_size, *value.shape[1:]))
    padded[: value.shape[0]] = value
    return padded.view(page_count, page_size, *value.shape[1:])
