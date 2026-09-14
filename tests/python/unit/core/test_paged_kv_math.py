from __future__ import annotations

import pytest
import torch

from uniserve.runtime import paged_kv_math


@pytest.mark.parametrize("encoded", (False, True))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_kv_write_replays_dynamic_rows_in_cuda_graph(encoded, index_dtype):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(71)
    pages, page_size, heads, head_dim = 5, 4, 8, 128
    rows = 3
    initial_k = torch.randn(
        pages,
        page_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    initial_v = torch.randn(
        pages,
        page_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_cache = initial_k.clone()
    v_cache = initial_v.clone()
    row_width = heads * head_dim
    k_storage = torch.randn(
        rows,
        row_width + 32,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v_storage = torch.randn(
        rows,
        row_width + 48,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_current = k_storage[:, 16 : 16 + row_width].view(rows, heads, head_dim)
    v_current = v_storage[:, 24 : 24 + row_width].view(rows, heads, head_dim)
    assert not k_current.is_contiguous()
    assert not v_current.is_contiguous()
    page_ids = torch.tensor([-1, 2, 4], dtype=index_dtype, device=device)
    offsets = torch.tensor([0, 3, 0], dtype=index_dtype, device=device)

    # Encoded slots reserve zero as a non-writing sentinel. Page/offset
    # addressing instead masks negative page IDs; both forms share K/V geometry.
    locations = torch.tensor([0, 11, 16], dtype=index_dtype, device=device) if encoded else page_ids
    address_offsets = None if encoded else offsets

    with torch.inference_mode():
        paged_kv_math.paged_kv_write(
            k_cache,
            v_cache,
            locations,
            address_offsets,
            k_current,
            v_current,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    k_cache.copy_(initial_k)
    v_cache.copy_(initial_v)
    with torch.cuda.graph(graph), torch.inference_mode():
        paged_kv_math.paged_kv_write(
            k_cache,
            v_cache,
            locations,
            address_offsets,
            k_current,
            v_current,
        )

    replay_k = torch.randn(
        k_current.shape,
        dtype=k_current.dtype,
        device=device,
        generator=generator,
    )
    replay_v = torch.randn(
        v_current.shape,
        dtype=v_current.dtype,
        device=device,
        generator=generator,
    )
    replay_page_ids = torch.tensor([3, -1, 4], dtype=index_dtype, device=device)
    replay_offsets = torch.tensor([2, 0, 3], dtype=index_dtype, device=device)
    k_cache.copy_(initial_k)
    v_cache.copy_(initial_v)
    k_current.copy_(replay_k)
    v_current.copy_(replay_v)
    page_ids.copy_(replay_page_ids)
    offsets.copy_(replay_offsets)
    if encoded:
        locations.copy_(torch.tensor([14, -1, 19], dtype=index_dtype, device=device))

    expected_k = initial_k.clone()
    expected_v = initial_v.clone()
    persisted_rows = torch.tensor([0, 2], dtype=torch.int64, device=device)
    flat_index = (
        replay_page_ids.index_select(0, persisted_rows) * page_size
        + replay_offsets.index_select(0, persisted_rows)
    ).to(dtype=torch.int64)
    expected_k.view(-1, heads, head_dim).index_copy_(
        0, flat_index, replay_k.index_select(0, persisted_rows)
    )
    expected_v.view(-1, heads, head_dim).index_copy_(
        0, flat_index, replay_v.index_select(0, persisted_rows)
    )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(k_cache, expected_k)
    assert torch.equal(v_cache, expected_v)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("cast", (False, True))
def test_encoded_kv_locations_preserve_unwritten_rows(device, cast):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    keys = torch.full((3, 4, 2, 3), -7, dtype=torch.bfloat16, device=device)
    values = torch.full_like(keys, -9)
    source_dtype = torch.float32 if cast else torch.bfloat16
    source_keys = torch.arange(30, dtype=source_dtype, device=device).reshape(5, 2, 3) / 7
    source_values = -source_keys
    locations = torch.tensor([0, -2, 4, 7, 11], device=device)
    expected_keys = keys.clone()
    expected_values = values.clone()
    for row, (page, offset) in enumerate(((1, 0), (1, 3), (2, 3)), start=2):
        expected_keys[page, offset] = source_keys[row]
        expected_values[page, offset] = source_values[row]

    with torch.inference_mode():
        paged_kv_math.paged_kv_write(
            keys, values, locations, None, source_keys, source_values, cast=cast
        )

    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
