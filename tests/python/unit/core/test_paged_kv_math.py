from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends import paged_kv_math
from uniserve_worker.foundation.triton_compat import triton_device_supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_paged_kv_write_replays_dynamic_rows_in_cuda_graph():
    if paged_kv_math.triton is None or not triton_device_supported(torch.device("cuda")):
        pytest.skip("Triton fused layers are not supported on this CUDA device")

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
    page_ids = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    offsets = torch.tensor([1, 3, 0], dtype=torch.int32, device=device)

    with torch.inference_mode():
        assert paged_kv_math._triton_paged_kv_write_eligible(
            k_cache,
            v_cache,
            page_ids,
            offsets,
            k_current,
            v_current,
        )
        paged_kv_math.paged_kv_write(
            k_cache,
            v_cache,
            page_ids,
            offsets,
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
            page_ids,
            offsets,
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
    replay_page_ids = torch.tensor([3, 1, 4], dtype=torch.int32, device=device)
    replay_offsets = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
    k_cache.copy_(initial_k)
    v_cache.copy_(initial_v)
    k_current.copy_(replay_k)
    v_current.copy_(replay_v)
    page_ids.copy_(replay_page_ids)
    offsets.copy_(replay_offsets)

    expected_k = initial_k.clone()
    expected_v = initial_v.clone()
    flat_index = (replay_page_ids * page_size + replay_offsets).to(dtype=torch.int64)
    expected_k.view(-1, heads, head_dim).index_copy_(0, flat_index, replay_k)
    expected_v.view(-1, heads, head_dim).index_copy_(0, flat_index, replay_v)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(k_cache, expected_k)
    assert torch.equal(v_cache, expected_v)
