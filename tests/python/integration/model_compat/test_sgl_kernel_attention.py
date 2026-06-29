"""SGL kernel attention backend tests."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.registry import get_attention_backend, has_attention_backend

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sgl_kernel_paged_decode_matches_flashinfer_and_updates_cache():
    if not has_attention_backend("sgl_kernel") or not has_attention_backend("flashinfer"):
        pytest.skip("optional paged attention backends are unavailable")
    sgl = get_attention_backend("sgl_kernel")
    flashinfer = get_attention_backend("flashinfer")
    if not sgl.capabilities().paged_kv or not flashinfer.capabilities().paged_kv:
        pytest.skip("paged KV kernels are unavailable")

    device = torch.device("cuda")
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    page_size = 256
    cache_len = 17
    block_table = torch.arange(batch, dtype=torch.int32, device=device).view(batch, 1)
    cache_seqlens = torch.full((batch,), cache_len, dtype=torch.int32, device=device)
    torch.manual_seed(3)
    q = torch.randn((batch, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((batch, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((batch, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    k_cache = torch.randn((batch, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()

    kwargs = {
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "k": k,
        "v": v,
        "causal": True,
        "scale": head_dim**-0.5,
    }
    expected = flashinfer.forward_paged(q, k_cache_ref, v_cache_ref, **kwargs)
    actual = sgl.forward_paged(q, k_cache, v_cache, **kwargs)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_cache[:, cache_len], k, atol=0, rtol=0)
    torch.testing.assert_close(v_cache[:, cache_len], v, atol=0, rtol=0)
