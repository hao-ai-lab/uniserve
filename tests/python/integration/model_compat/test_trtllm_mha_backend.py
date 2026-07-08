from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention import get_attention_backend, has_attention_backend

pytestmark = pytest.mark.integration

KERNEL_ATOL = 0.01
KERNEL_RTOL = 0.01


def _trtllm_mha_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA device not available"
    major, minor = torch.cuda.get_device_capability(torch.device("cuda"))
    if (int(major), int(minor)) < (10, 0):
        return "compute capability 10.0 or newer required"
    if not has_attention_backend("trtllm_mha"):
        return "trtllm_mha attention backend not registered"
    backend = get_attention_backend("trtllm_mha")
    if not bool(getattr(backend.capabilities(), "available", False)):
        return "FlashInfer TRT-LLM MHA kernels unavailable"
    return None


def _require_trtllm_mha() -> None:
    reason = _trtllm_mha_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)


@torch.inference_mode()
def test_trtllm_mha_paged_decode_updates_cache_and_matches_reference() -> None:
    _require_trtllm_mha()
    torch.manual_seed(9001)
    backend = get_attention_backend("trtllm_mha")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    page_size = 64
    pages = 8
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([5, 70], dtype=torch.int32, device=device)
    q = torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype)
    k_cache = torch.zeros(pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    for row in range(batch):
        for pos in range(int(cache_seqlens[row].item())):
            page = int(block_table[row, pos // page_size].item())
            offset = pos % page_size
            k_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)
            v_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)

    out = backend.forward_paged(q, k_cache, v_cache, block_table=block_table, cache_seqlens=cache_seqlens, k=k, v=v, causal=True, scale=head_dim**-0.5)
    torch.cuda.synchronize()

    expected = _decode_reference(q, k_cache, v_cache, block_table, cache_seqlens + 1, page_size)
    torch.testing.assert_close(out.float(), expected.float(), atol=KERNEL_ATOL, rtol=KERNEL_RTOL)


@torch.inference_mode()
def test_trtllm_mha_paged_varlen_prefill_matches_causal_reference() -> None:
    _require_trtllm_mha()
    torch.manual_seed(9002)
    backend = get_attention_backend("trtllm_mha")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    page_size = 64
    pages = 8
    query_lens = (3, 2)
    base_lens = (5, 70)
    kv_lens = tuple(base + query for base, query in zip(base_lens, query_lens))
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    q = torch.randn(sum(query_lens), q_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.zeros(pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    for row, kv_len in enumerate(kv_lens):
        for pos in range(kv_len):
            page = int(block_table[row, pos // page_size].item())
            offset = pos % page_size
            k_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)
            v_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)
    cu_q = torch.tensor([0, query_lens[0], sum(query_lens)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32, device=device)

    out = backend.forward_varlen(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(query_lens), max_seqlen_k=max(kv_lens), causal=True, scale=head_dim**-0.5, block_table=block_table)
    torch.cuda.synchronize()

    expected = _prefill_reference(q, k_cache, v_cache, block_table, page_size, base_lens, query_lens)
    torch.testing.assert_close(out.float(), expected.float(), atol=KERNEL_ATOL, rtol=KERNEL_RTOL)


@torch.inference_mode()
def test_trtllm_mha_paged_varlen_prefill_accepts_zero_query_rows() -> None:
    _require_trtllm_mha()
    torch.manual_seed(9003)
    backend = get_attention_backend("trtllm_mha")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    page_size = 64
    pages = 10
    query_lens = (3, 0, 2)
    base_lens = (5, 70, 73)
    kv_lens = tuple(base + query for base, query in zip(base_lens, query_lens))
    block_table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device=device)
    q = torch.randn(sum(query_lens), q_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.zeros(pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    for row, kv_len in enumerate(kv_lens):
        for pos in range(kv_len):
            page = int(block_table[row, pos // page_size].item())
            offset = pos % page_size
            k_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)
            v_cache[page, offset] = torch.randn(kv_heads, head_dim, device=device, dtype=dtype)
    cu_q = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, kv_lens[0], kv_lens[0] + kv_lens[1], sum(kv_lens)], dtype=torch.int32, device=device)

    out = backend.forward_varlen(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(query_lens), max_seqlen_k=max(kv_lens), causal=True, scale=head_dim**-0.5, block_table=block_table)
    torch.cuda.synchronize()

    expected = _prefill_reference(q, k_cache, v_cache, block_table, page_size, base_lens, query_lens)
    torch.testing.assert_close(out.float(), expected.float(), atol=KERNEL_ATOL, rtol=KERNEL_RTOL)


def _decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    rows = []
    q_heads = int(q.shape[1])
    kv_heads = int(k_cache.shape[2])
    group = q_heads // kv_heads
    scale = int(q.shape[-1]) ** -0.5
    for row in range(int(q.shape[0])):
        keys, values = _sequence_kv(k_cache, v_cache, block_table[row], int(kv_lens[row].item()), page_size)
        heads = []
        for head in range(q_heads):
            kv_head = head // group
            scores = (q[row, head, 0].float()[None, :] * keys[:, kv_head, :]).sum(-1) * scale
            weights = torch.softmax(scores, dim=0)
            heads.append((weights[:, None] * values[:, kv_head, :]).sum(0))
        rows.append(torch.stack(heads, dim=0))
    return torch.stack(rows, dim=0).to(q.dtype).unsqueeze(2)


def _prefill_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    base_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
) -> torch.Tensor:
    rows = []
    q_heads = int(q.shape[1])
    kv_heads = int(k_cache.shape[2])
    group = q_heads // kv_heads
    scale = int(q.shape[-1]) ** -0.5
    q_offset = 0
    for batch_row, query_len in enumerate(query_lens):
        keys, values = _sequence_kv(k_cache, v_cache, block_table[batch_row], int(base_lens[batch_row] + query_len), page_size)
        for query_offset in range(query_len):
            visible = int(base_lens[batch_row] + query_offset + 1)
            heads = []
            for head in range(q_heads):
                kv_head = head // group
                scores = (q[q_offset + query_offset, head].float()[None, :] * keys[:visible, kv_head, :]).sum(-1) * scale
                weights = torch.softmax(scores, dim=0)
                heads.append((weights[:, None] * values[:visible, kv_head, :]).sum(0))
            rows.append(torch.stack(heads, dim=0))
        q_offset += query_len
    return torch.stack(rows, dim=0).to(q.dtype)


def _sequence_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_row: torch.Tensor,
    length: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = []
    values = []
    for pos in range(int(length)):
        page = int(block_row[pos // int(page_size)].item())
        offset = pos % int(page_size)
        keys.append(k_cache[page, offset].float())
        values.append(v_cache[page, offset].float())
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)
