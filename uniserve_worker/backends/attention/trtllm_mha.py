"""FlashInfer TRT-LLM MHA attention backend."""
from __future__ import annotations

from typing import Any, NamedTuple

import torch

from ...contracts.forward_context import get_forward_context
from ...foundation.runtime_config import get_worker_config
from .base import AttentionCapabilities
from .flashinfer_kernels import _decode_effective_seqlens, _write_decode_token
from .layout import QKVLayout, normalize_kv, normalize_to
from .registry import register_attention_backend

_flashinfer: Any | None
try:  # pragma: no cover - depends on optional CUDA package availability.
    import flashinfer as _flashinfer_module
except Exception:  # pragma: no cover
    _flashinfer = None
else:  # pragma: no cover
    _flashinfer = _flashinfer_module

_trtllm_decode = None
_trtllm_context = None
if _flashinfer is not None:  # pragma: no cover - availability-specific.
    _trtllm_decode = getattr(getattr(_flashinfer, "decode", None), "trtllm_batch_decode_with_kv_cache", None)
    _trtllm_context = getattr(getattr(_flashinfer, "prefill", None), "trtllm_batch_context_with_kv_cache", None)


class _PagedDecodeInputs(NamedTuple):
    q: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    restore: Any


class _VarlenPrefillInputs(NamedTuple):
    q: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    batch_size: int


class TRTLLMMHAAttentionBackend:
    """Paged decode and varlen prefill through FlashInfer TRT-LLM MHA kernels."""

    name = "trtllm_mha"

    def __init__(self) -> None:
        self._workspaces: dict[torch.device, torch.Tensor] = {}

    def capabilities(self) -> AttentionCapabilities:
        available = _trtllm_decode is not None and _trtllm_context is not None
        return AttentionCapabilities(
            available=available,
            paged_kv=available,
            varlen_attention=available,
            varlen_paged_kv=available,
            requires_paged_varlen=True,
            paged_block_size_multiple=1,
            min_head_dim=64,
            paged_decode_only=True,
            paged_varlen_cuda_graph=available,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("trtllm_mha requires paged KV metadata")

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        del causal
        if _trtllm_decode is None:
            raise RuntimeError("FlashInfer TRT-LLM MHA decode is not available")
        inputs = self._prepare_paged_decode_inputs(q, k_cache, v_cache, block_table, cache_seqlens)
        current_tokens = self._maybe_write_decode_token(
            k_cache,
            v_cache,
            inputs.block_table,
            inputs.cache_seqlens,
            inputs.q,
            k,
            v,
        )
        plan = get_forward_context().attention_plan
        effective_seqlens = _decode_effective_seqlens(
            inputs.cache_seqlens,
            current_tokens,
            plan,
        )
        page_size = int(k_cache.shape[1])
        max_seq_len = _metadata_context_len(max(1, int(inputs.block_table.shape[1]) * page_size))
        out = _trtllm_decode(
            query=inputs.q.contiguous(),
            kv_cache=_hnd_kv_cache(k_cache, v_cache),
            workspace_buffer=self._workspace(inputs.q.device),
            block_tables=inputs.block_table,
            seq_lens=effective_seqlens.to(dtype=torch.int32),
            max_seq_len=max_seq_len,
            bmm1_scale=float(scale),
            bmm2_scale=1.0,
            window_left=-1,
            out_dtype=inputs.q.dtype,
            kv_layout="HND",
        )
        return inputs.restore.apply(out)

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float,
        block_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _trtllm_context is None:
            raise RuntimeError("FlashInfer TRT-LLM MHA context is not available")
        inputs = self._prepare_varlen_prefill_inputs(q, block_table, cu_seqlens_q, cu_seqlens_k)
        _validate_paged_cache(inputs.q, k, v)
        kv_seqlens = (inputs.cu_seqlens_k[1:] - inputs.cu_seqlens_k[:-1]).to(dtype=torch.int32)
        out = _trtllm_context(
            query=inputs.q.contiguous(),
            kv_cache=_hnd_kv_cache(k, v),
            workspace_buffer=self._workspace(inputs.q.device),
            block_tables=inputs.block_table,
            seq_lens=kv_seqlens,
            max_q_len=max(1, int(max_seqlen_q)),
            max_kv_len=_metadata_context_len(max(1, int(max_seqlen_k))),
            bmm1_scale=float(scale),
            bmm2_scale=1.0,
            batch_size=inputs.batch_size,
            cum_seq_lens_q=inputs.cu_seqlens_q,
            cum_seq_lens_kv=inputs.cu_seqlens_k,
            window_left=-1,
            out_dtype=inputs.q.dtype,
            kv_layout="HND",
            causal=causal,
        )
        return out

    def _prepare_paged_decode_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> _PagedDecodeInputs:
        q_bhd, restore = normalize_to(q, QKVLayout.BHD)
        _validate_paged_cache(q_bhd, k_cache, v_cache)
        _require_sm100(q_bhd.device)
        return _PagedDecodeInputs(
            q=q_bhd,
            block_table=block_table.to(device=q_bhd.device, dtype=torch.int32).contiguous(),
            cache_seqlens=cache_seqlens.to(device=q_bhd.device, dtype=torch.int32).contiguous(),
            restore=restore,
        )

    def _prepare_varlen_prefill_inputs(
        self,
        q: torch.Tensor,
        block_table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> _VarlenPrefillInputs:
        if block_table is None:
            raise RuntimeError("trtllm_mha varlen path requires a paged KV block table")
        if q.ndim != 3:
            raise ValueError("trtllm_mha varlen expects q in [total, heads, dim] layout")
        _require_sm100(q.device)
        block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
        if int(cu_seqlens_q.numel()) != int(cu_seqlens_k.numel()):
            raise ValueError("q and k cu_seqlens must describe the same batch")
        batch_size = int(cu_seqlens_q.numel()) - 1
        if batch_size <= 0:
            raise ValueError("trtllm_mha varlen requires a non-empty batch")
        if int(block_table.shape[0]) != batch_size:
            raise ValueError("block table rows must match cu_seqlens batch size")
        return _VarlenPrefillInputs(q.contiguous(), block_table, cu_seqlens_q, cu_seqlens_k, batch_size)

    def _maybe_write_decode_token(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        q_bhd: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> int:
        if k is None and v is None:
            return 0
        if k is None or v is None:
            raise ValueError("trtllm_mha paged update requires both k and v")
        k_bhd = normalize_kv(k, QKVLayout.BHD, contiguous=False)
        v_bhd = normalize_kv(v, QKVLayout.BHD, contiguous=False)
        if k_bhd.shape != v_bhd.shape:
            raise ValueError("current paged K/V tensors must have matching shapes")
        if k_bhd.shape[0] != q_bhd.shape[0]:
            raise ValueError("current K/V batch size must match q batch size")
        if k_bhd.shape[1:] != k_cache.shape[2:]:
            raise ValueError("current K/V head geometry does not match paged cache")
        plan = get_forward_context().attention_plan
        _write_decode_token(k_cache, v_cache, block_table, cache_seqlens, k_bhd, v_bhd, plan)
        return 1

    def _workspace(self, device: torch.device | str) -> torch.Tensor:
        resolved = torch.device(device)
        workspace = self._workspaces.get(resolved)
        if workspace is None:
            workspace = torch.zeros(
                get_worker_config().flashinfer.workspace_size,
                dtype=torch.uint8,
                device=resolved,
            )
            self._workspaces[resolved] = workspace
        return workspace


def _validate_paged_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    if q.shape[0] <= 0:
        raise ValueError("trtllm_mha paged decode requires a non-empty batch")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("trtllm_mha paged cache expects k/v [pages, page, heads, dim]")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("query head dim does not match paged KV cache")


def _hnd_kv_cache(k_cache: torch.Tensor, v_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if k_cache.ndim != 4 or v_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        raise ValueError("trtllm_mha paged cache expects matching 4D k/v tensors")
    k_hnd = k_cache.permute(0, 2, 1, 3)
    v_hnd = v_cache.permute(0, 2, 1, 3)
    if int(k_hnd.shape[1]) == 1:
        k_hnd = _canonicalize_stride(k_hnd)
    if int(v_hnd.shape[1]) == 1:
        v_hnd = _canonicalize_stride(v_hnd)
    return k_hnd, v_hnd


def _metadata_context_len(default: int) -> int:
    plan = get_forward_context().attention_plan
    value = getattr(plan, "max_context_len", 0)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(1, parsed if parsed > 0 else int(default))


def _canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    sizes = tensor.size()
    strides = tensor.stride()
    if not any(sizes[idx] == 1 and strides[idx] == strides[idx + 1] for idx in range(tensor.dim() - 1)):
        return tensor
    new_strides = [0] * tensor.dim()
    new_strides[-1] = 1
    for idx in range(tensor.dim() - 2, -1, -1):
        new_strides[idx] = new_strides[idx + 1] * sizes[idx + 1]
    return tensor.as_strided(sizes, new_strides)


def _require_sm100(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("trtllm_mha requires CUDA tensors")
    major, minor = torch.cuda.get_device_capability(device)
    if (int(major), int(minor)) < (10, 0):
        raise RuntimeError("trtllm_mha requires compute capability 10.0 or newer")


if _flashinfer is not None:  # pragma: no cover - availability-specific.
    register_attention_backend("trtllm_mha", TRTLLMMHAAttentionBackend())
