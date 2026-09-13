"""FlashInfer TRT-LLM MHA attention backend."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from uniserve_worker.modeling.tensors import AttentionMetadata

from .base import AttentionBackend
from .flashinfer_kernels import _decode_effective_seqlens, _write_decode_token
from .layout import QKVLayout, normalize_kv, normalize_to
from .tuning import FlashInferTuningConfig

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
    _trtllm_decode = getattr(
        getattr(_flashinfer, "decode", None), "trtllm_batch_decode_with_kv_cache", None
    )
    _trtllm_context = getattr(
        getattr(_flashinfer, "prefill", None), "trtllm_batch_context_with_kv_cache", None
    )


class _PagedDecodeInputs(NamedTuple):
    """Groups normalized queries, page tables, sequence lengths, and layout restoration for paged decode."""

    q: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    restore: Any


class _VarlenPrefillInputs(NamedTuple):
    """Groups packed queries and cumulative sequence offsets for paged variable-length prefill."""

    q: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    batch_size: int


class TRTLLMMHAAttentionBackend(AttentionBackend):
    """Paged decode and varlen prefill through FlashInfer TRT-LLM MHA kernels."""

    name = "trtllm_mha"
    available = _trtllm_decode is not None and _trtllm_context is not None
    paged_varlen = available
    paged_varlen_only = True
    min_head_dim = 64
    single_ar_decode = True
    paged_varlen_cuda_graph = available
    cuda_only = True
    min_compute_version = (10, 0)
    max_compute_version = (10, 9)
    dense_ranks = frozenset()

    def __init__(self, *, tuning: FlashInferTuningConfig) -> None:
        """Configure the per-device TensorRT-LLM workspace capacity."""

        self._workspaces: dict[torch.device, torch.Tensor] = {}
        self._workspace_size = int(tuning.workspace_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Compute dense attention through the TensorRT-LLM MHA context kernel."""

        del context
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
        context: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Write current K/V and execute TensorRT-LLM paged decode on SM10x."""

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
            context,
        )
        plan = context
        effective_seqlens = _decode_effective_seqlens(
            inputs.cache_seqlens,
            current_tokens,
            plan,
        )
        page_size = int(k_cache.shape[1])
        max_seq_len = _metadata_context_len(
            plan,
            max(1, int(inputs.block_table.shape[1]) * page_size),
        )
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
        context: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Execute TensorRT-LLM paged context attention for packed variable-length queries."""

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
            max_kv_len=_metadata_context_len(
                context,
                max(1, int(max_seqlen_k)),
            ),
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
        """Normalize decode query and caches to the TensorRT-LLM paged layout."""

        q_bhd, restore = normalize_to(q, QKVLayout.BHD)
        _validate_paged_cache(q_bhd, k_cache, v_cache)
        _require_sm10x(q_bhd.device)
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
        """Validate packed query bounds and normalize the optional paged cache table."""

        if block_table is None:
            raise RuntimeError("trtllm_mha varlen path requires a paged KV block table")
        if q.ndim != 3:
            raise ValueError("trtllm_mha varlen expects q in [total, heads, dim] layout")
        _require_sm10x(q.device)
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
        return _VarlenPrefillInputs(
            q.contiguous(), block_table, cu_seqlens_q, cu_seqlens_k, batch_size
        )

    def _maybe_write_decode_token(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        q_bhd: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        plan: object | None,
    ) -> int:
        """Append an optional decode key/value row and return effective cache lengths."""

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
        _write_decode_token(k_cache, v_cache, block_table, cache_seqlens, k_bhd, v_bhd, plan)
        return 1

    def _workspace(self, device: torch.device | str) -> torch.Tensor:
        """Return or allocate this backend's persistent workspace on one device."""

        resolved = torch.device(device)
        workspace = self._workspaces.get(resolved)
        if workspace is None:
            workspace = torch.zeros(
                self._workspace_size,
                dtype=torch.uint8,
                device=resolved,
            )
            self._workspaces[resolved] = workspace
        return workspace


def _validate_paged_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    """Validate query and KV cache device, dtype, and rank compatibility."""

    if q.shape[0] <= 0:
        raise ValueError("trtllm_mha paged decode requires a non-empty batch")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("trtllm_mha paged cache expects k/v [pages, page, heads, dim]")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("query head dim does not match paged KV cache")


def _hnd_kv_cache(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert layer-page-token-head caches to TensorRT-LLM head-major layout."""

    if k_cache.ndim != 4 or v_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        raise ValueError("trtllm_mha paged cache expects matching 4D k/v tensors")
    k_hnd = k_cache.permute(0, 2, 1, 3)
    v_hnd = v_cache.permute(0, 2, 1, 3)
    if int(k_hnd.shape[1]) == 1:
        k_hnd = _canonicalize_stride(k_hnd)
    if int(v_hnd.shape[1]) == 1:
        v_hnd = _canonicalize_stride(v_hnd)
    return k_hnd, v_hnd


def _metadata_context_len(plan: object | None, default: int) -> int:
    """Read a plan context bound or use the supplied default."""

    value = getattr(plan, "max_seqlen_k", 0)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(1, parsed if parsed > 0 else int(default))


def _canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a contiguous tensor when its stride cannot be represented canonically."""

    sizes = tensor.size()
    strides = tensor.stride()
    if not any(
        sizes[idx] == 1 and strides[idx] == strides[idx + 1] for idx in range(tensor.dim() - 1)
    ):
        return tensor
    new_strides = [0] * tensor.dim()
    new_strides[-1] = 1
    for idx in range(tensor.dim() - 2, -1, -1):
        new_strides[idx] = new_strides[idx + 1] * sizes[idx + 1]
    return tensor.as_strided(sizes, new_strides)


def _require_sm10x(device: torch.device) -> None:
    """Require an indexed CUDA device supported by TRTLLM-Gen FMHA."""

    if device.type != "cuda":
        raise RuntimeError("trtllm_mha requires CUDA tensors")
    major, minor = torch.cuda.get_device_capability(device)
    if int(major) != 10:
        raise RuntimeError("trtllm_mha requires compute capability 10.x")
