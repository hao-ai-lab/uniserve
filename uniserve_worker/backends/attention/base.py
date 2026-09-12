"""Attention backend interface."""

from __future__ import annotations

import torch

from ...execution.forward_batch import AttentionMetadata, AttentionMode
from ..triton import triton_available

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

__all__ = [
    "AttentionBackend",
    "merge_attention_states",
]


if triton is not None:

    @triton.jit
    def _merge_attention_states_kernel(
        first_output_ptr,
        first_lse_ptr,
        second_output_ptr,
        second_lse_ptr,
        output_ptr,
        merged_lse_ptr,
        state_count,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        """Merge two independently normalized attention states with online-softmax rescaling."""

        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        columns = tl.arange(0, block_dim)
        row_mask = rows < state_count
        first_lse = tl.load(first_lse_ptr + rows, mask=row_mask, other=-float("inf"))
        second_lse = tl.load(second_lse_ptr + rows, mask=row_mask, other=-float("inf"))
        maximum = tl.maximum(first_lse, second_lse)
        both_empty = (first_lse == -float("inf")) & (second_lse == -float("inf"))
        first_scale = tl.exp(first_lse - maximum)
        second_scale = tl.exp(second_lse - maximum)
        denominator = first_scale + second_scale
        first_weight = tl.where(both_empty, 0.0, first_scale / denominator)
        second_weight = tl.where(both_empty, 0.0, second_scale / denominator)
        merged_lse = tl.where(both_empty, -float("inf"), maximum + tl.log(denominator))

        offsets = rows[:, None] * head_dim + columns[None, :]
        mask = row_mask[:, None] & (columns[None, :] < head_dim)
        first_output = tl.load(first_output_ptr + offsets, mask=mask, other=0.0)
        second_output = tl.load(second_output_ptr + offsets, mask=mask, other=0.0)
        merged = (
            first_output.to(tl.float32) * first_weight[:, None]
            + second_output.to(tl.float32) * second_weight[:, None]
        )
        tl.store(output_ptr + offsets, merged, mask=mask)
        tl.store(merged_lse_ptr + rows, merged_lse, mask=row_mask)


def merge_attention_states(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently evaluated KV segments with stable online softmax."""

    if first_output.shape != second_output.shape or first_lse.shape != second_lse.shape:
        raise ValueError("attention states must have matching shapes")
    if _triton_merge_eligible(first_output, first_lse, second_output, second_lse):
        output = torch.empty_like(first_output)
        merged_lse = torch.empty_like(first_lse)
        state_count = int(first_lse.numel())
        head_dim = int(first_output.shape[-1])
        block_dim = triton.next_power_of_2(head_dim)
        block_rows = max(1, min(8, 1024 // block_dim))
        _merge_attention_states_kernel[(triton.cdiv(state_count, block_rows),)](
            first_output,
            first_lse,
            second_output,
            second_lse,
            output,
            merged_lse,
            state_count,
            head_dim,
            block_rows,
            block_dim,
            num_warps=8,
        )
        return output, merged_lse
    merged_lse = torch.logaddexp(first_lse, second_lse)
    first_weight = torch.exp(first_lse - merged_lse).nan_to_num(0.0)
    second_weight = torch.exp(second_lse - merged_lse).nan_to_num(0.0)
    output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ).to(first_output.dtype)
    return output, merged_lse


def _triton_merge_eligible(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> bool:
    """Return whether two attention states satisfy the fused merge kernel contract."""

    tensors = (first_output, first_lse, second_output, second_lse)
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and first_output.ndim == first_lse.ndim + 1
        and tuple(first_output.shape[:-1]) == tuple(first_lse.shape)
        and int(first_output.shape[-1]) > 0
        and int(first_lse.numel()) > 0
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and first_output.dtype == second_output.dtype
        and first_lse.dtype == second_lse.dtype
        and first_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and first_lse.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and triton_available(first_output.device)
    )


class AttentionBackend:
    """Base class for attention implementations selected at worker startup."""

    name = "attention"
    available: bool = True
    paged_varlen: bool = False
    paged_varlen_only: bool = False
    packed_cuda_graph: bool = False
    paged_varlen_cuda_graph: bool = False
    page_size_multiple: int = 1
    min_head_dim: int = 1
    single_ar_decode: bool = False
    head_geometries: frozenset[tuple[int, int, int]] = frozenset()
    cuda_only: bool = False
    min_compute_version: tuple[int, int] | None = None
    max_compute_version: tuple[int, int] | None = None
    dense_ranks: frozenset[int] = frozenset({3, 4})
    accepts_dense_mask: bool = False
    dense_dtypes: frozenset[torch.dtype] = frozenset()

    def supports_head_geometry(self, q_head_dim: int, k_head_dim: int, v_head_dim: int) -> bool:
        """Return whether the backend implements the supplied query, key, and value head widths."""

        if not self.head_geometries:
            return True
        return (int(q_head_dim), int(k_head_dim), int(v_head_dim)) in self.head_geometries

    def supports(self, mode: AttentionMode, *, cuda_graph: bool = False) -> bool:
        """Return whether this backend implements the mode, including its CUDA graph contract when requested."""

        if not self.available:
            return False
        implementation = type(self)
        if mode is AttentionMode.DENSE:
            supported = (
                bool(self.dense_ranks) and implementation.forward is not AttentionBackend.forward
            )
        elif mode is AttentionMode.PAGED_DECODE:
            supported = implementation.forward_paged is not AttentionBackend.forward_paged
        elif mode is AttentionMode.PAGED_VARLEN:
            supported = (
                self.paged_varlen
                and implementation.forward_varlen is not AttentionBackend.forward_varlen
            )
        elif mode is AttentionMode.PACKED:
            supported = implementation.forward_segmented is not AttentionBackend.forward_segmented
        else:
            supported = False
        if not supported or not cuda_graph:
            return supported
        if mode is AttentionMode.PACKED:
            return self.packed_cuda_graph
        if mode is AttentionMode.PAGED_VARLEN:
            return self.paged_varlen_cuda_graph or (
                callable(getattr(self, "bind_paged_prefill_graph_wrapper", None))
                and callable(getattr(self, "prepare_paged_prefill_cuda_graph", None))
            )
        return True

    def supports_varlen(self) -> bool:
        """Return whether this backend exposes a non-paged variable-length path."""

        return (
            self.available
            and not self.paged_varlen_only
            and type(self).forward_varlen is not AttentionBackend.forward_varlen
        )

    def can_bind(
        self,
        mode: AttentionMode,
        *,
        head_dim: int,
        block_size: int,
        device: torch.device,
        cuda_graph: bool = False,
    ) -> bool:
        """Check static head, page, device, and graph constraints before binding the backend."""

        if not self.supports(mode, cuda_graph=cuda_graph):
            return False
        return self._bind_geometry_supported(head_dim, block_size, device)

    def can_bind_varlen(
        self,
        *,
        head_dim: int,
        block_size: int,
        device: torch.device,
    ) -> bool:
        """Check static head, page, and device constraints for variable-length attention."""

        if not self.supports_varlen():
            return False
        return self._bind_geometry_supported(head_dim, block_size, device)

    def _bind_geometry_supported(
        self, head_dim: int, block_size: int, device: torch.device
    ) -> bool:
        """Return whether graph binding supports the requested head, page, and device geometry."""

        if int(head_dim) < int(self.min_head_dim):
            return False
        if not self.supports_head_geometry(head_dim, head_dim, head_dim):
            return False
        if int(block_size) % max(1, int(self.page_size_multiple)) != 0:
            return False
        if self.cuda_only and device.type != "cuda":
            return False
        if self.min_compute_version is not None:
            if device.type != "cuda":
                return False
            if torch.cuda.get_device_capability(device) < self.min_compute_version:
                return False
        if self.max_compute_version is not None:
            if device.type != "cuda":
                return False
            if torch.cuda.get_device_capability(device) > self.max_compute_version:
                return False
        return True

    def can_run(self, req: object) -> bool:
        """Validate a typed attention request against backend capabilities and tensor geometry."""

        from ...ops.requests import (
            DenseAttention,
            PagedDecodeAttention,
            VarlenAttention,
            VisibleEndAttention,
        )

        q = getattr(req, "q", None)
        if not self.available or not isinstance(q, torch.Tensor) or not self._device_supported(q):
            return False
        if int(q.shape[-1]) < int(self.min_head_dim):
            return False
        k_dim, v_dim = self._kv_dims(req)
        if not self.supports_head_geometry(int(q.shape[-1]), k_dim, v_dim):
            return False
        if isinstance(req, VisibleEndAttention):
            if req.prefix_k is not None:
                return (
                    self.supports(AttentionMode.PACKED)
                    and (
                        not bool(getattr(req.ctx, "cuda_graph_capture", False))
                        or self.packed_cuda_graph
                    )
                    and self._paged_storage_supported(req)
                )
            return type(self).forward_visible_end is not AttentionBackend.forward_visible_end
        if isinstance(req, VarlenAttention):
            if req.block_table is not None:
                return self.supports(AttentionMode.PAGED_VARLEN) and self._paged_storage_supported(
                    req
                )
            return self.supports_varlen()
        if isinstance(req, PagedDecodeAttention):
            if self.single_ar_decode and not self._is_one_ar_decode(req):
                return False
            return self.supports(AttentionMode.PAGED_DECODE) and self._paged_storage_supported(req)
        if not isinstance(req, DenseAttention):
            return False
        if not self.supports(AttentionMode.DENSE):
            return False
        if req.attn_mask is not None and not self.accepts_dense_mask:
            return False
        if req.q.ndim != req.k.ndim or req.q.ndim != req.v.ndim:
            return False
        if self.dense_dtypes and any(
            value.dtype not in self.dense_dtypes for value in (req.q, req.k, req.v)
        ):
            return False
        return int(req.q.ndim) in self.dense_ranks

    def run(self, req: object) -> torch.Tensor:
        """Dispatch a validated typed request to the matching dense, paged, variable-length, or segmented path."""

        from ...ops.requests import (
            DenseAttention,
            PagedDecodeAttention,
            VarlenAttention,
            VisibleEndAttention,
        )

        if not self.can_run(req):
            raise RuntimeError(
                f"bound attention backend {self.name!r} rejects the request geometry"
            )
        if isinstance(req, VisibleEndAttention) and req.prefix_k is not None:
            if (
                req.prefix_v is None
                or req.prefix_lens is None
                or req.cu_seqlens_q is None
                or req.page_table is None
            ):
                raise ValueError("segmented attention metadata is incomplete")
            return self.forward_segmented(
                req.q,
                req.k,
                req.v,
                req.prefix_k,
                req.prefix_v,
                page_table=req.page_table,
                prefix_lens=req.prefix_lens,
                cu_seqlens_q=req.cu_seqlens_q,
                visible_current_end=req.visible_end,
                scale=req.scale,
                fully_visible_current=req.fully_visible,
                context=req.ctx,
            )
        if isinstance(req, VisibleEndAttention):
            return self.forward_visible_end(
                req.q,
                req.k,
                req.v,
                visible_end=req.visible_end,
                cu_seqlens_q=req.cu_seqlens_q,
                cu_seqlens_k=req.cu_seqlens_k,
                page_table=req.page_table,
                seqused_k=req.seqused_k,
                max_seqlen_q=req.max_seqlen_q,
                max_seqlen_k=req.max_seqlen_k,
                scale=req.scale,
                use_prefix_bounds=req.use_prefix_bounds,
                fully_visible=req.fully_visible,
                context=req.ctx,
            )
        if isinstance(req, VarlenAttention):
            return self.forward_varlen(
                req.q,
                req.k,
                req.v,
                cu_seqlens_q=req.cu_seqlens_q,
                cu_seqlens_k=req.cu_seqlens_k,
                max_seqlen_q=int(req.max_seqlen_q),
                max_seqlen_k=int(req.max_seqlen_k),
                causal=req.causal,
                scale=req.scale,
                block_table=req.block_table,
                context=req.ctx,
            )
        if isinstance(req, PagedDecodeAttention):
            return self.forward_paged(
                req.q,
                req.k,
                req.v,
                block_table=req.block_table,
                cache_seqlens=req.cache_seqlens,
                k=req.current_k,
                v=req.current_v,
                causal=req.causal,
                scale=req.scale,
                context=req.ctx,
            )
        if not isinstance(req, DenseAttention):
            raise TypeError(f"unsupported attention request {type(req).__name__}")
        return self.forward(
            req.q,
            req.k,
            req.v,
            causal=req.causal,
            scale=req.scale,
            attn_mask=req.attn_mask,
            context=req.ctx,
        )

    def _device_supported(self, tensor: torch.Tensor) -> bool:
        """Return whether this backend may execute tensors on the given device."""

        if self.cuda_only and tensor.device.type != "cuda":
            return False
        if self.min_compute_version is None and self.max_compute_version is None:
            return True
        if tensor.device.type != "cuda":
            return False
        compute_version = torch.cuda.get_device_capability(tensor.device)
        return bool(
            (self.min_compute_version is None or compute_version >= self.min_compute_version)
            and (self.max_compute_version is None or compute_version <= self.max_compute_version)
        )

    def _paged_storage_supported(self, req: object) -> bool:
        """Return whether the request cache uses storage this backend can consume directly."""

        from ...ops.requests import VisibleEndAttention

        kv_cache = getattr(req, "kv_cache", None)
        view_block_size = getattr(kv_cache, "block_size", None)
        block_table = getattr(req, "block_table", None)
        if view_block_size is not None:
            if not bool(getattr(kv_cache, "supports_paged_attention_storage", True)):
                return False
            block_size = int(view_block_size or 0)
        elif block_table is not None:
            paged_k = (
                req.prefix_k
                if isinstance(req, VisibleEndAttention) and req.prefix_k is not None
                else getattr(req, "k", None)
            )
            if not isinstance(paged_k, torch.Tensor) or paged_k.ndim != 4:
                return False
            block_size = int(paged_k.shape[1])
        else:
            return True
        return block_size > 0 and block_size % max(1, int(self.page_size_multiple)) == 0

    @staticmethod
    def _kv_dims(req: object) -> tuple[int, int]:
        """Extract KV head count and head width from a typed attention request."""

        from ...ops.requests import PagedDecodeAttention, VisibleEndAttention

        if (
            isinstance(req, VisibleEndAttention)
            and req.prefix_k is not None
            and req.prefix_v is not None
        ):
            return int(req.prefix_k.shape[-1]), int(req.prefix_v.shape[-1])
        if isinstance(req, PagedDecodeAttention):
            k = req.current_k if req.current_k is not None else req.k
            v = req.current_v if req.current_v is not None else req.v
            return int(k.shape[-1]), int(v.shape[-1])
        return int(getattr(req, "k").shape[-1]), int(getattr(req, "v").shape[-1])

    @staticmethod
    def _is_one_ar_decode(req: object) -> bool:
        """Return whether a request describes exactly one autoregressive decode row."""

        from ...ops.requests import PagedDecodeAttention

        if not isinstance(req, PagedDecodeAttention):
            return False
        if req.q.ndim == 3:
            plan = req.ctx
            query_lens = getattr(plan, "query_lens_cpu", ()) or ()
            if getattr(plan, "attention_mode", None) is AttentionMode.PAGED_DECODE and len(
                query_lens
            ) == int(req.q.shape[0]):
                return all(int(length) == 1 for length in query_lens)
            return int(req.q.shape[0]) == 1
        return req.q.ndim == 4 and int(req.q.shape[2]) == 1

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
        """Compute dense attention over rank-three or rank-four Q/K/V tensors."""

        raise NotImplementedError

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
        """Compute decode attention against scheduler-indexed paged KV storage."""

        raise NotImplementedError

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
        """Compute attention over packed sequences delimited by cumulative query and KV offsets."""

        raise NotImplementedError

    def forward_visible_end(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        visible_end: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        use_prefix_bounds: bool = False,
        fully_visible: bool = False,
        context: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Compute dense attention with an independent visible KV boundary for each query row."""

        raise NotImplementedError

    def forward_segmented(
        self,
        q: torch.Tensor,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        *,
        page_table: torch.Tensor,
        prefix_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        visible_current_end: torch.Tensor,
        scale: float,
        fully_visible_current: bool,
        context: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Compute attention over current and cached KV segments and merge their online-softmax states."""

        raise NotImplementedError
