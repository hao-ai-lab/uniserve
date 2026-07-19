"""FlashInfer plan caching, plan-key building, and fast-plan quarantine."""
from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...foundation.sizing import ceil_div

if TYPE_CHECKING:
    from ...contracts.forward_context import ForwardStats
    from .flashinfer_pool import WrapperKey


@dataclass
class _DecodePlanWorkspace:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    page_counts: torch.Tensor


@dataclass
class _DecodePlanTensors:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    index_count: int


@dataclass
class _PrefillPlanWorkspace:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    page_counts: torch.Tensor


@dataclass
class _PrefillPlanTensors:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    index_count: int


@dataclass(frozen=True)
class _FastDecodePlanDefaults:
    q_data_type: torch.dtype | str
    kv_data_type: torch.dtype | str
    logits_soft_cap: float
    fixed_split_size: int


@dataclass(frozen=True)
class _FastDecodePlanImports:
    get_range_buf: Callable[..., torch.Tensor]
    get_seq_lens: Callable[..., torch.Tensor]


@dataclass(frozen=True)
class _FastDecodePlanHostTensors:
    qo_indptr: torch.Tensor
    indptr: torch.Tensor
    kv_lens: torch.Tensor


class _PlanCache:
    """Per-domain wrapper plan-key cache (decode or prefill).

    Owns the ``wrapper_key -> (plan_key, binding_ref)`` map, the currency
    check, the post-plan commit, and the plan/reuse stats recording. The domain
    differences (decode vs prefill stat counters) are confined to the injected
    ``record`` callback.
    """

    def __init__(
        self,
        record: Callable[..., None],
    ) -> None:
        self._plan_keys: dict[
            "WrapperKey",
            tuple[tuple[Any, ...], weakref.ReferenceType[Any] | None],
        ] = {}
        self._record = record

    def is_current(
        self,
        wrapper_key: "WrapperKey",
        plan_key: tuple[Any, ...],
        binding: Any,
    ) -> bool:
        cached = self._plan_keys.get(wrapper_key)
        if cached is None:
            return False
        cached_key, binding_ref = cached
        if cached_key != plan_key:
            return False
        return binding_ref is None or binding_ref() is binding

    def remember(
        self,
        wrapper_key: "WrapperKey",
        plan_key: tuple[Any, ...],
        binding: Any,
    ) -> None:
        self._plan_keys[wrapper_key] = (plan_key, _weakref_or_none(binding))

    def forget(self, wrapper_key: "WrapperKey") -> None:
        """Drop the cached plan for ``wrapper_key`` (graph-scoped wrapper release)."""

        self._plan_keys.pop(wrapper_key, None)

    def plan_or_reuse(
        self,
        *,
        wrapper_key: "WrapperKey",
        plan_key: tuple[Any, ...],
        binding: Any,
        stats: "ForwardStats | None",
        rows: int,
        build: Callable[[], int],
    ) -> None:
        """Reuse the cached plan, or run ``build`` to (re)plan and commit it.

        ``build`` constructs the plan tensors, issues the wrapper plan call, and
        returns the planned index count; it runs only when the cached plan is
        stale. Stats are recorded for both the planned and reused branches.
        """

        graph = wrapper_key.is_graph
        if self.is_current(wrapper_key, plan_key, binding):
            self._record(stats, planned=False, graph=graph, rows=rows, indices=0)
            return
        indices = build()
        self._record(stats, planned=True, graph=graph, rows=rows, indices=indices)
        self.remember(wrapper_key, plan_key, binding)


def _decode_plan_key(
    binding: Any,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    scale: float | None,
    current_tokens: int,
    wrapper_key: "WrapperKey",
) -> tuple[Any, ...]:
    return _decode_plan_key_from_shape(
        binding,
        block_table,
        cache_seqlens,
        batch_size=int(q.shape[0]),
        num_q_heads=int(q.shape[1]),
        num_kv_heads=int(k_cache.shape[2]),
        head_dim=int(q.shape[2]),
        page_size=int(k_cache.shape[1]),
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        scale=scale,
        current_tokens=current_tokens,
        wrapper_key=wrapper_key,
    )


def _decode_plan_key_from_shape(
    binding: Any,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float | None,
    current_tokens: int,
    wrapper_key: "WrapperKey",
) -> tuple[Any, ...]:
    return (
        wrapper_key,
        id(binding) if binding is not None else None,
        int(block_table.data_ptr()),
        int(cache_seqlens.data_ptr()),
        tuple(int(dim) for dim in block_table.shape),
        tuple(int(dim) for dim in cache_seqlens.shape),
        int(batch_size),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
        str(q_dtype),
        str(kv_dtype),
        None if scale is None else float(scale),
        int(current_tokens),
    )


def _prefill_plan_key(
    binding: Any,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    causal: bool,
    scale: float | None,
    wrapper_key: "WrapperKey",
) -> tuple[Any, ...]:
    return (
        wrapper_key,
        id(binding) if binding is not None else None,
        int(block_table.data_ptr()),
        int(cu_seqlens_q.data_ptr()),
        int(cu_seqlens_k.data_ptr()),
        tuple(int(dim) for dim in block_table.shape),
        tuple(int(dim) for dim in cu_seqlens_q.shape),
        tuple(int(dim) for dim in cu_seqlens_k.shape),
        tuple(int(dim) for dim in q.shape),
        tuple(int(dim) for dim in k_cache.shape),
        str(q.dtype),
        str(k_cache.dtype),
        bool(causal),
        None if scale is None else float(scale),
    )


def _decode_fast_plan_signature(
    *,
    wrapper_key: "WrapperKey",
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> tuple[Any, ...]:
    return (
        wrapper_key.backend,
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
        str(pos_encoding_mode),
        str(q_data_type),
        str(kv_data_type),
    )


def _fast_decode_plan_with_cpu_metadata(
    wrapper: Any,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    *,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: torch.dtype | str | None = None,
    kv_data_type: torch.dtype | str | None = None,
    data_type: torch.dtype | str | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int | None = None,
    disable_split_kv: bool = False,
    global_override_indptr_cpu: torch.Tensor | None = None,
    global_override_last_page_len_cpu: torch.Tensor | None = None,
) -> bool:
    imports = _fast_decode_plan_imports(wrapper, global_override_last_page_len_cpu)
    if imports is None or global_override_last_page_len_cpu is None:
        return False
    cached_module = getattr(wrapper, "_cached_module", None)
    if cached_module is None or not callable(getattr(cached_module, "plan", None)):
        return False

    defaults = _fast_decode_plan_defaults(
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        logits_soft_cap=logits_soft_cap,
        fixed_split_size=fixed_split_size,
    )

    _apply_fast_plan_overrides(
        wrapper,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        batch_size=int(last_page_len.shape[0]),
        cached_module=cached_module,
        get_range_buf=imports.get_range_buf,
        get_seq_lens=imports.get_seq_lens,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=defaults.logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=defaults.fixed_split_size,
        disable_split_kv=disable_split_kv,
        global_override_indptr_cpu=global_override_indptr_cpu,
        global_override_last_page_len_cpu=global_override_last_page_len_cpu,
    )
    return True


def _fast_decode_plan_imports(
    wrapper: Any,
    global_override_last_page_len_cpu: torch.Tensor | None,
) -> _FastDecodePlanImports | None:
    if not bool(getattr(wrapper, "use_tensor_cores", False)):
        return None
    if global_override_last_page_len_cpu is None:
        return None
    try:
        from flashinfer.decode import _get_range_buf, get_seq_lens
    except Exception:
        return None
    return _FastDecodePlanImports(_get_range_buf, get_seq_lens)


def _fast_decode_plan_defaults(
    *,
    q_data_type: torch.dtype | str | None,
    kv_data_type: torch.dtype | str | None,
    data_type: torch.dtype | str | None,
    logits_soft_cap: float | None,
    fixed_split_size: int | None,
) -> _FastDecodePlanDefaults:
    if data_type is not None:
        q_data_type = data_type if q_data_type is None else q_data_type
        kv_data_type = data_type if kv_data_type is None else kv_data_type
    elif q_data_type is None:
        q_data_type = "float16"
    if kv_data_type is None:
        kv_data_type = q_data_type
    return _FastDecodePlanDefaults(
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        logits_soft_cap=0.0 if logits_soft_cap is None else logits_soft_cap,
        fixed_split_size=-1 if fixed_split_size is None else fixed_split_size,
    )


# flashinfer-internals adapter (pinned to flashinfer 0.2.x). Quarantines the
# fast-plan path's reliance on private flashinfer wrapper attributes: it writes
# the wrapper's ``_paged_kv_*_buf`` / ``_qo_indptr_buf`` ring buffers, reads its
# ``_float_workspace_buffer`` / ``_int_workspace_buffer`` /
# ``_pin_memory_int_workspace_buffer`` / ``_backend`` fields, calls the cached
# module's ``plan`` to populate ``_plan_info``, and stamps the planning scalars
# (``_pos_encoding_mode`` / ``_window_left`` / ``_logits_soft_cap`` / ``_sm_scale``
# / ``_rope_scale`` / ``_rope_theta``). These names are not part of flashinfer's
# public API and can change across releases; ``_can_use_fast_decode_plan`` gates
# entry and ``_fast_decode_plan_with_cpu_metadata`` is the upstream-equivalent
# fallback when this path is unavailable. The values written here intentionally
# mirror flashinfer's own ``fast_decode_plan`` and must not be altered.
def _apply_fast_plan_overrides(
    wrapper: Any,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    *,
    batch_size: int,
    cached_module: Any,
    get_range_buf: Callable[..., torch.Tensor],
    get_seq_lens: Callable[..., torch.Tensor],
    pos_encoding_mode: str,
    window_left: int,
    logits_soft_cap: float | None,
    sm_scale: float | None,
    rope_scale: float | None,
    rope_theta: float | None,
    non_blocking: bool,
    fixed_split_size: int,
    disable_split_kv: bool,
    global_override_indptr_cpu: torch.Tensor | None,
    global_override_last_page_len_cpu: torch.Tensor,
) -> None:
    is_graph = bool(getattr(wrapper, "is_cuda_graph_enabled", False))
    _prepare_fast_decode_plan_buffers(
        wrapper,
        indptr,
        indices,
        last_page_len,
        batch_size=batch_size,
        is_graph=is_graph,
    )
    host_tensors = _fast_decode_plan_host_tensors(
        wrapper,
        indptr,
        global_override_indptr_cpu,
        global_override_last_page_len_cpu,
        get_range_buf=get_range_buf,
        get_seq_lens=get_seq_lens,
        batch_size=batch_size,
        page_size=page_size,
        non_blocking=non_blocking,
        is_graph=is_graph,
    )

    _invoke_fast_decode_plan(
        wrapper,
        cached_module,
        host_tensors.qo_indptr,
        host_tensors.indptr,
        host_tensors.kv_lens,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        window_left=window_left,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        is_graph=is_graph,
    )

    _stamp_fast_decode_plan_scalars(
        wrapper,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )


def _fast_decode_plan_host_tensors(
    wrapper: Any,
    indptr: torch.Tensor,
    global_override_indptr_cpu: torch.Tensor | None,
    global_override_last_page_len_cpu: torch.Tensor,
    *,
    get_range_buf: Callable[..., torch.Tensor],
    get_seq_lens: Callable[..., torch.Tensor],
    batch_size: int,
    page_size: int,
    non_blocking: bool,
    is_graph: bool,
) -> _FastDecodePlanHostTensors:
    qo_indptr_host = _prepare_fast_decode_qo_indptr(
        wrapper,
        indptr,
        get_range_buf,
        batch_size=batch_size,
        non_blocking=non_blocking,
        is_graph=is_graph,
    )
    indptr_host = _cpu_int32_tensor(
        global_override_indptr_cpu if global_override_indptr_cpu is not None else indptr.cpu()
    )
    last_page_len_host = _cpu_int32_tensor(global_override_last_page_len_cpu)
    return _FastDecodePlanHostTensors(
        qo_indptr=qo_indptr_host,
        indptr=indptr_host,
        kv_lens=get_seq_lens(indptr_host, last_page_len_host, int(page_size)),
    )


def _invoke_fast_decode_plan(
    wrapper: Any,
    cached_module: Any,
    qo_indptr_host: torch.Tensor,
    indptr_host: torch.Tensor,
    kv_lens_arr_host: torch.Tensor,
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    window_left: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    is_graph: bool,
) -> None:
    with _wrapper_device_context(wrapper):
        wrapper._plan_info = cached_module.plan(
            *_fast_decode_plan_args(
                wrapper,
                qo_indptr_host,
                indptr_host,
                kv_lens_arr_host,
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                window_left=window_left,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                is_graph=is_graph,
            )
        )


def _prepare_fast_decode_plan_buffers(
    wrapper: Any,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    *,
    batch_size: int,
    is_graph: bool,
) -> None:
    if is_graph:
        fixed_batch_size = int(getattr(wrapper, "_fixed_batch_size", batch_size))
        if batch_size != fixed_batch_size:
            raise ValueError(
                "The batch size should be fixed in cudagraph mode, the runtime "
                f"batch size {batch_size} mismatches the batch size set during "
                f"initialization {fixed_batch_size}"
            )
        indices_buffer = getattr(wrapper, "_paged_kv_indices_buf", None)
        if indices_buffer is not None and len(indices) > len(indices_buffer):
            raise ValueError("The size of indices should be less than or equal to the allocated buffer")
        return
    wrapper._paged_kv_indptr_buf = indptr
    wrapper._paged_kv_indices_buf = indices
    wrapper._paged_kv_last_page_len_buf = last_page_len


def _prepare_fast_decode_qo_indptr(
    wrapper: Any,
    indptr: torch.Tensor,
    get_range_buf: Callable[..., torch.Tensor],
    *,
    batch_size: int,
    non_blocking: bool,
    is_graph: bool,
) -> torch.Tensor:
    qo_indptr_host = get_range_buf(batch_size + 1, "cpu")
    if not is_graph:
        wrapper._qo_indptr_buf = qo_indptr_host.to(
            getattr(wrapper, "device", indptr.device),
            non_blocking=non_blocking,
        )
    return qo_indptr_host


def _cpu_int32_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu" and tensor.dtype == torch.int32:
        return tensor
    return tensor.to(device="cpu", dtype=torch.int32)


def _fast_decode_plan_args(
    wrapper: Any,
    qo_indptr_host: torch.Tensor,
    indptr_host: torch.Tensor,
    kv_lens_arr_host: torch.Tensor,
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    window_left: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    is_graph: bool,
) -> list[Any]:
    args = [
        wrapper._float_workspace_buffer,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        indptr_host,
        kv_lens_arr_host,
        batch_size,
        batch_size,
        int(num_qo_heads),
        int(num_kv_heads),
        int(page_size),
        is_graph,
        int(head_dim),
        int(head_dim),
        False,
        int(window_left),
    ]
    if getattr(wrapper, "_backend", None) == "fa2":
        args.extend((fixed_split_size, bool(disable_split_kv), 0))
    return args


def _stamp_fast_decode_plan_scalars(
    wrapper: Any,
    *,
    pos_encoding_mode: str,
    window_left: int,
    logits_soft_cap: float | None,
    sm_scale: float | None,
    rope_scale: float | None,
    rope_theta: float | None,
) -> None:
    wrapper._pos_encoding_mode = pos_encoding_mode
    wrapper._window_left = window_left
    wrapper._logits_soft_cap = logits_soft_cap
    wrapper._sm_scale = sm_scale
    wrapper._rope_scale = rope_scale
    wrapper._rope_theta = rope_theta


def _wrapper_device_context(wrapper: Any):
    raw_device = getattr(wrapper, "device", None)
    if raw_device is None:
        return nullcontext()
    try:
        device = (
            torch.device("cuda", int(raw_device))
            if isinstance(raw_device, int)
            else torch.device(raw_device)
        )
    except (TypeError, ValueError):
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.device(device)


def _cpu_paged_indptr(plan: Any, batch_size: int, page_size: int) -> torch.Tensor | None:
    if plan is None:
        return None
    kv_seqlens_cpu = tuple(int(x) for x in getattr(plan, "kv_seqlens_cpu", ()) or ())
    if len(kv_seqlens_cpu) != int(batch_size):
        return None
    page_size = max(1, int(page_size))
    indptr = torch.empty((int(batch_size) + 1,), dtype=torch.int32, device="cpu")
    indptr[0] = 0
    running = 0
    for row, seq_len in enumerate(kv_seqlens_cpu):
        running += ceil_div(seq_len, page_size)
        indptr[row + 1] = running
    return indptr


def _cpu_last_page_len(plan: Any, batch_size: int, page_size: int) -> torch.Tensor | None:
    if plan is None:
        return None
    kv_seqlens_cpu = tuple(int(x) for x in getattr(plan, "kv_seqlens_cpu", ()) or ())
    if len(kv_seqlens_cpu) != int(batch_size):
        return None
    page_size = max(1, int(page_size))
    values = [((max(1, int(seq_len)) - 1) % page_size) + 1 for seq_len in kv_seqlens_cpu]
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _indptr_last(indptr: torch.Tensor | None) -> int | None:
    if indptr is None or int(indptr.numel()) <= 0:
        return None
    return int(indptr[-1].item())


def _weakref_or_none(obj: Any) -> weakref.ReferenceType[Any] | None:
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return None


def _record_decode_plan_stats(
    stats: "ForwardStats | None",
    *,
    planned: bool,
    graph: bool,
    rows: int,
    indices: int,
) -> None:
    if stats is None:
        return
    if planned:
        stats.flashinfer_decode_plan_calls += 1
        stats.flashinfer_decode_plan_rows += max(0, int(rows))
        stats.flashinfer_decode_plan_indices += max(0, int(indices))
        if graph:
            stats.flashinfer_decode_graph_plan_calls += 1
    else:
        stats.flashinfer_decode_plan_reuses += 1
        if graph:
            stats.flashinfer_decode_graph_plan_reuses += 1


def _record_prefill_plan_stats(
    stats: "ForwardStats | None",
    *,
    planned: bool,
    rows: int,
    indices: int,
) -> None:
    if stats is None:
        return
    if planned:
        stats.flashinfer_prefill_plan_calls += 1
        stats.flashinfer_prefill_plan_rows += max(0, int(rows))
        stats.flashinfer_prefill_plan_indices += max(0, int(indices))
    else:
        stats.flashinfer_prefill_plan_reuses += 1
