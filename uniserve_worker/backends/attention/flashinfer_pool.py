"""FlashInfer wrapper pool, workspace buffers, and plan-tensor workspaces."""
from __future__ import annotations

import itertools
import weakref
from typing import Any, NamedTuple

import torch

from ...foundation.runtime_config import get_worker_config
from .flashinfer_plan import (
    _decode_fast_plan_signature,
    _DecodePlanWorkspace,
    _fast_decode_plan_with_cpu_metadata,
    _PrefillPlanWorkspace,
)

_DEFAULT_WORKSPACE_SIZE = 512 * 1024 * 1024

# Nonce source for graph-scoped exclusive prefill wrappers. Each CUDA-graph
# capture that bakes a prefill ``wrapper.run`` binds its own wrapper keyed by a
# fresh scope so no other plan call can ever touch the captured wrapper's state.
_PREFILL_GRAPH_SCOPES = itertools.count(1)


def _empty_mutable(
    size: int | tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # FlashInfer plan/workspace buffers are rewritten across forward passes.
    # Keep them out of inference tensor mode even when first allocated during
    # model inference.
    with torch.inference_mode(False):
        return torch.empty(size, dtype=dtype, device=device)


class WrapperKey(NamedTuple):
    """Identity of a cached flashinfer wrapper and its plan caches.

    This is the dict key for every per-wrapper cache (wrappers, graph buffers,
    plan workspaces, plan keys, fast-plan signatures) and is embedded inside the
    plan keys themselves. It stays a ``NamedTuple`` so it remains hashable and
    compares by value like the positional tuple it replaced, while the
    discriminating fields (``kind``, ``backend``) are read by name instead of by
    index. ``kind`` is one of ``"decode"``, ``"decode_graph"`` or ``"prefill"``;
    ``use_tensor_cores`` is ``None`` for prefill keys and ``batch_size`` /
    ``max_indices`` are set only for cuda-graph decode keys. ``scope`` is a
    nonce isolating a graph-scoped *exclusive* prefill wrapper (one per captured
    graph); the shared prefill wrapper keeps ``scope=None``.
    """

    kind: str
    device_key: str
    backend: str
    use_tensor_cores: bool | None = None
    batch_size: int | None = None
    max_indices: int | None = None
    scope: int | None = None

    @property
    def is_graph(self) -> bool:
        return self.kind == "decode_graph"


class _DecodePlanOptions(NamedTuple):
    fixed_split_size: int | None
    disable_split_kv: bool
    signature: tuple[Any, ...]


class _WrapperPool:
    """Owns the flashinfer wrapper, workspace, and plan-workspace caches.

    Wrapper construction is performed by the backend (which reads the optional
    flashinfer wrapper classes from its own module namespace so they remain
    monkeypatchable); the pool owns the cache dicts the backend populates and the
    device-keyed workspace / plan-workspace allocation.
    """

    def __init__(self) -> None:
        self._workspace_buffers: dict[tuple[str, int], torch.Tensor] = {}
        self._decode_wrappers: dict[WrapperKey, Any] = {}
        self._prefill_wrappers: dict[WrapperKey, Any] = {}
        self._decode_graph_buffers: dict[WrapperKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_plan_workspaces: dict[WrapperKey, _DecodePlanWorkspace] = {}
        self._prefill_plan_workspaces: dict[WrapperKey, _PrefillPlanWorkspace] = {}
        self._metadata_graph_wrappers: dict[int, tuple[WrapperKey, weakref.ReferenceType[Any] | None]] = {}
        self._metadata_prefill_graph_wrappers: dict[int, tuple[WrapperKey, weakref.ReferenceType[Any] | None]] = {}
        self._decode_fast_plan_signatures: dict[WrapperKey, tuple[Any, ...]] = {}

    def _decode_wrapper(
        self,
        device: torch.device,
        num_q_heads: int,
        num_kv_heads: int,
        kv_dtype: torch.dtype,
    ) -> tuple[WrapperKey, Any]:
        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = get_worker_config().flashinfer.decode_backend
        use_tensor_cores = _should_use_tensor_cores(
            kv_dtype=kv_dtype,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
        )
        key = WrapperKey("decode", device_key, backend, use_tensor_cores)
        wrapper = self._decode_wrappers.get(key)
        if wrapper is None:
            workspace = self._workspace(device)
            wrapper = _fi._BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=backend,
                use_tensor_cores=use_tensor_cores,
            )
            self._decode_wrappers[key] = wrapper
        return key, wrapper

    def _decode_cuda_graph_wrapper(
        self,
        device: torch.device,
        *,
        batch_size: int,
        max_indices: int,
        num_q_heads: int,
        num_kv_heads: int,
        kv_dtype: torch.dtype,
    ) -> tuple[WrapperKey, Any]:
        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = get_worker_config().flashinfer.decode_backend
        use_tensor_cores = _should_use_tensor_cores(
            kv_dtype=kv_dtype,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
        )
        key = WrapperKey(
            "decode_graph",
            device_key,
            backend,
            use_tensor_cores,
            int(batch_size),
            int(max_indices),
        )
        wrapper = self._decode_wrappers.get(key)
        if wrapper is None:
            workspace = self._workspace(device)
            indptr = _empty_mutable((int(batch_size) + 1,), dtype=torch.int32, device=device)
            indices = _empty_mutable((max(1, int(max_indices)),), dtype=torch.int32, device=device)
            last_page_len = _empty_mutable((int(batch_size),), dtype=torch.int32, device=device)
            wrapper = _fi._BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=backend,
                use_cuda_graph=True,
                use_tensor_cores=use_tensor_cores,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last_page_len,
            )
            self._decode_wrappers[key] = wrapper
            self._decode_graph_buffers[key] = (indptr, indices, last_page_len)
        return key, wrapper

    def _prefill_wrapper(self, device: torch.device) -> tuple[WrapperKey, Any]:
        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = get_worker_config().flashinfer.prefill_backend
        key = WrapperKey("prefill", device_key, backend)
        wrapper = self._prefill_wrappers.get(key)
        if wrapper is None:
            workspace = self._workspace(device)
            wrapper = _fi._BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=backend,
            )
            self._prefill_wrappers[key] = wrapper
        return key, wrapper

    def _prefill_graph_wrapper(self, device: torch.device, *, scope: int) -> tuple[WrapperKey, Any]:
        """Construct (or return) the exclusive prefill wrapper for ``scope``.

        Exclusive wrappers share the device float workspace (transient kernel
        scratch, serialized on the compute stream) but own their int workspace,
        so a plan against the shared prefill wrapper can never mutate the plan
        state a captured graph baked from this one.
        """
        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = get_worker_config().flashinfer.prefill_backend
        key = WrapperKey("prefill", device_key, backend, scope=int(scope))
        wrapper = self._prefill_wrappers.get(key)
        if wrapper is None:
            workspace = self._workspace(device)
            wrapper = _fi._BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=backend,
            )
            self._prefill_wrappers[key] = wrapper
        return key, wrapper

    def _prefill_graph_wrapper_for_metadata(self, metadata: Any) -> tuple[WrapperKey, Any] | None:
        if metadata is None:
            return None
        entry = self._metadata_prefill_graph_wrappers.get(id(metadata))
        if entry is None:
            return None
        wrapper_key, metadata_ref = entry
        if metadata_ref is not None and metadata_ref() is not metadata:
            self._metadata_prefill_graph_wrappers.pop(id(metadata), None)
            return None
        wrapper = self._prefill_wrappers.get(wrapper_key)
        if wrapper is None:
            return None
        return wrapper_key, wrapper

    def _plan_decode(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str = "NONE",
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        data_type: torch.dtype,
        sm_scale: float | None,
        block_tables: torch.Tensor | None,
        seq_lens: torch.Tensor | None,
        global_override_indptr_cpu: torch.Tensor | None = None,
        global_override_last_page_len_cpu: torch.Tensor | None = None,
        allow_fast: bool = True,
    ) -> None:
        options = self._decode_plan_options(
            wrapper_key=wrapper_key,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
        )
        if self._maybe_plan_decode_fast(
            wrapper_key,
            wrapper,
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            options=options,
            global_override_indptr_cpu=global_override_indptr_cpu,
            global_override_last_page_len_cpu=global_override_last_page_len_cpu,
            allow_fast=allow_fast,
        ):
            return

        self._fallback_decode_plan(
            wrapper,
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            block_tables=block_tables,
            seq_lens=seq_lens,
            options=options,
        )
        self._remember_decode_fast_signature(wrapper_key, options.signature)

    def _remember_decode_fast_signature(
        self,
        wrapper_key: WrapperKey,
        signature: tuple[Any, ...],
    ) -> None:
        from . import flashinfer as _fi

        if _fi._fast_decode_plan is not None:
            self._decode_fast_plan_signatures[wrapper_key] = signature

    def _maybe_plan_decode_fast(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        data_type: torch.dtype,
        sm_scale: float | None,
        options: _DecodePlanOptions,
        global_override_indptr_cpu: torch.Tensor | None,
        global_override_last_page_len_cpu: torch.Tensor | None,
        allow_fast: bool,
    ) -> bool:
        if not allow_fast or not self._can_use_fast_decode_plan(wrapper_key, wrapper, options.signature):
            return False
        return self._try_fast_decode_plan(
            wrapper,
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            options=options,
            global_override_indptr_cpu=global_override_indptr_cpu,
            global_override_last_page_len_cpu=global_override_last_page_len_cpu,
        )

    def _decode_plan_options(
        self,
        *,
        wrapper_key: WrapperKey,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
    ) -> _DecodePlanOptions:
        signature = _decode_fast_plan_signature(
            wrapper_key=wrapper_key,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
        )
        return _DecodePlanOptions(
            fixed_split_size=get_worker_config().flashinfer.decode_split_tile_size,
            disable_split_kv=get_worker_config().flashinfer.disable_split_kv,
            signature=signature,
        )

    def _try_fast_decode_plan(
        self,
        wrapper: Any,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        data_type: torch.dtype,
        sm_scale: float | None,
        options: _DecodePlanOptions,
        global_override_indptr_cpu: torch.Tensor | None,
        global_override_last_page_len_cpu: torch.Tensor | None,
    ) -> bool:
        from . import flashinfer as _fi

        assert _fi._fast_decode_plan is not None
        if _fast_decode_plan_with_cpu_metadata(
            wrapper,
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            non_blocking=True,
            fixed_split_size=options.fixed_split_size,
            disable_split_kv=options.disable_split_kv,
            global_override_indptr_cpu=global_override_indptr_cpu,
            global_override_last_page_len_cpu=global_override_last_page_len_cpu,
        ):
            return True
        _fi._fast_decode_plan(
            wrapper,
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            non_blocking=True,
            fixed_split_size=options.fixed_split_size,
            disable_split_kv=options.disable_split_kv,
            global_override_indptr_cpu=global_override_indptr_cpu,
        )
        return True

    def _fallback_decode_plan(
        self,
        wrapper: Any,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        data_type: torch.dtype,
        sm_scale: float | None,
        block_tables: torch.Tensor | None,
        seq_lens: torch.Tensor | None,
        options: _DecodePlanOptions,
    ) -> None:
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            non_blocking=True,
            block_tables=block_tables,
            seq_lens=seq_lens,
            fixed_split_size=options.fixed_split_size,
            disable_split_kv=options.disable_split_kv,
        )

    def _can_use_fast_decode_plan(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        signature: tuple[Any, ...],
    ) -> bool:
        from . import flashinfer as _fi

        if _fi._fast_decode_plan is None or not _fast_decode_plan_enabled():
            return False
        if wrapper_key.backend not in {"fa2", "fa3"}:
            return False
        if self._decode_fast_plan_signatures.get(wrapper_key) != signature:
            return False
        return getattr(wrapper, "_cached_module", None) is not None

    def _decode_graph_wrapper_for_metadata(self, metadata: Any) -> tuple[WrapperKey, Any] | None:
        if metadata is None:
            return None
        entry = self._metadata_graph_wrappers.get(id(metadata))
        if entry is None:
            return None
        wrapper_key, metadata_ref = entry
        if metadata_ref is not None and metadata_ref() is not metadata:
            self._metadata_graph_wrappers.pop(id(metadata), None)
            return None
        wrapper = self._decode_wrappers.get(wrapper_key)
        if wrapper is None:
            return None
        return wrapper_key, wrapper

    def _workspace(self, device: torch.device) -> torch.Tensor:
        size = _workspace_size()
        key = (_device_key(device), size)
        workspace = self._workspace_buffers.get(key)
        if workspace is None:
            workspace = _empty_mutable(size, dtype=torch.uint8, device=device)
            self._workspace_buffers[key] = workspace
        return workspace

    def _decode_plan_workspace(
        self,
        wrapper_key: WrapperKey,
        device: torch.device,
        *,
        batch_size: int,
        max_indices: int,
        graph_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> _DecodePlanWorkspace:
        batch_size = max(1, int(batch_size))
        max_indices = max(1, int(max_indices))
        if graph_buffers is not None:
            indptr, indices, last_page_len = graph_buffers
            workspace = self._decode_plan_workspaces.get(wrapper_key)
            if (
                workspace is None
                or workspace.indptr is not indptr
                or workspace.indices is not indices
                or workspace.last_page_len is not last_page_len
                or int(workspace.page_counts.numel()) < batch_size
            ):
                workspace = _DecodePlanWorkspace(
                    indptr=indptr,
                    indices=indices,
                    last_page_len=last_page_len,
                    page_counts=_empty_mutable(batch_size, dtype=torch.int32, device=device),
                )
                self._decode_plan_workspaces[wrapper_key] = workspace
            return workspace

        workspace = self._decode_plan_workspaces.get(wrapper_key)
        if (
            workspace is None
            or workspace.indptr.device != device
            or int(workspace.indptr.numel()) < batch_size + 1
            or int(workspace.indices.numel()) < max_indices
            or int(workspace.last_page_len.numel()) < batch_size
            or int(workspace.page_counts.numel()) < batch_size
        ):
            workspace = _DecodePlanWorkspace(
                indptr=_empty_mutable(batch_size + 1, dtype=torch.int32, device=device),
                indices=_empty_mutable(max_indices, dtype=torch.int32, device=device),
                last_page_len=_empty_mutable(batch_size, dtype=torch.int32, device=device),
                page_counts=_empty_mutable(batch_size, dtype=torch.int32, device=device),
            )
            self._decode_plan_workspaces[wrapper_key] = workspace
        return workspace

    def _prefill_plan_workspace(
        self,
        wrapper_key: WrapperKey,
        device: torch.device,
        *,
        batch_size: int,
        max_indices: int,
    ) -> _PrefillPlanWorkspace:
        batch_size = max(1, int(batch_size))
        max_indices = max(1, int(max_indices))
        workspace = self._prefill_plan_workspaces.get(wrapper_key)
        if (
            workspace is None
            or workspace.qo_indptr.device != device
            or int(workspace.qo_indptr.numel()) < batch_size + 1
            or int(workspace.kv_indptr.numel()) < batch_size + 1
            or int(workspace.indices.numel()) < max_indices
            or int(workspace.last_page_len.numel()) < batch_size
            or int(workspace.page_counts.numel()) < batch_size
        ):
            workspace = _PrefillPlanWorkspace(
                qo_indptr=_empty_mutable(batch_size + 1, dtype=torch.int32, device=device),
                kv_indptr=_empty_mutable(batch_size + 1, dtype=torch.int32, device=device),
                indices=_empty_mutable(max_indices, dtype=torch.int32, device=device),
                last_page_len=_empty_mutable(batch_size, dtype=torch.int32, device=device),
                page_counts=_empty_mutable(batch_size, dtype=torch.int32, device=device),
            )
            self._prefill_plan_workspaces[wrapper_key] = workspace
        return workspace


def _device_key(device: torch.device | str) -> str:
    dev = torch.device(device)
    if dev.type == "cuda":
        index = dev.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{int(index)}"
    return str(dev)


def _workspace_size() -> int:
    size = get_worker_config().flashinfer.workspace_size
    return max(1, size)


def _fast_decode_plan_enabled() -> bool:
    return get_worker_config().flashinfer.fast_decode_plan


def _should_use_tensor_cores(
    *,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
) -> bool:
    override = get_worker_config().flashinfer.use_tensor_core
    if override is not None:
        return override
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels  # type: ignore

        return not bool(_grouped_size_compiled_for_decode_kernels(num_q_heads, num_kv_heads))
    except (ImportError, AttributeError):
        pass
    fp8_dtypes = tuple(getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name))
    if kv_dtype in fp8_dtypes:
        return True
    if kv_dtype in {torch.float16, torch.half, torch.bfloat16}:
        return (int(num_q_heads) // max(1, int(num_kv_heads))) >= 4
    return False
