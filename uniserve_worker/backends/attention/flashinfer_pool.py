"""FlashInfer wrapper pool, workspace buffers, and plan-tensor workspaces."""

from __future__ import annotations

import itertools
import weakref
from importlib import import_module
from typing import Any, NamedTuple

import torch

from .flashinfer_plan import (
    _decode_fast_plan_signature,
    _DecodePlanWorkspace,
    _fast_decode_plan_with_cpu_metadata,
    _PrefillPlanWorkspace,
)
from .paged_attention_plan_pool import PagedAttentionPlanPool
from .tuning import FlashInferTuningConfig

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
    """Allocate mutable uninitialized plan storage with the requested shape and dtype."""

    with torch.inference_mode(False):
        return torch.empty(size, dtype=dtype, device=device)


class WrapperKey(NamedTuple):
    """Value identity shared by a wrapper and all of its plan caches.

    ``kind`` is ``"decode"``, ``"decode_graph"``, or ``"prefill"``;
    ``use_tensor_cores`` is unset for prefill keys. ``batch_size`` and
    ``max_indices`` bound CUDA-graph buffers. ``scope`` isolates each captured
    prefill graph while the shared eager prefill wrapper uses no scope.
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
        """Indicate whether this key identifies a graph-exclusive decode wrapper."""

        return self.kind == "decode_graph"


class _DecodePlanOptions(NamedTuple):
    """Captures split-KV controls and the cache signature for one decode plan."""

    fixed_split_size: int | None
    disable_split_kv: bool
    signature: tuple[Any, ...]


class _WrapperPool(PagedAttentionPlanPool):
    """Owns the flashinfer wrapper, workspace, and plan-workspace caches.

    Wrapper construction is performed by the backend (which reads the optional
    flashinfer wrapper classes from its own module namespace so they remain
    monkeypatchable); the pool owns the cache dicts the backend populates and the
    device-keyed workspace / plan-workspace allocation.
    """

    def __init__(self, *, tuning: FlashInferTuningConfig) -> None:
        """Initialize reusable eager and graph wrapper catalogs with device workspaces."""

        super().__init__()
        self._tuning = tuning
        self._decode_wrappers: dict[WrapperKey, Any] = {}
        self._prefill_wrappers: dict[WrapperKey, Any] = {}
        self._decode_graph_buffers: dict[
            WrapperKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._decode_plan_workspaces: dict[WrapperKey, _DecodePlanWorkspace] = {}
        self._prefill_plan_workspaces: dict[WrapperKey, _PrefillPlanWorkspace] = {}
        self._binding_graph_wrappers: dict[
            int, tuple[WrapperKey, weakref.ReferenceType[Any] | None]
        ] = {}
        self._binding_prefill_graph_wrappers: dict[
            int, tuple[WrapperKey, weakref.ReferenceType[Any] | None]
        ] = {}
        self._decode_fast_plan_signatures: dict[WrapperKey, tuple[Any, ...]] = {}

    def _decode_wrapper(
        self,
        device: torch.device,
        num_q_heads: int,
        num_kv_heads: int,
        kv_dtype: torch.dtype,
    ) -> tuple[WrapperKey, Any]:
        """Create and cache a decode wrapper for one device and attention geometry."""

        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = self._tuning.decode_backend
        use_tensor_cores = _should_use_tensor_cores(
            override=self._tuning.use_tensor_core,
            kv_dtype=kv_dtype,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
        )
        key = WrapperKey("decode", device_key, backend, use_tensor_cores)
        wrapper = self._decode_wrappers.get(key)
        if wrapper is None:
            wrapper_cls = _fi._BatchDecodeWithPagedKVCacheWrapper
            if wrapper_cls is None:
                raise RuntimeError("flashinfer paged decode wrapper is not available")
            workspace = self._workspace(device)
            wrapper = wrapper_cls(
                workspace,
                "NHD",
                backend=backend,
                use_tensor_cores=use_tensor_cores,
            )
            self._decode_wrappers[key] = wrapper
        self.plan_decode(tuple(key), workspace=self._workspace(device), wrapper=wrapper)
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
        """Create a fixed-capacity decode wrapper and buffers for CUDA graph replay."""

        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = self._tuning.decode_backend
        use_tensor_cores = _should_use_tensor_cores(
            override=self._tuning.use_tensor_core,
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
            wrapper_cls = _fi._BatchDecodeWithPagedKVCacheWrapper
            if wrapper_cls is None:
                raise RuntimeError("flashinfer paged decode wrapper is not available")
            workspace = self._workspace(device)
            indptr = _empty_mutable((int(batch_size) + 1,), dtype=torch.int32, device=device)
            indices = _empty_mutable((max(1, int(max_indices)),), dtype=torch.int32, device=device)
            last_page_len = _empty_mutable((int(batch_size),), dtype=torch.int32, device=device)
            wrapper = wrapper_cls(
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
        self.plan_decode(tuple(key), workspace=self._workspace(device), wrapper=wrapper)
        self.bind_graph(tuple(key), wrapper)
        return key, wrapper

    def _prefill_wrapper(self, device: torch.device) -> tuple[WrapperKey, Any]:
        """Create and cache a variable-length paged prefill wrapper for one device."""

        from . import flashinfer as _fi

        device_key = _device_key(device)
        backend = self._tuning.prefill_backend
        key = WrapperKey("prefill", device_key, backend)
        wrapper = self._prefill_wrappers.get(key)
        if wrapper is None:
            wrapper_cls = _fi._BatchPrefillWithPagedKVCacheWrapper
            if wrapper_cls is None:
                raise RuntimeError("flashinfer paged prefill wrapper is not available")
            workspace = self._workspace(device)
            wrapper = wrapper_cls(
                workspace,
                "NHD",
                backend=backend,
            )
            self._prefill_wrappers[key] = wrapper
        self.plan_prefill(tuple(key), workspace=self._workspace(device), wrapper=wrapper)
        return key, wrapper

    def _prefill_graph_wrapper(
        self,
        device: torch.device,
        *,
        scope: int,
        batch_size: int,
        max_indices: int,
    ) -> tuple[WrapperKey, Any]:
        """Construct (or return) the exclusive prefill wrapper for ``scope``.

        The wrapper and its paged side tables have stable addresses for the
        lifetime of one captured graph. Exclusive wrappers share the device
        float workspace (transient kernel scratch, serialized on the compute
        stream) while owning their plan and integer workspaces.
        """
        from . import flashinfer as _fi

        batch_size = max(1, int(batch_size))
        max_indices = max(1, int(max_indices))
        device_key = _device_key(device)
        backend = self._tuning.prefill_backend
        key = WrapperKey(
            "prefill",
            device_key,
            backend,
            batch_size=batch_size,
            max_indices=max_indices,
            scope=int(scope),
        )
        wrapper = self._prefill_wrappers.get(key)
        if wrapper is None:
            wrapper_cls = _fi._BatchPrefillWithPagedKVCacheWrapper
            if wrapper_cls is None:
                raise RuntimeError("flashinfer paged prefill wrapper is not available")
            workspace = self._workspace(device)
            plan_workspace = self._prefill_plan_workspace(
                key,
                device,
                batch_size=batch_size,
                max_indices=max_indices,
            )
            wrapper = wrapper_cls(
                workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=plan_workspace.qo_indptr,
                paged_kv_indptr_buf=plan_workspace.kv_indptr,
                paged_kv_indices_buf=plan_workspace.indices,
                paged_kv_last_page_len_buf=plan_workspace.last_page_len,
                backend=backend,
            )
            self._prefill_wrappers[key] = wrapper
        self.plan_prefill(tuple(key), workspace=self._workspace(device), wrapper=wrapper)
        self.bind_graph(tuple(key), wrapper)
        return key, wrapper

    def _prefill_graph_wrapper_for_binding(self, binding: Any) -> tuple[WrapperKey, Any] | None:
        """Resolve the prefill graph wrapper registered for one binding identity."""

        if binding is None:
            return None
        entry = self._binding_prefill_graph_wrappers.get(id(binding))
        if entry is None:
            return None
        wrapper_key, binding_ref = entry
        if binding_ref is not None and binding_ref() is not binding:
            self._binding_prefill_graph_wrappers.pop(id(binding), None)
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
        """Select a planning path and bind one decode wrapper to page metadata."""

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
        self.plan_decode(
            (
                tuple(wrapper_key),
                options.signature,
                int(indptr.numel()),
                int(indices.numel()),
                int(last_page_len.numel()),
                None if sm_scale is None else float(sm_scale),
            ),
            workspace=self._workspace(indptr.device),
            wrapper=wrapper,
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
        """Record a wrapper specialization that accepts fast decode planning."""

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
        """Attempt fast decode planning when wrapper geometry and metadata support it."""

        if not allow_fast or not self._can_use_fast_decode_plan(
            wrapper_key, wrapper, options.signature
        ):
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
        """Derive split-KV and tensor-core planning options for one decode geometry."""

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
            fixed_split_size=self._tuning.decode_split_tile_size,
            disable_split_kv=self._tuning.disable_split_kv,
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
        """Run the optional fast planner and report whether it accepted the decode plan."""

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
        """Plan decode through the public wrapper interface with normalized metadata."""

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
        """Return whether a wrapper and specialization are eligible for fast planning."""

        from . import flashinfer as _fi

        if _fi._fast_decode_plan is None or not self._tuning.fast_decode_plan:
            return False
        if wrapper_key.backend not in {"fa2", "fa3"}:
            return False
        if self._decode_fast_plan_signatures.get(wrapper_key) != signature:
            return False
        return getattr(wrapper, "_cached_module", None) is not None

    def _decode_graph_wrapper_for_binding(self, binding: Any) -> tuple[WrapperKey, Any] | None:
        """Resolve the decode graph wrapper registered for one binding identity."""

        if binding is None:
            return None
        entry = self._binding_graph_wrappers.get(id(binding))
        if entry is None:
            return None
        wrapper_key, binding_ref = entry
        if binding_ref is not None and binding_ref() is not binding:
            self._binding_graph_wrappers.pop(id(binding), None)
            return None
        wrapper = self._decode_wrappers.get(wrapper_key)
        if wrapper is None:
            return None
        return wrapper_key, wrapper

    def _workspace(self, device: torch.device) -> torch.Tensor:
        """Return the persistent FlashInfer workspace for one device."""

        return self.workspace(device, max(1, int(self._tuning.workspace_size)), dtype=torch.uint8)

    def _decode_plan_workspace(
        self,
        wrapper_key: WrapperKey,
        device: torch.device,
        *,
        batch_size: int,
        max_indices: int,
        graph_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> _DecodePlanWorkspace:
        """Reserve bounded decode plan tensors for eager or graph-bound execution."""

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
        """Reserve bounded packed-query and paged-KV metadata for prefill planning."""

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
    """Return the canonical device string used by workspace and wrapper caches."""

    dev = torch.device(device)
    if dev.type == "cuda":
        index = dev.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{int(index)}"
    return str(dev)


def _should_use_tensor_cores(
    *,
    override: bool | None,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
) -> bool:
    """Resolve the tensor-core decode policy from override, dtype, and grouped-query ratio."""

    if override is not None:
        return override
    try:
        compiled_group_size = getattr(
            import_module("flashinfer.decode"),
            "_grouped_size_compiled_for_decode_kernels",
            None,
        )
        if callable(compiled_group_size):
            return not bool(compiled_group_size(num_q_heads, num_kv_heads))
    except ImportError:
        pass
    fp8_dtypes = tuple(
        getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
    )
    if kv_dtype in fp8_dtypes:
        return True
    if kv_dtype in {torch.float16, torch.half, torch.bfloat16}:
        return (int(num_q_heads) // max(1, int(num_kv_heads))) >= 4
    return False
