"""Model adapters for unified forward execution."""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ...contracts.batches import UniForwardBatch
from ...contracts.forward_batch import ForwardBatch
from ...contracts.forward_mode import ForwardMode
from ...contracts.model_protocols import ModelHooks
from ...foundation.errors import capability_mismatch, invalid_descriptor
from ...foundation.profiling import profile_range
from .descriptor import ForwardModelDescriptor, descriptor_from_model
from .result import ForwardResult

__all__ = [
    "ForwardAdapterContext",
    "ForwardModelAdapter",
    "WorkerForwardAdapter",
]

_TEXT_DRIVER_MODES = frozenset(
    {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT}
)


@runtime_checkable
class ForwardModelAdapter(Protocol):
    """Small neural execution seam consumed by ``ForwardExecutor``."""

    descriptor: ForwardModelDescriptor

    def forward(self, batch: ForwardBatch) -> ForwardResult: ...


@dataclass(frozen=True)
class ForwardAdapterContext:
    """Per-group execution context bound around one adapter forward."""

    dispatch_batch: UniForwardBatch
    group: tuple[tuple[int, Mapping[str, Any]], ...]
    defer_text_cpu_results: bool = False


class WorkerForwardAdapter:
    """Adapter over the worker's model and system execution services."""

    def __init__(
        self,
        *,
        model: Any,
        request_states: Any,
        text_driver: Any,
        denoise_driver: Any,
        encode_driver: Any,
        image_decode_driver: Any,
        descriptor: ForwardModelDescriptor | None = None,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
        mixed_proof_callback: Any | None = None,
    ) -> None:
        self.model = model
        self.request_states = request_states
        self.text_driver = text_driver
        self.denoise_driver = denoise_driver
        self.encode_driver = encode_driver
        self.image_decode_driver = image_decode_driver
        self.descriptor = descriptor or descriptor_from_model(model)
        self.defer_sampling = bool(defer_sampling)
        self.tensor_store = tensor_store
        self.mixed_proof_callback = mixed_proof_callback
        self._context: ForwardAdapterContext | None = None
        self._has_text_forward = hasattr(model, "forward")
        self._has_text_logits_batch = _overrides_model_hook(model, "run_text_logits_batch")
        self._is_text_capable = self._has_text_forward or self._has_text_logits_batch
        self._has_predict_velocity = _overrides_model_hook(model, "predict_velocity")
        self._has_decode_image = _overrides_model_hook(model, "decode_image")
        self._has_encode = _overrides_model_hook(model, "encode_image") or _overrides_model_hook(
            model, "encode_latents"
        )
        self._whole_batch_forward = bool(getattr(model, "whole_batch_forward", False))

    @contextmanager
    def bind(
        self,
        *,
        dispatch_batch: UniForwardBatch,
        group: Sequence[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> Iterator[None]:
        previous = self._context
        self._context = ForwardAdapterContext(
            dispatch_batch=dispatch_batch,
            group=tuple(group),
            defer_text_cpu_results=bool(defer_text_cpu_results),
        )
        try:
            yield
        finally:
            self._context = previous

    def forward(self, batch: ForwardBatch) -> ForwardResult:
        context = self._context
        if context is None:
            fb = UniForwardBatch.from_ops(batch.ops)
            group = tuple(enumerate(batch.ops))
            defer_text_cpu_results = False
        else:
            fb = context.dispatch_batch
            group = context.group
            defer_text_cpu_results = context.defer_text_cpu_results
        result = self.dispatch(
            fb,
            list(group),
            defer_text_cpu_results=defer_text_cpu_results,
        )
        if isinstance(result, ForwardResult):
            return result
        outputs = result
        if len(outputs) != len(batch.ops):
            raise invalid_descriptor(
                f"adapter returned {len(outputs)} outputs for {len(batch.ops)} forward ops"
            )
        return ForwardResult(runtime_outputs=tuple(outputs))

    def dispatch(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        *,
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any]:
        if fb.mode is ForwardMode.MIXED:
            result = self._run_mixed_mode(fb, group, defer_text_cpu_results)
        elif fb.mode in _TEXT_DRIVER_MODES:
            result = self._run_text_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.DENOISE:
            result = self._run_denoise_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.COMMIT:
            result = self._run_commit_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.ENCODE:
            result = self._run_encode_mode(fb, group, defer_text_cpu_results)
        else:
            result = None
        if result is None:
            result = self._run_whole_batch_forward(fb, group, defer_text_cpu_results)
        if result is None:
            raise capability_mismatch(
                f"model advertises ops for mode {fb.mode.value!r} but implements no "
                f"matching forward adapter path"
            )
        return result

    def can_run_forward(self, fb: UniForwardBatch) -> bool:
        """Whether this adapter can execute ``fb`` as one unified forward."""
        if fb.mode is not ForwardMode.MIXED:
            return True
        if self._whole_batch_forward or _can_run_private_forward_adapter(self.model, fb):
            return True
        return all(mode in _TEXT_DRIVER_MODES for mode in fb.op_modes)

    def _run_mixed_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        if self._whole_batch_forward:
            return self._run_model_forward(fb)
        if _can_run_private_forward_adapter(self.model, fb):
            if callable(self.mixed_proof_callback):
                self.mixed_proof_callback(fb, group)
            with profile_range("uniserve.forward_adapter.forward"):
                return _run_private_forward_adapter(
                    model=self.model,
                    batch=fb,
                    group=group,
                    request_states=self.request_states,
                    defer_text_cpu_results=defer_text_cpu_results,
                )
        if not all(mode in _TEXT_DRIVER_MODES for mode in fb.op_modes):
            return None
        if callable(self.mixed_proof_callback):
            self.mixed_proof_callback(fb, group)
        return self._run_text_driver(fb, defer_text_cpu_results)

    def _run_text_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del group
        if self._whole_batch_forward:
            return self._run_model_forward(fb)
        if self._is_text_capable:
            return self._run_text_driver(fb, defer_text_cpu_results)
        return None

    def _run_text_driver(
        self,
        fb: UniForwardBatch,
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any]:
        forward_logits = getattr(self.text_driver, "forward_logits", None)
        if callable(forward_logits):
            with profile_range("uniserve.forward_adapter.text_logits"):
                text_result = forward_logits(
                    fb,
                    self.request_states,
                    self.model,
                    defer_cpu_results=defer_text_cpu_results,
                    defer_sampling=self.defer_sampling,
            )
            if text_result is not None:
                req_ids = tuple(int(req_id) for req_id in getattr(text_result, "req_ids", ()))
                expected_req_ids = tuple(int(op["req_id"]) for op in fb.ops)
                if req_ids != expected_req_ids:
                    raise invalid_descriptor("text logits result req_ids must align with forward ops")
                return ForwardResult(
                    text_logits=text_result.logits,
                    text_cuda_ready_start_event=getattr(
                        text_result,
                        "cuda_ready_start_event",
                        None,
                    ),
                )
        with profile_range("uniserve.forward_adapter.text_driver"):
            return self.text_driver.step(
                fb,
                self.request_states,
                self.model,
                defer_cpu_results=defer_text_cpu_results,
                defer_sampling=self.defer_sampling,
                tensor_store=self.tensor_store,
            )

    def _run_denoise_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_predict_velocity:
            return None
        forward_result = getattr(self.denoise_driver, "forward_result", None)
        items = [
            (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op)
            for _, op in group
        ]
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.denoise_forward"):
                kwargs: dict[str, Any] = {"row_indices": tuple(range(len(group)))}
                if _accepts_keyword(forward_result, "graph_mode"):
                    kwargs["graph_mode"] = "eager"
                result = forward_result(
                    items,
                    self.model,
                    **kwargs,
                )
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.denoise_driver"):
            kwargs = {"graph_mode": "eager"} if _accepts_keyword(self.denoise_driver.step_many, "graph_mode") else {}
            return self.denoise_driver.step_many(items, self.model, **kwargs)

    def _run_commit_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_decode_image:
            return None
        items = [
            (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op)
            for _, op in group
        ]
        forward_result = getattr(self.image_decode_driver, "forward_result", None)
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.commit_forward"):
                result = forward_result(items, self.model, row_indices=tuple(range(len(group))))
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.image_decode"):
            return [
                self.image_decode_driver.step(
                    int(op["req_id"]),
                    self.request_states.get(int(op["req_id"])),
                    self.model,
                    op,
                )
                for _, op in group
            ]

    def _run_encode_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del group, defer_text_cpu_results
        if not self._has_encode:
            return None
        forward_result = getattr(self.encode_driver, "forward_result", None)
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.encode_forward"):
                result = forward_result(fb, self.model, row_indices=tuple(range(len(fb.ops))))
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.encode_driver"):
            return self.encode_driver.step(fb, self.model)

    def _run_whole_batch_forward(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del group, defer_text_cpu_results
        if not self._whole_batch_forward:
            return None
        return self._run_model_forward(fb)

    def _run_model_forward(self, fb: UniForwardBatch) -> list[Any]:
        with profile_range("uniserve.forward_adapter.model_forward"):
            result = self.model.forward(fb)
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
            raise invalid_descriptor("model forward must return one result per op")
        return list(result)


def _overrides_model_hook(model: Any, name: str) -> bool:
    hook = getattr(type(model), name, None)
    default = getattr(ModelHooks, name, None)
    return hook is not None and hook is not default


def _can_run_private_forward_adapter(model: Any, fb: UniForwardBatch) -> bool:
    hook = getattr(model, "_run_forward_adapter", None)
    if not callable(hook):
        return False
    modes = tuple(getattr(fb, "op_modes", ()))
    if not modes:
        return False
    mode_set = set(modes)
    has_text = bool(mode_set & {ForwardMode.EXTEND, ForwardMode.DECODE})
    if not has_text:
        return False
    if mode_set & {ForwardMode.DENOISE, ForwardMode.COMMIT}:
        return True
    return {ForwardMode.EXTEND, ForwardMode.DECODE} <= mode_set <= {
        ForwardMode.EXTEND,
        ForwardMode.DECODE,
    }


def _run_private_forward_adapter(
    *,
    model: Any,
    batch: UniForwardBatch,
    group: Sequence[tuple[int, Mapping[str, Any]]],
    request_states: Any,
    defer_text_cpu_results: bool,
) -> ForwardResult | list[Any]:
    hook = getattr(model, "_run_forward_adapter", None)
    if not callable(hook):
        raise capability_mismatch("model does not implement a private forward adapter")
    kwargs: dict[str, Any] = {"request_states": request_states, "group": list(group)}
    if _accepts_deferred_text_cpu_results(hook):
        kwargs["defer_text_cpu_results"] = bool(defer_text_cpu_results)
    result = hook(batch, **kwargs)
    if isinstance(result, ForwardResult):
        return result
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        raise invalid_descriptor("private forward adapter must return one result per mixed op")
    if len(result) != len(group):
        raise invalid_descriptor("private forward adapter returned the wrong number of results")
    return list(result)


def _accepts_deferred_text_cpu_results(hook: Any) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "defer_text_cpu_results" in signature.parameters


def _accepts_keyword(hook: Any, name: str) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters
