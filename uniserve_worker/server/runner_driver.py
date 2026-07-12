"""WorkerRuntime adapter for runner-backed UniModel classes."""
from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from ..backends.attention import (
    get_attention_backend,
    has_attention_backend,
    normalize_attention_backend_name,
)
from ..contracts.caps import Caps, validate_caps
from ..contracts.model_family import ModelFamilyDescriptor
from ..contracts.model_protocols import UniModel, verify_model_conformance
from ..execution.runner import ModelRunner, RunnerConfig
from ..foundation.env import DEFAULT_ATTENTION_BACKEND
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..loader import ModelBringUp, get_loader_for_descriptor
from ..loader.paths import read_config, resolve_model_path
from ..models.registry import detect_model_architectures, resolve_model_descriptor
from ..nn.mesh import get_current_mesh
from ..nn.quant.base import process_quantized_modules
from ..processors import get_processor_for_descriptor
from ..runtime.resources import ResourceRuntime
from .base_driver import BaseWorkerDriver
from .worker_kind import GEN, UND, tower_role_for_kind

__all__ = [
    'call_with_supported_kwargs',
    'RunnerDriver',
    'load_runner_engine',
]

logger = logging.getLogger(__name__)


def call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts.

    Optional model hooks (``caps``/``configure_tokenizer``) accept evolving,
    overlapping keyword sets; this drops any keyword the concrete hook does not
    declare so a hook may opt into just the arguments it needs.
    """
    sig = inspect.signature(fn)
    supported = {key: value for key, value in kwargs.items() if key in sig.parameters}
    return fn(**supported)


class RunnerDriver(BaseWorkerDriver):
    """Thin IPC-facing adapter around the shared ``ModelRunner``.

    The runtime shell still expects a caps/execute/drop_request object. This
    adapter is that object for runner-backed models; it contains no
    model-specific scheduling or tensor code.
    """

    def __init__(
        self,
        model: UniModel,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        defer_sampling: bool = False,
        transfer_backend: str = "local",
        worker_kind: str | None = None,
        family_descriptor: ModelFamilyDescriptor | None = None,
        simulation: bool = False,
    ) -> None:
        super().__init__(block_size=block_size)
        self.model = model
        self.attention_backend = attention_backend or DEFAULT_ATTENTION_BACKEND
        self.defer_sampling = bool(defer_sampling)
        self.transfer_backend = transfer_backend
        self.worker_kind = worker_kind
        self.family_descriptor = family_descriptor or ModelFamilyDescriptor.from_model_class(type(model))
        self.block_size = int(self.block_size)
        self.block_size = self._adjust_block_size(self.block_size)
        self.kv_token_capacity = kv_token_capacity
        self._caps = self._caps_with_current_rank(
            validate_caps(self._build_caps(), owner=f"{type(model).__name__}.RunnerDriver")
        )
        self._validate_declared_controls()
        self.model.configure_runtime(
            block_size=self.block_size,
            kv_token_capacity=self.kv_token_capacity,
            caps=self._caps.to_wire(),
        )
        # Deferred sampling: a data-plane TensorStore publishes logits to a separate
        # Sampler worker. Built only when defer mode is on (else inline sampling).
        tensor_store = None
        if self.defer_sampling:
            from ..runtime.tensor_store import TensorStore
            from ..runtime.transfer import make_transport

            tensor_store = TensorStore(transport=make_transport(self.transfer_backend))
        ledger = self._make_resource_runtime()
        residency = self._build_residency(ledger)
        self.runner = ModelRunner(
            model,
            config=RunnerConfig(simulation=bool(simulation)),
            attention_backend=self.attention_backend,
            resource_runtime=ledger,
            multimodal_processor=get_processor_for_descriptor(self.family_descriptor),
            defer_sampling=self.defer_sampling,
            tensor_store=tensor_store,
            residency=residency,
        )
        self._maybe_bind_data_plane()

    def _maybe_bind_data_plane(self) -> None:
        """Tower disaggregation (Mode A): bind the model's cross-process und<->gen
        handoff to this worker's data-plane transport.

        The und/gen pools cross the conditioning KV over the registered transport
        (``cuda_ipc`` same-node / ``mooncake`` cross-node). A no-op for whole-model
        kinds, a model without the hook, or a local/shm transport (no real edge)."""
        if self.worker_kind not in (UND, GEN):
            return
        if self.transfer_backend in (None, "local", "shm"):
            return
        from ..runtime.transfer import make_transport

        self.model.bind_data_plane_handoff(make_transport(self.transfer_backend))
        logger.info(
            "tower disaggregation: bound %s data plane on the %s worker",
            self.transfer_backend, self.worker_kind,
        )

    def _build_residency(self, ledger: ResourceRuntime):
        """Build the system-owned residency (KV pool) from the model geometry.

        A thin model declares ``kv_cache_spec``; the worker runtime — not the
        model — allocates and owns the resulting paged KV pool. Models without
        ``kv_cache_spec`` keep their own pool and this returns ``None``.
        """

        from ..runtime.residency import ResidencyManager

        # A self-managing multimodal model (SenseNova/Bagel) builds its own
        # generation pools (gen_device split, latent-aware scratch) into a
        # ResidencyManager; the runtime adopts it as the system owner + attaches
        # the lease ledger. A thin text model instead declares only its KV
        # geometry and the system constructs the pool.
        existing = getattr(self.model, "residency", None)
        if isinstance(existing, ResidencyManager):
            existing.ledger = ledger
            return existing
        spec = self.model.kv_cache_spec()
        if spec is None:
            return None
        num_blocks = int(self._caps.num_blocks or 0)
        block_size = int(self._caps.block_size or self.block_size)
        device = str(getattr(self.model, "device", "cuda") or "cuda")
        return ResidencyManager.build(
            spec,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            ledger=ledger,
        )

    def _build_caps(self) -> Caps:
        caps = call_with_supported_kwargs(
            self.model.caps,
            block_size=self.block_size,
            kv_token_capacity=self.kv_token_capacity,
        )
        if isinstance(caps, Caps):
            return caps
        raise capability_mismatch(f"{type(self.model).__name__}.caps() must return Caps")

    @staticmethod
    def _caps_with_current_rank(caps: Caps) -> Caps:
        mesh = get_current_mesh()
        return replace(caps, tp_rank=int(mesh.tp_rank), tp_size=int(mesh.tp_size))

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        return self.runner.execute(
            dict(batch),
            defer_text_cpu_results=defer_text_cpu_results,
        )

    def drop_request(self, rid: int) -> None:
        self.runner.drop_request(int(rid))

    def copy_blocks(self, copies: Any) -> None:
        self.model.copy_blocks(copies)

    def load_lora(self, lora_id: int, lora_path: str) -> None:
        self.model.load_lora(lora_id, lora_path)

    def unload_lora(self, lora_id: int) -> None:
        self.model.unload_lora(lora_id)

    def free_encoder(self, handles: Any) -> None:
        self.model.free_encoder(handles)

    def reset_prefix_cache(self) -> None:
        self.model.reset_prefix_cache()

    def resource_pressure(self) -> list[dict[str, Any]]:
        return self.runner.resource_runtime.pressure()

    def _make_resource_runtime(self) -> ResourceRuntime:
        # Per-class units: kv_block=blocks, scratch=CFG branch slots,
        # image_latent=latent tokens, encoder_output=handles, adapter=slots.
        # Scratch totals use scratch_capacity_tokens (token magnitude) while
        # used() counts branch slots; physical scratch exhaustion is enforced
        # by the model pool, and this ledger cross-checks lease balance only.
        caps = self._caps
        totals = {
            "kv_block": int(caps.num_blocks or 0),
            "scratch": int(caps.scratch_capacity_tokens or 0),
            "image_latent": int(caps.max_latent_size or 0),
            "encoder_output": int(caps.encoder_cache_budget or 0),
            "adapter": 0,
        }
        return ResourceRuntime(caps.resource_classes or ("kv_block",), totals=totals)

    def _validate_declared_controls(self) -> None:
        caps = self._caps
        for control in caps.supported_controls:
            if not hasattr(self.model, str(control)):
                raise capability_mismatch(
                    f"model declares control {control!r} but does not implement it"
                )

    def _adjust_block_size(self, block_size: int) -> int:
        name = normalize_attention_backend_name(self.attention_backend)
        if name == "auto" or not has_attention_backend(name):
            return block_size
        mult = get_attention_backend(name).capabilities().paged_block_size_multiple
        if mult <= 1 or block_size % mult == 0:
            return block_size
        adjusted = ((block_size + mult - 1) // mult) * mult
        logger.warning(
            "%s requires block_size to be a multiple of %d; adjusting %d -> %d",
            name, mult, block_size, adjusted,
        )
        return adjusted

    def _resource_classes(self) -> tuple[str, ...]:
        return self.model.resource_plan.classes()


def load_runner_engine(
    model_path: str,
    *,
    device: str,
    block_size: int,
    kv_token_capacity: int | None = None,
    load_format: str = "default",
    attention_backend: str | None = None,
    defer_sampling: bool = False,
    transfer_backend: str = "local",
    worker_kind: str | None = None,
    **kwargs: Any,
) -> RunnerDriver:
    model_path = resolve_model_path(model_path)
    descriptor = resolve_model_descriptor(_architectures(model_path))
    model_cls = descriptor.model_class
    # An und/gen worker materializes only its tower's modules (partial weight load).
    # Whole-model kinds map to ``tower_role=None`` and load everything.
    tower_role = tower_role_for_kind(worker_kind) if worker_kind is not None else None
    # Models with ``ModelBringUp.from_pretrained`` self-construct; others are
    # built from config via a registry loader.
    if issubclass(model_cls, ModelBringUp):
        model = _bring_up_via_model(
            model_cls,
            model_path,
            device=device,
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            attention_backend=attention_backend,
            tower_role=tower_role,
            **kwargs,
        )
    else:
        # Registry loaders return LoadResult; tokenizer is configured below.
        loader_override = None if str(load_format).lower() == "default" else load_format
        model = get_loader_for_descriptor(descriptor, override=loader_override).load_model(
            cast(type[UniModel], model_cls),
            read_config(model_path),
            device=device,
            model_path=model_path,
            checkpoint_layout=descriptor.checkpoint_layout,
        ).model
    _configure_model_tokenizer(model, model_path)
    _check_model_conformance(model)
    return RunnerDriver(
        model,
        block_size=block_size,
        kv_token_capacity=kv_token_capacity,
        attention_backend=attention_backend,
        defer_sampling=defer_sampling,
        transfer_backend=transfer_backend,
        worker_kind=worker_kind,
        family_descriptor=descriptor,
    )


def _bring_up_via_model(
    model_cls: type[ModelBringUp],
    model_path: str,
    *,
    device: str,
    block_size: int,
    kv_token_capacity: int | None,
    attention_backend: str | None,
    **kwargs: Any,
) -> UniModel:
    """Construct a ``ModelBringUp`` model through its ``from_pretrained`` hook."""
    model = model_cls.from_pretrained(
        model_path,
        device=device,
        block_size=block_size,
        kv_token_capacity=kv_token_capacity,
        attention_backend=attention_backend,
        **kwargs,
    )
    # Finalize quantized modules after construction (no-op for unquantized
    # models). Bring-up wrappers may expose ``modules()`` on ``self.model``.
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        modules_fn = getattr(getattr(model, "model", None), "modules", None)
    if callable(modules_fn):
        process_quantized_modules(modules_fn())
    return model


def _check_model_conformance(model: UniModel) -> None:
    violations = verify_model_conformance(model)
    if violations:
        joined = "\n  - ".join(violations)
        raise capability_mismatch(
            f"{type(model).__name__} fails capability Protocol conformance:\n  - {joined}"
        )


def _configure_model_tokenizer(model: UniModel, model_path: str) -> None:
    hook = getattr(model, "configure_tokenizer", None)
    if not callable(hook):
        return
    call_with_supported_kwargs(
        hook,
        model_path=model_path,
        tokenizer_vocab_size=_read_tokenizer_vocab_size(model_path),
    )


def _read_tokenizer_vocab_size(model_path: str) -> int | None:
    root = Path(model_path)
    max_id = -1
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            vocab = ((data.get("model") or {}).get("vocab") or {})
            if isinstance(vocab, dict):
                ids = [int(value) for value in vocab.values() if isinstance(value, int)]
                if ids:
                    max_id = max(max_id, max(ids))
            added = data.get("added_tokens") or []
            if isinstance(added, list):
                ids = [
                    int(token["id"])
                    for token in added
                    if isinstance(token, dict) and isinstance(token.get("id"), int)
                ]
                if ids:
                    max_id = max(max_id, max(ids))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Best-effort vocab probe: an unreadable/malformed tokenizer.json
            # falls back to no cap. Log so the swallow is observable.
            logger.debug("could not read vocab size from %s", tokenizer_json, exc_info=True)
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists():
        try:
            data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
            decoder = data.get("added_tokens_decoder") or {}
            if isinstance(decoder, dict):
                ids = [int(key) for key in decoder.keys()]
                if ids:
                    max_id = max(max_id, max(ids))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Best-effort vocab probe: an unreadable/malformed tokenizer_config
            # falls back to no cap. Log so the swallow is observable.
            logger.debug("could not read vocab size from %s", tokenizer_config, exc_info=True)
    return max_id + 1 if max_id >= 0 else None


def _architectures(model_path: str) -> list[str]:
    cfg = read_config(model_path)
    archs = [str(v) for v in cfg.get("architectures") or []]
    model_type = cfg.get("model_type")
    if model_type is not None:
        archs.append(str(model_type))
    if not archs:
        archs.append(Path(model_path).name)
    archs.extend(detect_model_architectures(model_path))
    return archs
