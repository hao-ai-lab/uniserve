"""System-owned model compile planning and application."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from weakref import WeakSet

import torch
import torch.nn as nn

from ..foundation.env import DEFAULT_COMPILE_BACKEND
from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import get_execution_config
from ..foundation.triton_compat import ensure_blackwell_ptxas

__all__ = [
    "is_compiled",
    "TorchCompileConfig",
    "CompileTarget",
    "CompileReport",
    "maybe_compile_module",
    "compile_targets",
    "compile_model_pieces",
    "named_child_compile_targets",
]

logger = logging.getLogger(__name__)

# Modules already wrapped by ``maybe_compile_module`` in this process, tracked by
# the compile subsystem instead of stamping a marker attribute onto foreign
# module objects. A ``WeakSet`` lets compiled modules be garbage-collected
# normally; membership drives the idempotent "don't double-compile" guard.
_COMPILED_MODULES: "WeakSet[nn.Module]" = WeakSet()


def is_compiled(module: nn.Module) -> bool:
    """Whether this module was produced/registered by ``maybe_compile_module``."""
    try:
        return module in _COMPILED_MODULES
    except TypeError:  # pragma: no cover - unhashable/weakref-incapable wrappers
        return False


def _mark_compiled(module: nn.Module) -> None:
    try:
        _COMPILED_MODULES.add(module)
    except TypeError:  # pragma: no cover - wrappers that reject weak references
        logger.debug("could not track torch.compile marker for a module")


@dataclass(frozen=True)
class TorchCompileConfig:
    """Opt-in torch.compile settings read from worker runtime config."""

    enabled: bool = False
    backend: str = DEFAULT_COMPILE_BACKEND
    # No torch.compile mode by default: inductor fusion only, not CUDA graphs.
    # ``reduce-overhead`` would wrap the model in torch cudagraph trees that
    # collide with in-place fused_add_rmsnorm and explicit ``torch.cuda.graph``.
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None

    @classmethod
    def from_runtime_config(cls) -> "TorchCompileConfig":
        cfg = get_execution_config().torch_compile
        return cls(
            enabled=cfg.enabled,
            backend=cfg.backend,
            mode=cfg.mode,
            fullgraph=cfg.fullgraph,
            dynamic=cfg.dynamic,
        )


@dataclass(frozen=True)
class CompileTarget:
    """One replaceable module selected by a model for piecewise compilation."""

    label: str
    module: nn.Module
    owner: nn.Module | None = None
    attr_name: str | None = None


@dataclass(frozen=True)
class CompileReport:
    """Summary of a piecewise compile pass."""

    attempted: int
    compiled: int
    labels: tuple[str, ...]


def maybe_compile_module(
    module: nn.Module,
    *,
    label: str,
    config: TorchCompileConfig | None = None,
) -> nn.Module:
    """Compile ``module`` when explicitly enabled; otherwise return it unchanged."""

    cfg = config or TorchCompileConfig.from_runtime_config()
    if not cfg.enabled:
        return module
    # UniServe owns explicit CUDA-graph capture; reject cudagraph-tree modes.
    if cfg.mode in {"reduce-overhead", "max-autotune"}:
        raise capability_mismatch(
            f"torch.compile mode {cfg.mode!r} drives its own cudagraph trees, which "
            "collide with UniServe's explicit CUDA-graph capture; leave mode unset"
        )
    if is_compiled(module):
        return module
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        raise capability_mismatch("torch.compile is enabled but torch.compile is unavailable")
    _prepare_compile_toolchain(module, label=label)
    kwargs: dict[str, Any] = {
        "backend": cfg.backend,
        "fullgraph": cfg.fullgraph,
    }
    if cfg.mode is not None:
        kwargs["mode"] = cfg.mode
    if cfg.dynamic is not None:
        kwargs["dynamic"] = cfg.dynamic
    try:
        compiled = compile_fn(module, **kwargs)
    except Exception as exc:  # noqa: BLE001 - explicit opt-in should fail loudly.
        raise capability_mismatch(f"torch.compile failed while compiling {label}") from exc
    _mark_compiled(compiled)
    logger.info(
        "torch.compile enabled for %s backend=%s mode=%s fullgraph=%s dynamic=%s",
        label,
        cfg.backend,
        cfg.mode,
        cfg.fullgraph,
        cfg.dynamic,
    )
    return compiled


def _prepare_compile_toolchain(module: nn.Module, *, label: str) -> None:
    device = _first_module_device(module)
    if device is None or device.type != "cuda":
        return
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return
    if int(major) < 10:
        return
    if ensure_blackwell_ptxas():
        return
    raise capability_mismatch(
        f"torch.compile for {label} requires a CUDA 13+ ptxas on Blackwell GPUs"
    )


def _first_module_device(module: nn.Module) -> torch.device | None:
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def compile_targets(
    targets: Iterable[CompileTarget],
    *,
    config: TorchCompileConfig | None = None,
) -> CompileReport:
    """Compile and replace a declared set of piecewise modules."""

    cfg = config or TorchCompileConfig.from_runtime_config()
    if not cfg.enabled:
        return CompileReport(attempted=0, compiled=0, labels=())
    attempted = 0
    compiled = 0
    labels: list[str] = []
    for target in targets:
        attempted += 1
        before = is_compiled(target.module)
        compiled_module = maybe_compile_module(target.module, label=target.label, config=cfg)
        if target.owner is not None and target.attr_name:
            setattr(target.owner, target.attr_name, compiled_module)
        after = is_compiled(compiled_module)
        if after and not before:
            compiled += 1
            labels.append(target.label)
    return CompileReport(attempted=attempted, compiled=compiled, labels=tuple(labels))


def compile_model_pieces(
    model: Any,
    *,
    config: TorchCompileConfig | None = None,
) -> CompileReport:
    """Compile model-declared piecewise targets when the opt-in flag is set."""

    cfg = config or TorchCompileConfig.from_runtime_config()
    if not cfg.enabled:
        return CompileReport(attempted=0, compiled=0, labels=())
    hook = getattr(model, "compile_targets", None)
    if not callable(hook):
        return CompileReport(attempted=0, compiled=0, labels=())
    return compile_targets(hook(), config=cfg)


def named_child_compile_targets(
    root: nn.Module,
    *,
    predicate: Callable[[str, nn.Module], bool],
    label_prefix: str = "",
) -> tuple[CompileTarget, ...]:
    """Collect direct-child modules matching ``predicate`` with replace handles."""

    prefix = label_prefix.rstrip(".")
    targets: list[CompileTarget] = []
    for owner_name, owner in root.named_modules():
        for child_name, child in owner.named_children():
            qualname = child_name if not owner_name else f"{owner_name}.{child_name}"
            if predicate(qualname, child):
                label = f"{prefix}.{qualname}" if prefix else qualname
                targets.append(
                    CompileTarget(
                        label=label,
                        module=child,
                        owner=owner,
                        attr_name=child_name,
                    )
                )
    return tuple(targets)
