"""Backend/operator dispatch framework for worker compute ops."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, runtime_checkable
import os
import time
import torch

from ..foundation.torch_compat import torch_is_compiling as _torch_is_compiling

Req = TypeVar("Req")
Res = TypeVar("Res")
DispatchReq = TypeVar("DispatchReq")
CombineRes = TypeVar("CombineRes")
Adapted = TypeVar("Adapted")


@dataclass(frozen=True)
class Capabilities:
    """Static provider capabilities used for coarse selection and introspection."""

    tags: frozenset[str] = frozenset()
    priority: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.attrs.get(name, default)


@runtime_checkable
class Provider(Protocol[Req, Res]):
    name: str
    operator: str

    def capabilities(self) -> Capabilities: ...

    def can_run(self, req: Req) -> bool: ...

    def run(self, req: Req) -> Res: ...


@dataclass(frozen=True)
class Handoff:
    """Format-tagged state passed between phases of a communication op.

    Composite ops such as MoE choose a comm provider and a compute provider on
    separate axes. The comm dispatch phase returns one of these handoffs; adapter
    pools can then translate ``handoff.format`` into the compute provider's
    preferred input layout before the same comm provider combines the result.
    """

    format: str
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    operator: str | None = None
    provider: str | None = None


@runtime_checkable
class CommProvider(Protocol[DispatchReq, CombineRes]):
    name: str
    operator: str

    def capabilities(self) -> Capabilities: ...

    def can_dispatch(self, req: DispatchReq, *, mesh: Any | None = None) -> bool: ...

    def dispatch(self, req: DispatchReq, *, mesh: Any | None = None) -> Handoff: ...

    def can_combine(self, handoff: Handoff, *, mesh: Any | None = None) -> bool: ...

    def combine(self, handoff: Handoff, *, mesh: Any | None = None) -> CombineRes: ...




class Dispatcher(Generic[Req, Res]):
    """Run the first eligible provider for one operator.

    Candidate order is explicit override first, then registered preference order,
    with the eager provider as the terminal fallback.
    """

    def __init__(
        self,
        operator: str,
        providers: list[Provider[Req, Res]],
        *,
        env_override: str | None = None,
        signature: Callable[[Req], Any] | None = None,
        fallback_names: tuple[str, ...] = ("eager",),
    ) -> None:
        if not providers:
            raise ValueError(f"operator {operator!r} needs at least one provider")
        self.operator = operator
        self._providers = list(providers)
        self._env_override = env_override
        self._signature = signature
        self._memo: dict[tuple[str | None, Any], str] = {}
        self._fallback_names = fallback_names
        if not any(p.name in self._fallback_names for p in self._providers):
            raise ValueError(f"operator {operator!r} has no terminal fallback provider")

    @property
    def providers(self) -> tuple[Provider[Req, Res], ...]:
        return tuple(self._providers)

    def provider_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._providers)

    def ordered(self, override: str | None = None) -> tuple[Provider[Req, Res], ...]:
        return tuple(self._ordered(override))

    def _resolve_override(self, override: str | None) -> str | None:
        selected = override
        if selected is None and self._env_override:
            selected = os.environ.get(self._env_override)
        if selected is None:
            return None
        selected = str(selected).strip()
        if not selected or selected.lower() == "auto":
            return None
        if selected.lower() in {"0", "false", "off"}:
            return self._fallback_names[0]
        if selected.lower() == "eager":
            if any(provider.name == "eager" for provider in self._providers):
                return "eager"
            return self._fallback_names[0]
        if selected.lower() in {"1", "true", "on"}:
            return None
        return selected

    def _raise_unknown_override(self, selected: str) -> None:
        available = ", ".join(self.provider_names())
        raise ValueError(
            f"unknown provider override {selected!r} for operator {self.operator!r}; "
            f"available providers: {available}"
        )

    def _ordered(self, override: str | None) -> list[Provider[Req, Res]]:
        selected = self._resolve_override(override)
        if selected is None:
            return list(self._providers)
        matches = [p for p in self._providers if p.name == selected]
        rest = [p for p in self._providers if p.name != selected]
        if not matches:
            self._raise_unknown_override(selected)
        return matches + rest

    def _observe(self, provider: Provider[Req, Res], req: Req) -> Res:
        stats = getattr(req, "stats", None)
        if stats is None:
            try:
                from ..contracts.forward_context import get_forward_context

                stats = getattr(get_forward_context(), "stats", None)
            except Exception:
                stats = None
        if stats is None or _torch_is_compiling():
            return provider.run(req)
        start = time.perf_counter_ns()
        try:
            return provider.run(req)
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            if self.operator == "attention":
                record_attention = getattr(stats, "record_attention_launch", None)
                if callable(record_attention):
                    display_name = getattr(provider, "display_name", None)
                    name = display_name(req) if callable(display_name) else provider.name
                    record_attention(name, elapsed_ns)
                else:
                    record = getattr(stats, "record_operator_launch", None)
                    if callable(record):
                        record(self.operator, provider.name, elapsed_ns)
            else:
                record = getattr(stats, "record_operator_launch", None)
                if callable(record):
                    record(self.operator, provider.name, elapsed_ns)

    def run(self, req: Req, *, override: str | None = None) -> Res:
        key = None
        if self._signature is not None:
            try:
                key = (self._resolve_override(override), self._signature(req))
                memo_name = self._memo.get(key)
                if memo_name is not None:
                    for provider in self._ordered(memo_name):
                        if provider.name == memo_name and provider.can_run(req):
                            return self._observe(provider, req)
            except Exception:
                key = None
        for provider in self._ordered(override):
            if provider.can_run(req):
                if key is not None:
                    self._memo[key] = provider.name
                return self._observe(provider, req)
        raise RuntimeError(f"operator {self.operator!r} has no eligible provider; eager provider is broken")


class CommDispatcher(Generic[DispatchReq, CombineRes]):
    """Two-phase dispatcher for stateful collective/communication providers."""

    def __init__(
        self,
        operator: str,
        providers: list[CommProvider[DispatchReq, CombineRes]],
        *,
        env_override: str | None = None,
        fallback_names: tuple[str, ...] = ("standard", "eager"),
    ) -> None:
        if not providers:
            raise ValueError(f"comm operator {operator!r} needs at least one provider")
        self.operator = operator
        self._providers = list(providers)
        self._env_override = env_override
        self._fallback_names = fallback_names
        if not any(p.name in self._fallback_names for p in self._providers):
            raise ValueError(f"comm operator {operator!r} has no terminal fallback provider")

    @property
    def providers(self) -> tuple[CommProvider[DispatchReq, CombineRes], ...]:
        return tuple(self._providers)

    def provider_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._providers)

    def _resolve_override(self, override: str | None) -> str | None:
        selected = override
        if selected is None and self._env_override:
            selected = os.environ.get(self._env_override)
        if selected is None:
            return None
        selected = str(selected).strip()
        if not selected or selected.lower() == "auto":
            return None
        if selected.lower() in {"0", "false", "off", "standard", "eager"}:
            return self._fallback_names[0]
        if selected.lower() in {"1", "true", "on"}:
            return None
        return selected

    def _raise_unknown_override(self, selected: str) -> None:
        available = ", ".join(self.provider_names())
        raise ValueError(
            f"unknown comm provider override {selected!r} for operator {self.operator!r}; "
            f"available providers: {available}"
        )

    def _ordered(self, override: str | None) -> list[CommProvider[DispatchReq, CombineRes]]:
        selected = self._resolve_override(override)
        if selected is None:
            return list(self._providers)
        matches = [p for p in self._providers if p.name == selected]
        rest = [p for p in self._providers if p.name != selected]
        if not matches:
            self._raise_unknown_override(selected)
        return matches + rest

    def dispatch(
        self,
        req: DispatchReq,
        *,
        override: str | None = None,
        mesh: Any | None = None,
    ) -> Handoff:
        for provider in self._ordered(override):
            if provider.can_dispatch(req, mesh=mesh):
                handoff = provider.dispatch(req, mesh=mesh)
                return replace(
                    handoff,
                    operator=handoff.operator or self.operator,
                    provider=handoff.provider or provider.name,
                )
        raise RuntimeError(
            f"comm operator {self.operator!r} has no eligible dispatch provider; "
            "terminal fallback provider is broken"
        )

    def combine(
        self,
        handoff: Handoff,
        *,
        override: str | None = None,
        mesh: Any | None = None,
    ) -> CombineRes:
        selected = override or handoff.provider
        for provider in self._ordered(selected):
            if provider.can_combine(handoff, mesh=mesh):
                return provider.combine(handoff, mesh=mesh)
        raise RuntimeError(
            f"comm operator {self.operator!r} has no eligible combine provider "
            f"for handoff format {handoff.format!r}"
        )


class AdapterPool:
    """Format adapters for composite ops with independent comm/compute axes."""

    def __init__(self) -> None:
        self._pre: dict[tuple[str, str], Callable[[Handoff], Any]] = {}
        self._post: dict[tuple[str, str], Callable[[Any], Handoff]] = {}

    def register_pre(
        self,
        handoff_format: str,
        compute_provider: str,
        fn: Callable[[Handoff], Any],
    ) -> None:
        self._pre[(str(handoff_format), str(compute_provider))] = fn

    def register_post(
        self,
        compute_provider: str,
        combine_format: str,
        fn: Callable[[Any], Handoff],
    ) -> None:
        self._post[(str(compute_provider), str(combine_format))] = fn

    def adapt_pre(self, handoff: Handoff, compute_provider: str) -> Any:
        key = (handoff.format, str(compute_provider))
        try:
            return self._pre[key](handoff)
        except KeyError as exc:
            raise KeyError(
                f"no pre-adapter for handoff format {handoff.format!r} "
                f"and compute provider {compute_provider!r}"
            ) from exc

    def adapt_post(self, value: Any, compute_provider: str, combine_format: str) -> Handoff:
        key = (str(compute_provider), str(combine_format))
        try:
            return self._post[key](value)
        except KeyError as exc:
            raise KeyError(
                f"no post-adapter for compute provider {compute_provider!r} "
                f"and combine format {combine_format!r}"
            ) from exc


class FusedOpPool:
    """Registry for co-designed composite fast paths."""

    def __init__(self) -> None:
        self._ops: dict[tuple[str, ...], Callable[..., Any]] = {}

    def register(self, axes: tuple[str, ...], fn: Callable[..., Any]) -> None:
        if not axes:
            raise ValueError("fused op axes must not be empty")
        self._ops[tuple(str(axis) for axis in axes)] = fn

    def get(self, axes: tuple[str, ...]) -> Callable[..., Any] | None:
        return self._ops.get(tuple(str(axis) for axis in axes))
