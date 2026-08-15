"""Backend/operator dispatch framework for worker compute ops."""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

from ..foundation.torch_compat import torch_is_compiling as _torch_is_compiling

Req = TypeVar("Req")
Res = TypeVar("Res")
ProviderReq = TypeVar("ProviderReq", contravariant=True)
ProviderRes = TypeVar("ProviderRes", covariant=True)


@runtime_checkable
class Provider(Protocol[ProviderReq, ProviderRes]):
    name: str
    operator: str

    def can_run(self, req: ProviderReq) -> bool: ...

    def run(self, req: ProviderReq) -> ProviderRes: ...

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
