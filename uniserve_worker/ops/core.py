"""Operator dispatch: one provider protocol and one selector."""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

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

    def launch_name(self, req: ProviderReq) -> str: ...


class Operator:
    """Shared provider identity; ``launch_name`` defaults to ``name``."""

    def __init__(self, name: str, operator: str) -> None:
        self.name = name
        self.operator = operator

    def launch_name(self, req: object) -> str:
        del req
        return self.name


class Dispatcher(Generic[Req, Res]):
    """Run the first eligible provider for one operator.

    Candidate order is an explicit override first, then registered preference
    order. An empty, missing, or ``auto`` override leaves that order unchanged.
    """

    def __init__(
        self,
        operator: str,
        providers: list[Provider[Req, Res]],
        *,
        env_override: str | None = None,
        signature: Callable[[Req], Any] | None = None,
    ) -> None:
        if not providers:
            raise ValueError(f"operator {operator!r} needs at least one provider")
        self.operator = operator
        self._providers = list(providers)
        self._by_name = {provider.name: provider for provider in self._providers}
        if len(self._by_name) != len(self._providers):
            raise ValueError(f"operator {operator!r} has duplicate provider names")
        self._env_override = env_override
        self._signature = signature
        self._memo: dict[tuple[str | None, Any], str] = {}

    def provider_names(self) -> tuple[str, ...]:
        return tuple(provider.name for provider in self._providers)

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
        return selected

    def _ordered(self, override: str | None) -> list[Provider[Req, Res]]:
        selected = self._resolve_override(override)
        if selected is None:
            return list(self._providers)
        match = self._by_name.get(selected)
        if match is None:
            available = ", ".join(self.provider_names())
            raise ValueError(
                f"unknown provider override {selected!r} for operator {self.operator!r}; "
                f"available providers: {available}"
            )
        return [match, *[provider for provider in self._providers if provider is not match]]

    def _observe(self, provider: Provider[Req, Res], req: Req) -> Res:
        stats = getattr(req, "stats", None)
        if stats is None:
            ctx = getattr(req, "ctx", None)
            stats = getattr(ctx, "stats", None) if ctx is not None else None
        record = getattr(stats, "record_operator_launch", None) if stats is not None else None
        if not callable(record):
            return provider.run(req)
        start = time.perf_counter_ns()
        try:
            return provider.run(req)
        finally:
            record(self.operator, provider.launch_name(req), time.perf_counter_ns() - start)

    def run(self, req: Req, *, override: str | None = None) -> Res:
        key = None
        if self._signature is not None:
            key = (self._resolve_override(override), self._signature(req))
            memo_name = self._memo.get(key)
            if memo_name is not None:
                provider = self._by_name[memo_name]
                if provider.can_run(req):
                    return self._observe(provider, req)
        for provider in self._ordered(override):
            if provider.can_run(req):
                if key is not None:
                    self._memo[key] = provider.name
                return self._observe(provider, req)
        names = ", ".join(self.provider_names())
        raise RuntimeError(f"operator {self.operator!r} has no eligible provider among {names}")
