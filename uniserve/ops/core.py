"""Provider selection for concrete numerical operator inputs.

Explicit or environment-selected ordering and numerical signatures choose one
eligible provider. Execution observations remain with the calling
infrastructure.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Generic, TypeVar

Req = TypeVar("Req")
Res = TypeVar("Res")


class Operator:
    """Provider interface for one named implementation of an operator family."""

    def __init__(self, name: str, operator: str) -> None:
        """Identify the provider and the operator family it implements."""
        self.name = name
        self.operator = operator

    def launch_name(self, req: object) -> str:
        """Return the provider label recorded for a concrete launch."""
        del req
        return self.name

    def can_run(self, req: Any) -> bool:
        """Report whether this provider supports ``req``.

        Report whether this provider supports ``req`` on its current device.
        """
        raise NotImplementedError

    def run(self, req: Any) -> Any:
        """Execute a supported request and return the operator result."""
        raise NotImplementedError


class Dispatcher(Generic[Req, Res]):
    """Priority-ordered dispatcher for interchangeable operator providers.

    Candidate order is an explicit override first, then registered preference
    order. An empty, missing, or ``auto`` override leaves preference order
    unchanged. Signature-based memoization avoids repeating full selection for
    equivalent requests while retaining the provider's eligibility check.
    """

    def __init__(
        self,
        operator: str,
        providers: list[Operator],
        *,
        env_override: str | None = None,
        signature: Callable[[Req], Any] | None = None,
    ) -> None:
        """Register providers and optional sources.

        Register ordered providers and optional override and signature
        sources.
        """
        if not providers:
            raise ValueError(
                f"operator {operator!r} needs at least one provider"
            )

        self.operator = operator
        self._providers = list(providers)
        self._by_name = {
            provider.name: provider for provider in self._providers
        }
        if len(self._by_name) != len(self._providers):
            raise ValueError(
                f"operator {operator!r} has duplicate provider names"
            )

        self._env_override = env_override
        self._signature = signature
        self._memo: dict[tuple[str | None, Any], str] = {}

    def provider_names(self) -> tuple[str, ...]:
        """Return provider names in default preference order."""
        return tuple(provider.name for provider in self._providers)

    def ordered(self, override: str | None = None) -> tuple[Operator, ...]:
        """Return provider order after applying an optional override."""
        return tuple(self._ordered(override))

    def _resolve_override(self, override: str | None) -> str | None:
        """Normalize an explicit or environment-provided provider selection."""
        selected = override
        if selected is None and self._env_override:
            selected = os.environ.get(self._env_override)
        if selected is None:
            return None

        selected = str(selected).strip()
        if not selected or selected.lower() == "auto":
            return None

        return selected

    def _ordered(self, override: str | None) -> list[Operator]:
        """Place an overridden provider first.

        Place an overridden provider before the registered preference order.
        """
        selected = self._resolve_override(override)
        if selected is None:
            return list(self._providers)

        match = self._by_name.get(selected)
        if match is None:
            available = ", ".join(self.provider_names())
            raise ValueError(
                f"unknown provider override {selected!r} for operator "
                f"{self.operator!r}; "
                f"available providers: {available}"
            )

        return [
            match,
            *[
                provider
                for provider in self._providers
                if provider is not match
            ],
        ]

    def run(self, req: Req, *, override: str | None = None) -> Res:
        """Select and execute an eligible provider for ``req``."""
        # A request signature memoizes selection rather than execution. The
        # cached provider still confirms eligibility because device state and
        # request capabilities can vary within one signature bucket.
        key = None
        if self._signature is not None:
            key = (self._resolve_override(override), self._signature(req))
            memo_name = self._memo.get(key)
            if memo_name is not None:
                provider = self._by_name[memo_name]
                if provider.can_run(req):
                    return provider.run(req)

        # Scan current preference order when no reusable selection exists and
        # remember the winner for later requests with the same signature.
        for provider in self._ordered(override):
            if provider.can_run(req):
                if key is not None:
                    self._memo[key] = provider.name
                return provider.run(req)

        names = ", ".join(self.provider_names())
        raise RuntimeError(
            f"operator {self.operator!r} has no eligible provider among {names}"
        )
