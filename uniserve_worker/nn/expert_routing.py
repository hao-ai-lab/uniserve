"""Contiguous expert state for packed text and flow transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..execution.forward_batch import ExpertRoute, RouteSpan

__all__ = ["RoutedTensor"]


@dataclass(frozen=True, slots=True)
class RoutedTensor:
    """Text and flow token tensors kept separate between shared attention calls."""

    text: torch.Tensor | None
    flow: torch.Tensor | None

    def __post_init__(self) -> None:
        """Require at least one routed tensor and validate compatible text/flow widths."""

        if self.text is None and self.flow is None:
            raise ValueError("routed tensor must contain text or flow tokens")

    @classmethod
    def from_packed(
        cls,
        value: torch.Tensor,
        spans: tuple[RouteSpan, ...],
    ) -> RoutedTensor:
        """Split a packed token axis into contiguous text and flow expert tensors."""

        if value.ndim < 1 or not spans or spans[-1].token_end != int(value.shape[0]):
            raise ValueError("packed tensor does not match its expert spans")
        text_parts = tuple(
            value.narrow(0, span.token_start, span.token_count)
            for span in spans
            if span.route is ExpertRoute.TEXT
        )
        flow_parts = tuple(
            value.narrow(0, span.token_start, span.token_count)
            for span in spans
            if span.route is ExpertRoute.FLOW
        )
        return cls(_join(text_parts), _join(flow_parts))

    def packed(self, spans: tuple[RouteSpan, ...]) -> torch.Tensor:
        """Restore text and flow tensors to the scheduler-defined packed span order."""

        if not spans:
            raise ValueError("packed expert layout must contain a span")
        offsets = {ExpertRoute.TEXT: 0, ExpertRoute.FLOW: 0}
        values = {ExpertRoute.TEXT: self.text, ExpertRoute.FLOW: self.flow}
        parts: list[torch.Tensor] = []
        for span in spans:
            source = values[span.route]
            if source is None:
                raise ValueError(f"packed layout requires missing {span.route.value} tokens")
            start = offsets[span.route]
            parts.append(source.narrow(0, start, span.token_count))
            offsets[span.route] += span.token_count
        for route, source in values.items():
            expected = offsets[route]
            actual = 0 if source is None else int(source.shape[0])
            if expected != actual:
                raise ValueError(f"{route.value} tensor does not match its packed spans")
        packed = _join(tuple(parts))
        if packed is None:
            raise RuntimeError("packed expert layout produced no tensor")
        return packed

    def map(
        self,
        text: Callable[[torch.Tensor], torch.Tensor],
        flow: Callable[[torch.Tensor], torch.Tensor],
    ) -> RoutedTensor:
        """Apply independent transforms to the populated expert tensors."""

        return RoutedTensor(
            None if self.text is None else text(self.text),
            None if self.flow is None else flow(self.flow),
        )

    def add(self, other: RoutedTensor) -> RoutedTensor:
        """Add residuals after verifying both operands contain the same expert routes."""

        if (self.text is None) != (other.text is None) or (self.flow is None) != (
            other.flow is None
        ):
            raise ValueError("routed residual operands have different experts")
        text: torch.Tensor | None = None
        if self.text is not None:
            assert other.text is not None
            text = self.text + other.text
        flow: torch.Tensor | None = None
        if self.flow is not None:
            assert other.flow is not None
            flow = self.flow + other.flow
        return RoutedTensor(text, flow)


def _join(parts: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    """Concatenate routed tensor fragments while preserving an existing single view."""

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)
