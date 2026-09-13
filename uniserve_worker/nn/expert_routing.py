"""Contiguous expert state for packed text and flow transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from uniserve_worker.modeling.tensors import ExpertRoute, RouteSpan

__all__ = ["RoutedTensor"]


def slice_route_spans(spans: tuple[RouteSpan, ...], interval: slice) -> tuple[RouteSpan, ...]:
    """Intersect ordered expert spans with a packed row interval and rebase it."""

    return tuple(
        RouteSpan(
            span.route,
            max(interval.start, span.token_start) - interval.start,
            min(interval.stop, span.token_end) - max(interval.start, span.token_start),
        )
        for span in spans
        if span.token_start < interval.stop and span.token_end > interval.start
    )


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
        *,
        routes: frozenset[ExpertRoute] = frozenset(),
    ) -> RoutedTensor:
        """Split a packed token axis into contiguous text and flow expert tensors."""

        extent = spans[-1].token_end if spans else 0
        if value.ndim < 1 or extent != int(value.shape[0]):
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
        text, flow = _join(text_parts), _join(flow_parts)
        if text is None and ExpertRoute.TEXT in routes:
            text = value[:0]
        if flow is None and ExpertRoute.FLOW in routes:
            flow = value[:0]
        return cls(text, flow)

    @property
    def routes(self) -> frozenset[ExpertRoute]:
        """Experts participating in this execution, including empty local shards."""

        return frozenset(
            route
            for route, value in ((ExpertRoute.TEXT, self.text), (ExpertRoute.FLOW, self.flow))
            if value is not None
        )

    def packed(self, spans: tuple[RouteSpan, ...]) -> torch.Tensor:
        """Restore text and flow tensors to the scheduler-defined packed span order."""

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
            empty = self.text if self.text is not None else self.flow
            assert empty is not None
            return empty
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

    def narrow(self, interval: slice, spans: tuple[RouteSpan, ...]) -> RoutedTensor:
        """Select packed rows while retaining each expert's contiguous storage."""

        starts = {ExpertRoute.TEXT: 0, ExpertRoute.FLOW: 0}
        counts = {ExpertRoute.TEXT: 0, ExpertRoute.FLOW: 0}
        for span in spans:
            starts[span.route] += max(0, min(span.token_end, interval.start) - span.token_start)
            counts[span.route] += max(
                0, min(span.token_end, interval.stop) - max(span.token_start, interval.start)
            )
        return RoutedTensor(
            None
            if self.text is None
            else self.text.narrow(0, starts[ExpertRoute.TEXT], counts[ExpertRoute.TEXT]),
            None
            if self.flow is None
            else self.flow.narrow(0, starts[ExpertRoute.FLOW], counts[ExpertRoute.FLOW]),
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

    def apply(
        self,
        *,
        text: torch.nn.Module,
        flow: torch.nn.Module,
    ) -> RoutedTensor:
        """Apply the bound numerical modules to each populated expert."""

        return RoutedTensor(
            None if self.text is None else text(self.text),
            None if self.flow is None else flow(self.flow),
        )


def _join(parts: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    """Concatenate routed tensor fragments while preserving an existing single view."""

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)
