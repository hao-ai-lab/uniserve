"""Mathematical token routes independent of request or device ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from uniserve._slices import within


@dataclass(frozen=True, slots=True)
class RouteSpan:
    """One route's contiguous token interval within a packed row layout."""

    route: str
    start: int
    length: int

    def __post_init__(self):
        if not isinstance(self.route, str) or not self.route:
            raise ValueError("route names must be nonempty strings")
        if any(
            type(value) is not int or value < 0
            for value in (self.start, self.length)
        ):
            raise ValueError("route spans require nonnegative integer bounds")

    @property
    def stop(self) -> int:
        return self.start + self.length


def _counts(spans, routes):
    counts = dict.fromkeys(routes, 0)
    position = 0
    for span in spans:
        if span.route not in counts or span.start != position:
            raise ValueError(
                "route spans must cover ordered contiguous tokens using "
                "declared routes"
            )
        counts[span.route] += span.length
        position = span.stop
    return counts, position


@dataclass(frozen=True, slots=True)
class RoutedTensor:
    """Borrow per-route tensors whose row order follows their packed spans."""

    values: Mapping[str, torch.Tensor]

    def __post_init__(self):
        if not self.values or any(
            not route or value.ndim < 1 for route, value in self.values.items()
        ):
            raise ValueError(
                "routed tensors require named tensors with a token axis"
            )
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @classmethod
    def from_packed(
        cls,
        value: torch.Tensor,
        spans: tuple[RouteSpan, ...],
        *,
        routes: frozenset[str],
    ) -> RoutedTensor:
        """Split packed token rows into per-route tensors in span order."""
        _, count = _counts(spans, routes)
        if value.ndim < 1 or value.shape[0] != count:
            raise ValueError(
                "packed tensor and route spans must cover the same tokens"
            )

        values = {}
        for route in sorted(routes):
            pieces = [
                value[span.start : span.stop]
                for span in spans
                if span.route == route and span.length
            ]
            values[route] = (
                torch.cat(pieces, dim=0)
                if len(pieces) > 1
                else pieces[0]
                if pieces
                else value[:0]
            )
        return cls(values)

    def _validate(self, spans):
        counts, size = _counts(spans, self.values)
        if any(
            value.shape[0] != counts[route]
            for route, value in self.values.items()
        ):
            raise ValueError(
                "route tensors must contain exactly their declared token counts"
            )
        return size

    def packed(self, spans: tuple[RouteSpan, ...]) -> torch.Tensor:
        """Join the per-route tensors back into packed span order."""
        self._validate(spans)
        offsets = dict.fromkeys(self.values, 0)
        pieces = []
        for span in spans:
            start = offsets[span.route]
            if span.length:
                pieces.append(
                    self.values[span.route][start : start + span.length]
                )
            offsets[span.route] += span.length

        if not pieces:
            return next(iter(self.values.values()))[:0]
        return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

    def narrow(
        self, interval: slice, spans: tuple[RouteSpan, ...]
    ) -> RoutedTensor:
        """Borrow the routes' rows covered by one packed token interval."""
        size = self._validate(spans)
        if not within((interval,), (size,)):
            raise ValueError("routed interval exceeds its packed token extent")

        offsets = dict.fromkeys(self.values, 0)
        pieces: dict[str, list[torch.Tensor]] = {
            route: [] for route in self.values
        }
        for span in spans:
            start, stop = (
                max(span.start, interval.start),
                min(span.stop, interval.stop),
            )
            if start < stop:
                local = offsets[span.route] + start - span.start
                pieces[span.route].append(
                    self.values[span.route][local : local + stop - start]
                )
            offsets[span.route] += span.length

        return RoutedTensor(
            {
                route: torch.cat(parts, dim=0)
                if len(parts) > 1
                else parts[0]
                if parts
                else self.values[route][:0]
                for route, parts in pieces.items()
            }
        )

    def apply(
        self, modules: Mapping[str, nn.Module] | nn.ModuleDict
    ) -> RoutedTensor:
        """Run each route's tensor through its corresponding module."""
        if set(self.values).difference(modules):
            raise ValueError(
                "every numerical route requires a corresponding module"
            )
        # Empty routes still execute: their layers may participate in shared
        # quantization statistics or other necessary numerical collectives.
        return RoutedTensor(
            {
                route: modules[route](value)
                for route, value in self.values.items()
            }
        )

    def add(self, other: RoutedTensor) -> RoutedTensor:
        """Add another routed tensor route by route."""
        if set(other.values) != set(self.values) or any(
            value.shape != other.values[route].shape
            for route, value in self.values.items()
        ):
            raise ValueError(
                "routed addition requires matching route names and tensor "
                "shapes"
            )
        return RoutedTensor(
            {
                route: value + other.values[route]
                for route, value in self.values.items()
            }
        )
