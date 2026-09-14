"""Numerical methods and their mathematical component participation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ComponentCall:
    """Identify one actual module method, its pipeline stage and collective axes.

    An empty component path selects the root module. A kind suffix on encode
    or decode describes the numerical modality; it does not name a worker op.
    """

    component: str
    method: str
    stage: Literal["all", "first", "last"] = "all"
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.component and any(not part.isidentifier() for part in self.component.split(".")):
            raise ValueError("component must be an ordinary module path")
        if not self.method or self.stage not in {"all", "first", "last"}:
            raise ValueError("component calls require a method and a pipeline stage")
        if len(set(self.groups)) != len(self.groups) or any(
            axis not in {"tp", "pp", "sp", "ulysses", "cp", "cp_row", "cp_col"}
            for axis in self.groups
        ):
            raise ValueError("component communication must name distinct mathematical axes")
