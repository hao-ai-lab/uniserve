"""Semantic scope of model parameters materialized for a worker."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ModelLoadScope"]


class ModelLoadScope(StrEnum):
    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"

    @property
    def tower_role(self) -> str | None:
        if self is ModelLoadScope.UNDERSTANDING:
            return "und"
        if self is ModelLoadScope.GENERATION:
            return "gen"
        return None
