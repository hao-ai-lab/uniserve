"""Observable validation results shared by benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    checks: dict[str, bool]
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": dict(self.checks),
            "statistics": dict(self.statistics),
            "warnings": list(self.warnings),
        }

    def merged(self, other: ValidationResult) -> ValidationResult:
        overlap = set(self.checks) & set(other.checks)
        if overlap:
            raise ValueError(f"duplicate validation checks: {', '.join(sorted(overlap))}")
        return ValidationResult(
            checks={**self.checks, **other.checks},
            statistics={**self.statistics, **other.statistics},
            warnings=(*self.warnings, *other.warnings),
        )
