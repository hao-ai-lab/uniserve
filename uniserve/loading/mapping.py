"""Checkpoint-name mapping and load-result reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

# Each rule maps an installed path segment, a checkpoint path segment, and a packed shard.
WeightNameMap = tuple[tuple[str, str, str | int | None], ...]

__all__ = ["LoadReport", "WeightNameMap", "stacked_weight_name"]


@dataclass(slots=True)
class LoadReport:
    """Loaded, intentionally skipped, and unexpected checkpoint tensor names."""

    loaded: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)


def stacked_weight_name(name: str, mapping: WeightNameMap) -> tuple[str, str | int | None]:
    """Map a checkpoint tensor name to its installed parameter and packed shard identifier."""

    for target, source, shard_id in mapping:
        mapped = _replace_path_segment(name, source, target)
        if mapped is not None:
            return mapped, shard_id
    return name, None


def _replace_path_segment(name: str, source: str, target: str) -> str | None:
    """Replace the first complete dotted path segment sequence that matches a source."""

    name_parts = name.split(".")
    source_parts = source.split(".")
    width = len(source_parts)
    for index in range(len(name_parts) - width + 1):
        if name_parts[index : index + width] == source_parts:
            return ".".join((*name_parts[:index], *target.split("."), *name_parts[index + width :]))
    return None
