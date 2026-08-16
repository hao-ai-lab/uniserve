"""Completeness validation for installed checkpoint tensors."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from ..nn.placement import get_shard_plan
from ..nn.quant.load_state import is_optional_checkpoint
from .mapping import LoadReport

__all__ = ["audit_load_report", "required_parameter_names"]


def required_parameter_names(
    module: nn.Module,
    *,
    included: Iterable[str] | None = None,
    optional: Iterable[str] = (),
) -> set[str]:
    expected = (
        {name for name, _ in module.named_parameters()}
        if included is None
        else {str(name) for name in included}
    )
    marked_optional = {
        name
        for name, parameter in module.named_parameters()
        if is_optional_checkpoint(parameter)
    }
    return expected.difference(marked_optional).difference(str(name) for name in optional)


def audit_load_report(
    module: nn.Module,
    report: LoadReport,
    *,
    included: Iterable[str] | None = None,
    optional: Iterable[str] = (),
    label: str = "checkpoint",
    require_packed_shards: bool = True,
) -> None:
    expected = required_parameter_names(module, included=included, optional=optional)
    missing = sorted(expected.difference(report.loaded))
    for name, parameter in module.named_parameters():
        if name not in report.loaded:
            continue
        plan = get_shard_plan(parameter)
        if not require_packed_shards or plan is None or not plan.slots:
            continue
        loaded_shards = set(getattr(parameter, "_uniserve_checkpoint_shards", set()))
        required_shards = set(plan.slots)
        if not required_shards.issubset(loaded_shards):
            missing.append(f"{name}[packed shards {sorted(required_shards - loaded_shards, key=str)!r}]")
    unexpected = sorted(set(report.unexpected))
    if missing or unexpected:
        raise RuntimeError(
            f"{label} load mismatch: missing={len(missing)} {_preview(missing)} "
            f"unexpected={len(unexpected)} {_preview(unexpected)}"
        )


def _preview(names: list[str], limit: int = 50) -> str:
    if len(names) <= limit:
        return repr(names)
    return f"{names[:limit]!r} (+{len(names) - limit} more)"
