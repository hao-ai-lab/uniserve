"""Completeness validation for installed checkpoint tensors."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from uniserve.loading.mapping import LoadReport
from uniserve.nn.quant.load_state import is_optional_checkpoint
from uniserve.nn.shard import get_shard_plan

__all__ = ["validate_loaded_weights", "required_parameter_names"]


def required_parameter_names(
    module: nn.Module,
    *,
    included: Iterable[str] | None = None,
    optional: Iterable[str] = (),
) -> set[str]:
    """Return required parameters after applying scope and checkpoint-optional policy."""

    expected = (
        {name for name, _ in module.named_parameters()}
        if included is None
        else {str(name) for name in included}
    )
    marked_optional = {
        name for name, parameter in module.named_parameters() if is_optional_checkpoint(parameter)
    }
    return expected.difference(marked_optional).difference(str(name) for name in optional)


def validate_loaded_weights(
    module: nn.Module,
    report: LoadReport,
    *,
    included: Iterable[str] | None = None,
    optional: Iterable[str] = (),
    label: str = "checkpoint",
) -> None:
    """Reject an incomplete or inconsistent load report for the selected model scope.

    Packed parameters are complete only when every logical shard declared by their
    assignment plan has been installed.
    """

    # Derive the scalar parameter contract before checking packed-shard completeness.
    expected = required_parameter_names(module, included=included, optional=optional)
    missing = sorted(expected.difference(report.loaded))
    for name, parameter in module.named_parameters():
        if name not in report.loaded:
            continue
        plan = get_shard_plan(parameter)
        if plan is None or not plan.slots:
            continue
        loaded_shards = set(getattr(parameter, "_uniserve_checkpoint_shards", set()))
        required_shards = set(plan.slots)
        if not required_shards.issubset(loaded_shards):
            missing.append(
                f"{name}[packed shards {sorted(required_shards - loaded_shards, key=str)!r}]"
            )

    # Present both sides of the mismatch together so one load attempt is actionable.
    unexpected = sorted(set(report.unexpected))
    if missing or unexpected:
        raise RuntimeError(
            f"{label} load mismatch: missing={len(missing)} {_preview(missing)} "
            f"unexpected={len(unexpected)} {_preview(unexpected)}"
        )


def _preview(names: list[str], limit: int = 50) -> str:
    """Format a bounded name list while retaining the omitted-item count."""

    if len(names) <= limit:
        return repr(names)
    return f"{names[:limit]!r} (+{len(names) - limit} more)"
