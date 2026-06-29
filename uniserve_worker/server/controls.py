"""Control-kind dispatch registry.

Each control kind maps to a :class:`ControlSpec` describing the driver method it
targets and how its (untrusted) wire fields become that method's keyword
arguments. Control methods are invoked by name with explicit keyword mapping from
wire fields to driver parameters.

The registry covers exactly ``core.contracts.CONTROL_KINDS``; the equality is
asserted at import time so a control kind added in one layer cannot silently
become a ``scheduler_bug`` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts.caps import CONTROL_KINDS
from ..foundation.errors import invalid_descriptor

__all__ = [
    'ControlSpec',
    'CONTROL_SPECS',
]


@dataclass(frozen=True)
class ControlSpec:
    """How one control kind binds its wire fields to a driver method call.

    ``method`` is the driver attribute invoked. ``scalars`` and ``lists`` map a
    *wire field name* to the *driver keyword* it fills; a wire field and the
    driver keyword usually share a name, but ``free_encoder`` reads wire field
    ``free_handles`` into the driver's ``handles`` parameter, so the mapping is
    explicit rather than positional. ``scalars`` are required (a missing field
    raises ``InvalidDescriptor`` before the driver is touched); ``lists`` default
    to an empty list when absent and reject a non-list payload.
    """

    method: str
    scalars: dict[str, str] = field(default_factory=dict)
    lists: dict[str, str] = field(default_factory=dict)

    def build_kwargs(self, kind: str, req) -> dict:
        kwargs: dict = {}
        for wire_field, param in self.scalars.items():
            kwargs[param] = _require(req, wire_field, kind)
        for wire_field, param in self.lists.items():
            kwargs[param] = _require_list(req, wire_field, kind)
        return kwargs


def _require(req, field_name: str, kind: str):
    """Read a required control field, raising InvalidDescriptor if absent.

    Control kwargs are built from untrusted wire fields; a missing field must
    fail before invocation with a typed descriptor error rather than silently
    passing ``None`` into the driver's control method.
    """
    value = req.get(field_name)
    if value is None:
        raise invalid_descriptor(
            f"control {kind!r} is missing required field {field_name!r}",
            op_kind=kind,
        )
    return value


def _require_list(req, field_name: str, kind: str) -> list:
    """Read an optional list-valued control field, validating its type."""
    value = req.get(field_name)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise invalid_descriptor(
            f"control {kind!r} field {field_name!r} must be a list, got {type(value).__name__}",
            op_kind=kind,
        )
    return list(value)


# One entry per control kind. reset_prefix_cache / sleep / wake_up take no
# arguments, so their specs carry the target method with no field bindings.
CONTROL_SPECS: dict[str, ControlSpec] = {
    "copy_blocks": ControlSpec("copy_blocks", lists={"copies": "copies"}),
    "load_lora": ControlSpec(
        "load_lora", scalars={"lora_id": "lora_id", "lora_path": "lora_path"}
    ),
    "unload_lora": ControlSpec("unload_lora", scalars={"lora_id": "lora_id"}),
    "free_encoder": ControlSpec("free_encoder", lists={"free_handles": "handles"}),
    "reset_prefix_cache": ControlSpec("reset_prefix_cache"),
    "sleep": ControlSpec("sleep"),
    "wake_up": ControlSpec("wake_up"),
}

# The registry must cover the control vocabulary exactly (no missing kind, no
# stray kind) so the dispatch table and the caps validator cannot drift.
assert set(CONTROL_SPECS) == set(CONTROL_KINDS), (
    "CONTROL_SPECS keys must equal core.contracts.CONTROL_KINDS; "
    f"missing={set(CONTROL_KINDS) - set(CONTROL_SPECS)} "
    f"extra={set(CONTROL_SPECS) - set(CONTROL_KINDS)}"
)
