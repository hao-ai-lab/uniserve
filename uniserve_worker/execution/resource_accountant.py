"""Per-group resource accounting for the model runner.

Owns the all-or-nothing per-step resource accounting (KV blocks, image latent,
scratch) for an op group, with rollback of any residency a partially-accounted
group acquired before a later op failed admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..contracts.op_kinds import DENOISE_GEN
from ..contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan
from ..foundation.errors import WorkerError
from ..foundation.sizing import ceil_div
from ..runtime.image_defaults import image_height, image_width
from ..runtime.request_state import RequestStateTable
from ..runtime.resources import ResourceRuntime

if TYPE_CHECKING:
    from ..runtime.residency import ResidencyManager

__all__ = [
    'ResourceAccountant',
]


@dataclass
class _AccountingDelta:
    """What a single op newly acquired during per-group resource accounting.

    Used to undo a partially-accounted group when a later op in the same group
    fails admission. Only per-step transient residency is tracked here;
    resident KV blocks mirror host leases and are not rolled back.
    """

    req_id: int
    acquired_latent: bool = False
    acquired_scratch: bool = False


class ResourceAccountant:
    """All-or-nothing per-step resource accounting with rollback."""

    def __init__(
        self,
        resource_runtime: ResourceRuntime,
        request_states: RequestStateTable,
        resource_plan: ResourcePlan,
        residency: "ResidencyManager | None" = None,
    ):
        self.resource_runtime = resource_runtime
        self.request_states = request_states
        self.resource_plan = resource_plan
        self.residency = residency

    def account_group(self, group: list[tuple[int, Mapping[str, Any]]]) -> None:
        """All-or-nothing per-step resource accounting for one group's ops.

        If a later op cannot be admitted, undo residency the earlier ops in
        this group acquired so a partially-accounted group never leaves stranded
        leases behind.
        """
        accounted: list[_AccountingDelta] = []
        try:
            for _, op in group:
                accounted.append(self._account_op_resources(op))
        except (WorkerError, RuntimeError):
            self.rollback(accounted)
            raise

    def rollback(self, accounted: list["_AccountingDelta"]) -> None:
        """Undo the per-step transient residency this group's ops acquired.

        Only the image_latent/scratch views newly activated by *this* group are
        released. Resident KV blocks are intentionally left in place: they mirror
        the host's logical lease for `new_block_ids` and outlive a single step, so
        releasing them here would desync the worker from the host's ledger. They
        are reclaimed on request teardown via release_request/drop_request.
        """
        for delta in reversed(accounted):
            req_id = delta.req_id
            state = self.request_states.get(req_id)
            if delta.acquired_scratch:
                self.release_class_if_managed("scratch", req_id)
                state.deactivate_scratch()
            if delta.acquired_latent:
                self.release_class_if_managed("image_latent", req_id)
                state.deactivate_image_latent()

    def _account_op_resources(self, op: Mapping[str, Any]) -> "_AccountingDelta":
        req_id = int(op["req_id"])
        self.account_blocks(req_id, op.get("new_block_ids") or [])
        delta = _AccountingDelta(req_id=req_id)
        kind = op.get("kind")
        if kind == DENOISE_GEN:
            state = self.request_states.get(req_id)
            acquired_latent = False
            acquired_scratch = False
            latent_rule = self.resource_plan.image_latent
            scratch_rule = self.resource_plan.scratch
            if latent_rule is not None and not state.residency.image_latent_active:
                self._acquire_if_managed(
                    "image_latent",
                    req_id,
                    self.latent_units(op, state.image, latent_rule),
                )
                state.activate_image_latent()
                acquired_latent = True
            try:
                if scratch_rule is not None and not state.residency.scratch_active:
                    self._acquire_if_managed(
                        "scratch",
                        req_id,
                        self._scratch_units(op, scratch_rule),
                    )
                    state.activate_scratch()
                    acquired_scratch = True
            except (WorkerError, RuntimeError):
                if acquired_scratch:
                    self.release_class_if_managed("scratch", req_id)
                    state.deactivate_scratch()
                if acquired_latent:
                    self.release_class_if_managed("image_latent", req_id)
                    state.deactivate_image_latent()
                raise
            delta.acquired_latent = acquired_latent
            delta.acquired_scratch = acquired_scratch
        return delta

    def account_blocks(
        self,
        req_id: int,
        block_ids: list[int] | tuple[int, ...],
        *,
        append_to_state: bool = True,
    ) -> None:
        state = self.request_states.get(req_id)
        if append_to_state and block_ids:
            state.extend_block_ids(tuple(int(block_id) for block_id in block_ids))
        if "kv_block" not in self.resource_runtime.classes:
            return
        incoming = {int(block_id) for block_id in block_ids}
        new_blocks = incoming - state.resident_block_ids
        if new_blocks:
            self.resource_runtime.acquire("kv_block", req_id, len(new_blocks))
            state.resident_block_ids.update(new_blocks)

    def latent_units(
        self,
        op: Mapping[str, Any],
        image: Mapping[str, Any],
        rule: LatentTokens,
    ) -> int:
        shape = op.get("latent_shape") or image.get("latent_shape")
        if shape:
            units = 1
            for value in shape:
                units *= max(1, int(value))
            return units
        h = image_height(image)
        w = image_width(image)
        downsample = int(image.get("latent_downsample") or rule.downsample or 16)
        return _ceil_div(max(1, h), downsample) * _ceil_div(max(1, w), downsample)

    def _scratch_units(self, op: Mapping[str, Any], rule: PerBranch) -> int:
        # Scratch is accounted in CFG branch slots (one per active branch).
        cfg = op.get("cfg")
        branch_count = 1
        if isinstance(cfg, Mapping):
            branch_count = int(cfg.get("branch_count") or 1)
        return max(int(rule.minimum), branch_count)

    def _acquire_if_managed(self, cls: str, req_id: int, units: int) -> None:
        if cls in self.resource_runtime.classes:
            self.resource_runtime.acquire(cls, req_id, max(1, int(units)))

    def release_class_if_managed(self, cls: str, req_id: int) -> None:
        if cls in self.resource_runtime.classes:
            self.resource_runtime.release_class(cls, req_id)


def _ceil_div(value: int, divisor: int) -> int:
    return ceil_div(value, divisor)
