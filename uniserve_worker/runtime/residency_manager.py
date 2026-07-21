"""Logical residency leases over worker-owned physical resource pools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..contracts.op_kinds import DENOISE_GEN
from ..contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan
from ..foundation.errors import WorkerError
from ..foundation.sizing import ceil_div
from .image_params import required_image_height, required_image_width

if TYPE_CHECKING:
    from .request_state import RequestStateTable
    from .residency import ResidencyManager
    from .resources import ResourceRuntime

__all__ = [
    "LeaseAccountingDelta",
    "ResidencyLeaseManager",
]


@dataclass
class LeaseAccountingDelta:
    """Per-step transient residency acquired by one op."""

    req_id: int
    acquired_latent: bool = False
    acquired_scratch: bool = False


class ResidencyLeaseManager:
    """Owns resource lease accounting and release policy for request steps."""

    def __init__(
        self,
        resource_runtime: "ResourceRuntime",
        request_states: "RequestStateTable",
        resource_plan: ResourcePlan,
        residency: "ResidencyManager | None" = None,
    ):
        self.resource_runtime = resource_runtime
        self.request_states = request_states
        self.resource_plan = resource_plan
        self.residency = residency

    def account_group(self, group: Sequence[tuple[int, Mapping[str, Any]]]) -> None:
        accounted: list[LeaseAccountingDelta] = []
        try:
            for _, op in group:
                accounted.append(self._account_op_resources(op))
        except (WorkerError, RuntimeError):
            self.rollback(accounted)
            raise

    def rollback(self, accounted: list[LeaseAccountingDelta]) -> None:
        for delta in reversed(accounted):
            req_id = delta.req_id
            state = self.request_states.get(req_id)
            if delta.acquired_scratch:
                self.release_class_if_managed("scratch", req_id)
                state.deactivate_scratch()
            if delta.acquired_latent:
                self.release_class_if_managed("image_latent", req_id)
                state.deactivate_image_latent()

    def account_blocks(
        self,
        req_id: int,
        block_ids: list[int] | tuple[int, ...],
        *,
        append_to_state: bool = True,
    ) -> None:
        self.acquire_blocks(req_id, block_ids, append_to_state=append_to_state)

    def acquire_blocks(
        self,
        req_id: int,
        block_ids: list[int] | tuple[int, ...],
        *,
        append_to_state: bool = True,
    ) -> None:
        state = self.request_states.get(req_id)
        if append_to_state and block_ids:
            ingest = getattr(self.request_states, "ingest_new_blocks", None)
            if callable(ingest):
                ingest(req_id, tuple(int(block_id) for block_id in block_ids))
            else:
                from .request_state import append_new_block_ids

                append_new_block_ids(state.block_ids, tuple(int(block_id) for block_id in block_ids))
        if "kv_block" not in self.resource_runtime.classes:
            return
        incoming = {int(block_id) for block_id in block_ids}
        new_blocks = incoming - state.resident_block_ids
        if new_blocks:
            self.resource_runtime.acquire("kv_block", req_id, len(new_blocks))
            state.resident_block_ids.update(new_blocks)

    def open_scratch_cache(self, cache: Any) -> Any:
        pool = getattr(cache, "pool", None)
        if self.residency is not None:
            self.residency.require_allocator_for_pool(pool)
        return cache

    def release_step(self, req_id: int, *, image_latent: bool = False, scratch: bool = False) -> None:
        state = self.request_states.get(req_id)
        if scratch:
            self.release_class_if_managed("scratch", req_id)
            state.deactivate_scratch()
        if image_latent:
            self.release_class_if_managed("image_latent", req_id)
            state.deactivate_image_latent()

    def release_generation(self, req_id: int, *, committed: bool) -> None:
        self.release_step(req_id, image_latent=True, scratch=True)
        finish = getattr(self.request_states, "finish_generation", None)
        if callable(finish):
            finish(req_id, committed=committed)
        elif committed:
            self.request_states.commit(req_id)
        else:
            self.request_states.abort(req_id)

    def release_request(self, req_id: int) -> int:
        rid = int(req_id)
        freed = self.resource_runtime.release_request(rid)
        if self.residency is not None:
            latent = getattr(self.residency, "latent", None)
            free = getattr(latent, "free", None)
            if callable(free):
                free(rid)
        return int(freed)

    def pressure(self) -> list[dict[str, Any]]:
        return self.resource_runtime.pressure()

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
        h = required_image_height(image)
        w = required_image_width(image)
        downsample = int(image.get("latent_downsample") or rule.downsample or 16)
        return ceil_div(max(1, h), downsample) * ceil_div(max(1, w), downsample)

    def release_class_if_managed(self, cls: str, req_id: int) -> None:
        if cls in self.resource_runtime.classes:
            self.resource_runtime.release_class(cls, req_id)

    def _account_op_resources(self, op: Mapping[str, Any]) -> LeaseAccountingDelta:
        req_id = int(op["req_id"])
        self.account_blocks(req_id, op.get("new_block_ids") or [])
        delta = LeaseAccountingDelta(req_id=req_id)
        if op.get("kind") != DENOISE_GEN:
            return delta
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

    def _scratch_units(self, op: Mapping[str, Any], rule: PerBranch) -> int:
        cfg = op.get("cfg")
        branch_count = 1
        if isinstance(cfg, Mapping):
            branch_count = int(cfg.get("branch_count") or 1)
        return max(int(rule.minimum), branch_count)

    def _acquire_if_managed(self, cls: str, req_id: int, units: int) -> None:
        if cls in self.resource_runtime.classes:
            self.resource_runtime.acquire(cls, req_id, max(1, int(units)))
