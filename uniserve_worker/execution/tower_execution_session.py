"""Tower execution serving-session orchestration."""
from __future__ import annotations

from typing import Any

import torch

from ..foundation.errors import invalid_descriptor
from ..nn.diffusion.cfg import Branch, CfgRecipe, build_text_image_cfg_plan
from ..runtime.tower_handoff import ConditioningSnapshot, DataPlaneTowerHandoff
from ..runtime.transfer import Locator

__all__ = [
    "TowerExecutionSession",
]


class TowerExecutionSession:
    """Owns tower conditioning and mixed-forward orchestration hooks."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def bind_data_plane_handoff(self, transport: Any) -> None:
        self.owner._dataplane_handoff = DataPlaneTowerHandoff(
            data_plane=transport,
            bind=self.owner._resolve_tower_binding,
        )

    def wait_gen_cache_ready(self, cache: Any) -> None:
        self.owner._tower_handoff.await_ready(cache)

    def publish_conditioning(self, req_id: int, sampled_token_id: int) -> Any:
        handoff = self.owner._dataplane_handoff
        if handoff is None:
            return None
        if int(sampled_token_id) != int(self.owner.img_start_id):
            return None
        state = self.owner.reqs.get(int(req_id))
        if state is None or state.cond.past is None:
            return None
        self.owner._ensure_img_start(state.cond)
        params = self.owner._parse_image_params(state.image or {})
        cfg_plan = build_text_image_cfg_plan(
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            recipe=CfgRecipe.ADDITIVE_DELTAS,
            renorm=params.cfg_norm,
            renorm_min=params.cfg_renorm_min,
        )
        needs_text_uncond = Branch.TEXT_UNCOND in cfg_plan.branches
        if needs_text_uncond:
            if state.tu.past is None:
                state.tu = self.owner._empty_img_start_prefix()
            else:
                self.owner._ensure_img_start(state.tu)
        needs_img_uncond = Branch.IMG_UNCOND in cfg_plan.branches
        if needs_img_uncond:
            if state.iu.past is None:
                state.iu = self.owner._empty_img_start_prefix()
            else:
                self.owner._ensure_img_start(state.iu)
        snapshot = handoff.publish_conditioning(
            state.cond.past,
            t_index=int(state.cond.t_index),
            last_token_id=state.cond.last_token_id,
            tu_cache=state.tu.past if needs_text_uncond else None,
            tu_t_index=int(state.tu.t_index),
            tu_last_token_id=state.tu.last_token_id,
            iu_cache=state.iu.past if needs_img_uncond else None,
            iu_t_index=int(state.iu.t_index),
            iu_last_token_id=state.iu.last_token_id,
        )
        return snapshot.to_wire() if snapshot is not None else None

    def fetch_conditioning(self, locator: Any) -> Any:
        handoff = self.owner._dataplane_handoff
        if handoff is not None and hasattr(handoff, "fetch_conditioning"):
            return handoff.fetch_conditioning(locator)
        return None

    def stage_text_cache_from_snapshot(
        self,
        target: Any,
        snapshot: ConditioningSnapshot,
        *,
        locators: tuple[Any, ...],
        length: int,
        t_index: int,
        last_token_id: int | None,
    ) -> None:
        handoff = self.owner._dataplane_handoff
        if handoff is None or not locators:
            return
        branch = ConditioningSnapshot(
            locators=tuple(locators),
            length=int(length),
            num_layers=int(snapshot.num_layers),
            t_index=int(t_index),
            last_token_id=last_token_id,
        )
        replica = handoff.stage_conditioning(branch)
        if replica is None:
            return
        target.past = replica
        target.past.allocate_blocks = self.owner.residency.allocator_for_cache(replica)
        target.block_ids = list(replica.block_ids)
        target.t_index = int(t_index)
        target.last_token_id = (
            int(last_token_id) if last_token_id is not None else int(self.owner.img_start_id)
        )

    def stage_conditioning_from_op(self, state: Any, op: dict[str, Any]) -> None:
        handoff = self.owner._dataplane_handoff
        if handoff is None:
            return
        locator = op.get("locator") if isinstance(op, dict) else None
        if not locator:
            return
        snapshot = ConditioningSnapshot.from_wire(locator)
        for cache in (state.cond, state.tu, state.iu):
            self.owner.residency.release_scratch_cache(getattr(cache, "past", None))
        text_cache_cls = type(state.cond)
        state.cond = text_cache_cls()
        state.tu = text_cache_cls()
        state.iu = text_cache_cls()
        replica = handoff.stage_conditioning(snapshot)
        if replica is None:
            return
        state.cond.past = replica
        state.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(replica)
        state.cond.block_ids = list(replica.block_ids)
        state.cond.t_index = int(snapshot.t_index)
        state.cond.last_token_id = (
            int(snapshot.last_token_id)
            if snapshot.last_token_id is not None
            else int(self.owner.img_start_id)
        )
        state.cond.last_logits = torch.zeros(
            1,
            dtype=next(self.owner.model.parameters()).dtype,
            device=self.owner.gen_device,
        )
        self.stage_text_cache_from_snapshot(
            state.tu,
            snapshot,
            locators=snapshot.tu_locators,
            length=snapshot.tu_length,
            t_index=snapshot.tu_t_index,
            last_token_id=snapshot.tu_last_token_id,
        )
        self.stage_text_cache_from_snapshot(
            state.iu,
            snapshot,
            locators=snapshot.iu_locators,
            length=snapshot.iu_length,
            t_index=snapshot.iu_t_index,
            last_token_id=snapshot.iu_last_token_id,
        )

    def denoise_cache(self, cache: Any) -> Any:
        return self.owner._tower_handoff.stage_conditioning(cache)

    def prepare_commit_latent(self, image_state: Any) -> torch.Tensor:
        if self.owner._dataplane_handoff is not None:
            return image_state.x_t[0].unsqueeze(0).to(
                device=self.owner.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
        return self.owner._tower_handoff.writeback_commit(
            image_state.x_t[0].unsqueeze(0),
            device=self.owner.device,
            dtype=torch.bfloat16,
        )

    def publish_commit_latent(self, image_state: Any) -> Any:
        if self.owner._dataplane_handoff is None:
            return self.prepare_commit_latent(image_state)
        return self.owner._dataplane_handoff.publish_commit_latent(image_state.x_t[0].unsqueeze(0))

    def fetch_commit_latent(self, locator: Any) -> torch.Tensor:
        if self.owner._dataplane_handoff is None:
            raise RuntimeError("commit_writeback requires a data-plane handoff")
        if isinstance(locator, str):
            locator = Locator.from_wire_json(locator)
        if not isinstance(locator, Locator):
            raise invalid_descriptor("commit_writeback locator must be a typed data-plane Locator")
        latent = self.owner._dataplane_handoff.data_plane.fetch(locator)
        return latent.to(device=self.owner.device, dtype=torch.bfloat16, non_blocking=True)

    @staticmethod
    def encode_commit_locator(locator: Any) -> str:
        if not isinstance(locator, Locator):
            raise invalid_descriptor("commit locator must be a typed data-plane Locator")
        return locator.to_wire_json()

    def stage_cache(self, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(self.owner, "_stage_text_cache_for_forward", None)
        if callable(stage):
            return stage(*args, **kwargs)
        raise RuntimeError("tower session owner does not expose cache staging")

    def run_mixed_forward(self, *args: Any, **kwargs: Any) -> Any:
        run = getattr(self.owner, "_run_forward_adapter", None)
        if callable(run):
            return run(*args, **kwargs)
        raise RuntimeError("tower session owner does not expose mixed forward")

    def commit_handoff(self, *args: Any, **kwargs: Any) -> Any:
        commit = getattr(self.owner, "commit_handoff", None)
        if callable(commit):
            return commit(*args, **kwargs)
        handoff = getattr(self.owner, "_dataplane_handoff", None)
        if handoff is not None and hasattr(handoff, "commit_handoff"):
            return handoff.commit_handoff(*args, **kwargs)
        return None
