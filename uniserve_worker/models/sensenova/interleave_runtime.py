"""SenseNova interleave runtime: commit, input-image ingest, tower handoff."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from uniserve_worker.foundation.errors import invalid_descriptor, model_execution_error
from uniserve_worker.models.interleaved_image import ImageState
from uniserve_worker.models.interleaved_text import InterleavedModelOwner, TextCache
from uniserve_worker.nn.diffusion import FlowMatchSchedule
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_text_image_cfg_plan
from uniserve_worker.nn.vision import build_abs_positions_from_grid_hw
from uniserve_worker.runtime.image_utils import tensor_to_png_b64
from uniserve_worker.runtime.masks import build_commit_attention_mask
from uniserve_worker.runtime.tower_handoff import ConditioningSnapshot, DataPlaneTowerHandoff
from uniserve_worker.runtime.transfer import Locator

# ---------------------
# Generated-image commit
# ---------------------

# ImageNet channel statistics for re-normalizing a generated image before ViT
# re-encoding at commit. Private module constants until a model with different
# encoder statistics adopts this driver.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GeneratedImageCommitOwner(InterleavedModelOwner, Protocol):
    """Collaborator surface the commit driver needs beyond the base owner.

    Extends :class:`~uniserve_worker.models.interleaved_text.InterleavedModelOwner`
    with image-parameter access plus the dataplane commit hooks. The dataplane
    hooks are exercised only when ``_dataplane_handoff`` is not ``None``;
    single-device owners may implement them as raising stubs.
    """

    latent_downsample: int
    residency: Any                 # ResidencyManager; .latent backs ImageState.x_t
    _dataplane_handoff: Any | None
    img_end_id: int
    denoise_schedule_direction: Any
    denoise_schedule_shift_domain: Any

    def _parse_image_params(self, ip: dict) -> Any: ...
    def _state(self, op: dict[str, Any]) -> Any: ...
    def _extend_cache_blocks(self, cache: TextCache, op: dict[str, Any]) -> None: ...
    def _ensure_host_cache(self, cache: TextCache) -> None: ...
    def _release_image_state_caches(self, image_state: Any) -> None: ...
    def _prepare_generated_image_for_commit(self, image_state: Any) -> torch.Tensor: ...
    def interleaved_image_patch_size(self) -> int: ...
    def interleaved_image_downsample_ratio(self) -> float: ...
    def interleaved_image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor: ...
    def interleaved_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: Any,
    ) -> torch.Tensor: ...
    def publish_generated_latent_for_commit(self, image_state: Any) -> Any: ...
    def fetch_commit_latent(self, locator: Any) -> torch.Tensor: ...
    def encode_commit_locator(self, locator: Any) -> str: ...


class GeneratedImageCommitDriver:
    """Append a finished image into text caches and finalize the commit response."""

    def __init__(self, owner: GeneratedImageCommitOwner) -> None:
        self.owner = owner

    def append_generated_image(self, cache: TextCache, image_state: Any) -> int:
        if cache.past is None:
            raise model_execution_error("cannot append generated image without an initialized text cache")
        pred_img = self.owner._prepare_generated_image_for_commit(image_state)
        raw_img = pred_img * 0.5 + 0.5
        mean = torch.tensor(_IMAGENET_MEAN, dtype=raw_img.dtype, device=self.owner.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(_IMAGENET_STD, dtype=raw_img.dtype, device=self.owner.device).view(
            1, 3, 1, 1
        )
        und_img = (raw_img - mean) / std
        channels, height, width = und_img[0].shape
        patch_size = self.owner.interleaved_image_patch_size()
        grid_h = height // patch_size
        grid_w = width // patch_size
        flattened = (
            und_img[0]
            .view(channels, grid_h, patch_size, grid_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch_size**2)
        )
        vit_embeds = self.owner.interleaved_image_features(
            flattened,
            grid_hw=image_state.grid_hw[:1].to(self.owner.device),
        ).unsqueeze(0)
        img_end = torch.tensor([[self.owner.img_end_id]], dtype=torch.long, device=self.owner.device)
        img_end_embed = self.owner.interleaved_text_embeddings(img_end)
        embeds = torch.cat([vit_embeds, img_end_embed], dim=1)
        num_image_tokens = vit_embeds.shape[1]

        abs_w, abs_h = build_abs_positions_from_grid_hw(
            image_state.grid_hw[:1] // int(1 / self.owner.interleaved_image_downsample_ratio()),
            device=self.owner.device,
        )
        past_len = cache.past.get_seq_length()
        target_len = num_image_tokens + 1
        t_indexes = torch.zeros(target_len, dtype=torch.long, device=self.owner.device)
        t_indexes[:num_image_tokens] = cache.t_index + 1
        t_indexes[num_image_tokens] = cache.t_index + 2
        h_indexes = torch.zeros(target_len, dtype=torch.long, device=self.owner.device)
        w_indexes = torch.zeros(target_len, dtype=torch.long, device=self.owner.device)
        h_indexes[:num_image_tokens] = abs_h
        w_indexes[:num_image_tokens] = abs_w
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
        mask = build_commit_attention_mask(
            num_image_tokens=num_image_tokens,
            past_len=past_len,
            device=self.owner.device,
        )
        outputs = self.owner.interleaved_text_forward(
            inputs_embeds=embeds,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index += 2
        cache.last_logits = outputs.logits
        cache.last_token_id = int(self.owner.img_end_id)
        return int(target_len)

    def commit_generated_image(self, op: dict[str, Any]) -> dict[str, Any]:
        st = self.owner._state(op)
        image_state = st.image_state
        if image_state is None:
            return {"req_id": op["req_id"]}
        png_b64 = (
            tensor_to_png_b64(image_state.x_t)
            if _tp_rank(self.owner) == 0
            else None
        )
        if getattr(self.owner, "_dataplane_handoff", None) is not None:
            locator = self.owner.publish_generated_latent_for_commit(image_state)
            st.image_state = None
            self.owner._release_image_state_caches(image_state)
            self.owner.residency.release_scratch_cache(st.cond.past)
            self.owner.residency.release_scratch_cache(st.tu.past)
            self.owner.residency.release_scratch_cache(st.iu.past)
            st.cond = TextCache()
            st.tu = TextCache()
            st.iu = TextCache()
            return {
                "req_id": op["req_id"],
                "image_png_b64": png_b64,
                "image_hw": [image_state.height, image_state.width],
                "locator": self.owner.encode_commit_locator(locator),
            }
        retain_images = bool(st.image.get("retain_images", True))
        num_tokens = 0
        if retain_images:
            self.owner._extend_cache_blocks(st.cond, op)
            self.owner._ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            num_tokens = self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.tu.past
                )
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, png_b64, num_tokens)

    def commit_writeback(self, op: dict[str, Any]) -> dict[str, Any]:
        st = self.owner._state(op)
        image_state = st.image_state
        locator = op.get("locator")
        if not locator:
            raise invalid_descriptor("commit_writeback requires a commit latent locator")
        latent = self.owner.fetch_commit_latent(locator)
        if image_state is None:
            image_state = self._writeback_image_state(st, op, latent)
            st.image_state = image_state
        image_state.x_t = latent
        retain_images = bool(st.image.get("retain_images", True))
        num_tokens = 0
        if retain_images:
            self.owner._extend_cache_blocks(st.cond, op)
            self.owner._ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            num_tokens = self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.tu.past
                )
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, None, num_tokens)

    def _writeback_image_state(self, st: Any, op: dict[str, Any], latent: torch.Tensor) -> ImageState:
        ip = st.image or {}
        params = self.owner._parse_image_params(ip)
        height = int(params.height)
        width = int(params.width)
        token_h = height // self.owner.latent_downsample
        token_w = width // self.owner.latent_downsample
        patch_size = self.owner.interleaved_image_patch_size()
        grid_h = height // patch_size
        grid_w = width // patch_size
        device = self.owner.device
        latent_handle = int(op["req_id"])
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        schedule = FlowMatchSchedule(
            num_steps=int(params.steps),
            shift=float(params.timestep_shift),
            direction=self.owner.denoise_schedule_direction,
            shift_domain=self.owner.denoise_schedule_shift_domain,
        )
        indexes_cond = self.owner.interleaved_image_indexes(
            token_h,
            token_w,
            st.cond.t_index + 1,
            device=device,
        )
        indexes_tu = (
            self.owner.interleaved_image_indexes(
                token_h,
                token_w,
                st.tu.t_index + 1,
                device=device,
            )
            if st.tu.past is not None
            else None
        )
        indexes_iu = (
            self.owner.interleaved_image_indexes(
                token_h,
                token_w,
                st.iu.t_index + 1,
                device=device,
            )
            if st.iu.past is not None
            else None
        )
        self.owner.residency.latent.set(latent_handle, latent)
        return ImageState(
            latent_pool=self.owner.residency.latent,
            latent_handle=latent_handle,
            schedule=schedule,
            timesteps=schedule.timesteps(device=device),
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            grid_hw=grid_hw,
            indexes_cond=indexes_cond,
            indexes_tu=indexes_tu,
            indexes_iu=indexes_iu,
            cond_cache=st.cond.past,
            tu_cache=st.tu.past,
            iu_cache=st.iu.past,
            cfg_text_scale=float(params.cfg_text),
            cfg_img_scale=float(params.cfg_img),
            cfg_interval=(float(params.cfg_interval[0]), float(params.cfg_interval[1])),
            cfg_norm=params.cfg_norm,
            cfg_renorm_min=float(params.cfg_renorm_min),
            noise_scale=0.0,
            height=height,
            width=width,
        )

    def _finalize_commit(
        self,
        op: dict[str, Any],
        st: Any,
        image_state: Any,
        png_b64: str | None,
        num_tokens: int,
    ) -> dict[str, Any]:
        logits = st.cond.last_logits[:, -1, :].float()
        st.image_state = None
        self.owner._release_image_state_caches(image_state)
        self.owner.residency.release_scratch_cache(st.iu.past)
        st.iu = TextCache()
        out = {
            "req_id": op["req_id"],
            "image_hw": [image_state.height, image_state.width],
            "logits": logits,
            "num_tokens": int(num_tokens),
        }
        if png_b64 is not None:
            out["image_png_b64"] = png_b64
        return out


def _tp_rank(owner: Any) -> int:
    mesh = getattr(owner, "mesh", None)
    try:
        return int(getattr(mesh, "tp_rank", 0))
    except (TypeError, ValueError):
        return 0


# ---------------------
# Input-image ingest
# ---------------------

class InputImageIngestOwner(Protocol):
    """Collaborator surface for ingesting understanding images."""

    device: Any

    def interleaved_text_forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        text_only_rope: bool = False,
        causal_paged_update: bool = False,
    ) -> Any: ...
    def interleaved_image_features(
        self, image_input: torch.Tensor, *, grid_hw: torch.Tensor, gen_model: bool = ...
    ) -> torch.Tensor: ...
    def interleaved_image_downsample_ratio(self) -> float: ...


class InputImageIngestDriver:
    """Append an external image's vision tokens into a paged text cache."""

    def __init__(self, owner: InputImageIngestOwner) -> None:
        self.owner = owner

    def ingest_understanding_image(
        self,
        cache: TextCache,
        flattened_patches: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        t_index: int,
    ) -> int:
        """Encode + append one image's patch block at temporal index ``t_index``.

        Returns the number of vision tokens appended. The patch block is
        bidirectional within itself (all patches share ``t_index``) and attends
        to the whole existing prefix, matching the block-causal semantics the
        reference pipeline builds from its expanded placeholder stream.
        """
        vit_embeds = self.encode_understanding_image(flattened_patches, grid_hw)
        return self.ingest_understanding_embeddings(
            cache,
            vit_embeds,
            grid_hw,
            t_index=t_index,
        )

    def encode_understanding_image(
        self,
        flattened_patches: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the reusable vision-encoder output for one image."""
        owner = self.owner
        return owner.interleaved_image_features(
            flattened_patches.to(owner.device),
            grid_hw=grid_hw.to(owner.device),
        )

    def ingest_understanding_embeddings(
        self,
        cache: TextCache,
        vit_embeds: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        t_index: int,
    ) -> int:
        """Append reusable vision embeddings into one request's paged text cache."""
        owner = self.owner
        if cache.past is None:
            raise model_execution_error(
                "input-image ingest requires an initialized paged text cache"
            )
        device = owner.device
        vit_embeds = vit_embeds.to(device).unsqueeze(0)
        num_tokens = int(vit_embeds.shape[1])

        merge = int(1 / owner.interleaved_image_downsample_ratio())
        abs_w, abs_h = build_abs_positions_from_grid_hw(
            grid_hw[:1].to(device) // merge, device=device
        )
        t_indexes = torch.full((num_tokens,), int(t_index), dtype=torch.long, device=device)
        indexes = torch.stack(
            [t_indexes, abs_h.to(torch.long), abs_w.to(torch.long)], dim=0
        )

        past_len = cache.past.get_seq_length()
        # Patches attend to the full prefix and to each other: an all-zeros
        # additive mask over [num_tokens, past + num_tokens].
        mask = torch.zeros(1, 1, num_tokens, past_len + num_tokens, device=device)

        outputs = owner.interleaved_text_forward(
            inputs_embeds=vit_embeds,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(t_index)
        cache.last_logits = outputs.logits
        return num_tokens


# ---------------------
# Tower execution session
# ---------------------

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
