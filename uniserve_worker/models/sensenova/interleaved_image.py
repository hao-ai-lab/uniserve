"""Engine-owned helpers for interleaved text and image generation models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, Sequence

import torch

import uniserve_worker.ops as ops
from ...contracts.forward_context import get_forward_context
from ...execution.denoise_driver import TextImageDenoiseStep
from ...execution.interleaved_text_stepper import (
    InterleavedModelOwner,
    InterleavedTextCacheDriver,
    TextCache,
)
from ...foundation.errors import invalid_descriptor, model_execution_error
from ...nn.diffusion import FlowMatchSchedule, ScheduleDirection, ScheduleShiftDomain, init_latent
from ...nn.diffusion.cfg import Branch, CfgRecipe, build_text_image_cfg_plan
from ...runtime.masks import build_commit_attention_mask
from ...nn.vision import build_abs_positions_from_grid_hw, patchify_batch, unpatchify_batch
from ...runtime.image_params import required_image_height, required_image_width
from ...runtime.image_utils import tensor_to_png_b64
from ...runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache

__all__ = [
    'NoiseScaleMode',
    'TextCache',
    'ImageState',
    'DenoiseRow',
    'InterleavedImageRequestState',
    'InterleavedModelOwner',
    'InterleavedTextCacheDriver',
    'TextImageDenoiseOwner',
    'TextImageDenoiseOps',
    'GeneratedImageCommitDriver',
]

if TYPE_CHECKING:
    pass

# ImageNet channel statistics for re-normalizing a generated image before ViT re-encoding.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Resolution-aware noise scaling raises the per-token sequence-length ratio to
# this power (a square root: noise std scales with sqrt of the token count
# relative to the reference sequence length). ``dynamic_sqrt`` then applies the
# same square-root exponent a second time to further damp the resolution term.
_NOISE_RESOLUTION_EXPONENT = 0.5
_NOISE_DYNAMIC_SQRT_EXPONENT = 0.5


class NoiseScaleMode(str, Enum):
    """Initial-noise scaling convention.

    Values match ``model.noise_scale_mode`` string config. ``FIXED`` leaves
    ``noise_scale`` unchanged; resolution-aware modes scale it by token-count ratio.
    """

    FIXED = "fixed"
    RESOLUTION = "resolution"
    DYNAMIC = "dynamic"
    DYNAMIC_SQRT = "dynamic_sqrt"


# Resolution-aware modes apply sqrt sequence-length scaling; the others leave the
# base noise scale unchanged. Unknown string modes are treated like FIXED.
_NOISE_RESOLUTION_MODES = frozenset(
    {NoiseScaleMode.RESOLUTION, NoiseScaleMode.DYNAMIC, NoiseScaleMode.DYNAMIC_SQRT}
)


@dataclass
class ImageState:
    """Mutable denoise state for one in-flight image generation request.

    The latent trajectory ``x_t`` is not stored on the model state — it lives in
    the system-owned :class:`~uniserve_worker.runtime.residency.LatentPool`
    as a leased buffer addressed by ``latent_handle``. ``x_t`` here is a property
    reading/writing that system buffer.
    """

    latent_pool: Any           # system LatentPool (residency.latent)
    latent_handle: int         # request-scoped handle into the LatentPool
    schedule: FlowMatchSchedule
    timesteps: torch.Tensor
    token_h: int
    token_w: int
    grid_h: int
    grid_w: int
    grid_hw: torch.Tensor
    indexes_cond: torch.Tensor
    indexes_tu: torch.Tensor | None
    indexes_iu: torch.Tensor | None
    cond_cache: Any
    tu_cache: Any
    iu_cache: Any
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_interval: tuple[float, float]
    cfg_norm: str
    cfg_renorm_min: float
    noise_scale: float
    height: int
    width: int

    @property
    def x_t(self) -> torch.Tensor:
        return self.latent_pool.get(self.latent_handle)

    @x_t.setter
    def x_t(self, value: torch.Tensor) -> None:
        self.latent_pool.set(self.latent_handle, value)


@dataclass
class _ImageParams:
    """Parsed per-request image-generation parameters.

    Bundles the scalar fields read out of ``InterleavedImageRequestState.image``
    so the parse step can be separated from cache setup, index construction and
    latent initialization without threading a dozen locals between them.
    """

    width: int
    height: int
    steps: int
    cfg_text: float
    cfg_img: float
    cfg_interval: tuple[float, float]
    cfg_norm: str
    cfg_renorm_min: float
    timestep_shift: float
    retain_images: bool
    seed: int | None


def _image_param(ip: dict, key: str, default: Any) -> Any:
    value = ip.get(key, default)
    return default if value is None else value


def _required_image_param(ip: dict, key: str) -> Any:
    value = ip.get(key)
    if value is None:
        raise invalid_descriptor(f"image.{key} is required")
    return value


@dataclass
class DenoiseRow:
    """One CFG branch of one denoise step queued for batched velocity prediction."""

    step_index: int
    step: TextImageDenoiseStep
    branch: str
    img: ImageState
    indexes: torch.Tensor
    cache: PagedTextCache


@dataclass
class InterleavedImageRequestState:
    """Per-request interleaved text/image caches and generation state."""

    sampling: dict = field(default_factory=dict)
    image: dict = field(default_factory=dict)
    neg_token_ids: list[int] = field(default_factory=list)
    cond: TextCache = field(default_factory=TextCache)
    tu: TextCache = field(default_factory=TextCache)
    iu: TextCache = field(default_factory=TextCache)
    image_state: ImageState | None = None
    rng: torch.Generator | None = None


class TextImageDenoiseOwner(Protocol):
    """Collaborator surface a concrete model must provide to TextImageDenoiseOps.

    The mixin owns the denoise step/branch/commit flow but delegates model- and
    cache-specific work back to the concrete owner. Every member declared below
    is part of the mixin's contract and is called directly: the concrete owner
    must define all of them, including the ``_denoise`` / ``_wait`` cache and
    stream hooks.
    """

    # Collaborator attributes.
    model: Any
    device: Any
    gen_device: Any
    latent_downsample: int
    merge_size: int
    _img_start_token: str

    # Collaborator methods.
    def _state(self, op: dict[str, Any]) -> "InterleavedImageRequestState": ...
    def _maybe_stage_conditioning_from_op(
        self, st: "InterleavedImageRequestState", op: dict[str, Any]
    ) -> None: ...
    def _extend_cache_blocks(self, cache: "TextCache", op: dict[str, Any]) -> None: ...
    def _ensure_img_start(self, cache: "TextCache | None") -> None: ...
    def _prefix_from_query(self, query: str) -> "TextCache": ...
    def _empty_img_start_prefix(self) -> "TextCache": ...
    def _denoise_cache(self, cache: Any) -> Any: ...
    def _wait_gen_cache_ready(self, cache: Any) -> None: ...

    # Mixin methods (from TextImageDenoiseOps) reached through ``self``.
    def _init_image_state(
        self, st: "InterleavedImageRequestState", op: dict | None = ...
    ) -> "ImageState": ...
    def _parse_image_params(self, ip: dict) -> "_ImageParams": ...
    def _setup_cfg_caches(
        self, st: "InterleavedImageRequestState", op: dict | None, params: "_ImageParams"
    ) -> "TextCache": ...
    def _build_indexes(
        self,
        st: "InterleavedImageRequestState",
        cond: "TextCache",
        token_h: int,
        token_w: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...
    def _compute_noise_scale(self, grid_h: int, grid_w: int) -> float: ...
    def _init_latent(
        self,
        st: "InterleavedImageRequestState",
        params: "_ImageParams",
        device: Any,
        noise_scale: float,
    ) -> torch.Tensor: ...
    def predict_denoise_velocity(
        self, step: "TextImageDenoiseStep", branch: str
    ) -> torch.Tensor: ...
    def _denoise_branch_inputs(
        self, img: "ImageState", branch: str
    ) -> tuple[torch.Tensor, Any]: ...
    def _batched_denoise_row_key(self, row: "DenoiseRow") -> "tuple[Any, ...] | None": ...
    def _batched_paged_denoise_available(
        self, image_embeds: torch.Tensor, cache: "PagedTextCache"
    ) -> bool: ...
    def _predict_v_batched(self, rows: "Sequence[DenoiseRow]") -> torch.Tensor: ...
    def _predict_v(
        self,
        img: "ImageState",
        image_embeds: torch.Tensor,
        indexes: torch.Tensor | None,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor: ...


class TextImageDenoiseOps:
    """Mixin implementing text/image denoise setup, batching, and velocity prediction."""

    def _init_image_state(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        op: dict | None = None,
    ) -> ImageState:
        ip = st.image or {}
        params = self._parse_image_params(ip)
        cond = self._setup_cfg_caches(st, op, params)

        token_h = params.height // self.latent_downsample
        token_w = params.width // self.latent_downsample
        grid_h = params.height // self.model.patch_size
        grid_w = params.width // self.model.patch_size
        device = getattr(self, "gen_device", self.device)
        indexes_cond, indexes_tu, indexes_iu = self._build_indexes(
            st, cond, token_h, token_w, device
        )

        schedule = FlowMatchSchedule(
            num_steps=params.steps,
            shift=params.timestep_shift,
            direction=ScheduleDirection.ASCENDING,
            shift_domain=ScheduleShiftDomain.SIGMA,
        )
        timesteps = schedule.timesteps(device=device)
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        noise_scale = self._compute_noise_scale(grid_h, grid_w)

        x_t = self._init_latent(st, params, device, noise_scale)
        cond_cache = self._denoise_cache(cond.past)
        tu_cache = self._denoise_cache(st.tu.past)
        iu_cache = self._denoise_cache(st.iu.past)
        # The latent lives in the system-owned LatentPool, keyed by the request
        # handle; ``ImageState.x_t`` reads/writes that buffer.
        latent_handle = int(op["req_id"]) if op and "req_id" in op else id(st)
        image_state = ImageState(
            latent_pool=self.residency.latent,
            latent_handle=latent_handle,
            schedule=schedule,
            timesteps=timesteps,
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            grid_hw=grid_hw,
            indexes_cond=indexes_cond,
            indexes_tu=indexes_tu,
            indexes_iu=indexes_iu,
            cond_cache=cond_cache,
            tu_cache=tu_cache,
            iu_cache=iu_cache,
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            cfg_interval=(float(params.cfg_interval[0]), float(params.cfg_interval[1])),
            cfg_norm=params.cfg_norm,
            cfg_renorm_min=params.cfg_renorm_min,
            noise_scale=float(noise_scale),
            height=params.height,
            width=params.width,
        )
        image_state.x_t = x_t  # store the initial noise into the system LatentPool
        return image_state

    def _parse_image_params(self: TextImageDenoiseOwner, ip: dict) -> _ImageParams:
        steps = int(_required_image_param(ip, "steps"))
        if steps <= 0:
            raise invalid_descriptor("image.steps must be positive")
        cfg_interval = tuple(_required_image_param(ip, "cfg_interval"))
        if len(cfg_interval) != 2:
            raise invalid_descriptor("image.cfg_interval must contain exactly two values")
        return _ImageParams(
            width=required_image_width(ip),
            height=required_image_height(ip),
            steps=steps,
            cfg_text=float(_required_image_param(ip, "cfg_text_scale")),
            cfg_img=float(_required_image_param(ip, "cfg_img_scale")),
            cfg_interval=(float(cfg_interval[0]), float(cfg_interval[1])),
            cfg_norm=str(_required_image_param(ip, "cfg_renorm_type")),
            cfg_renorm_min=float(_required_image_param(ip, "cfg_renorm_min")),
            timestep_shift=float(_required_image_param(ip, "timestep_shift")),
            retain_images=bool(_image_param(ip, "retain_images", True)),
            seed=ip.get("seed"),
        )

    def _setup_cfg_caches(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        op: dict | None,
        params: _ImageParams,
    ) -> TextCache:
        """Prepare the cond/text-uncond/img-uncond text caches for denoising.

        Returns the conditioning :class:`TextCache` to use (a fresh image-prompt
        prefix when ``op`` supplies one, otherwise the request's own ``st.cond``).
        Mutates ``st.tu``/``st.iu`` in place as required by the CFG scales.
        """
        if params.retain_images:
            self._ensure_img_start(st.cond)
        cond = st.cond
        image_prompt = (op or {}).get("image_prompt")
        if isinstance(image_prompt, str) and image_prompt.strip():
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor(
                    "Mode A cuda_ipc tower split does not support per-op image_prompt overrides yet"
                )
            query = self.model._build_t2i_query(
                image_prompt.strip(),
                append_text=self._img_start_token,
            )
            cond = self._prefix_from_query(query)
        elif not params.retain_images:
            self._ensure_img_start(st.cond)
        cfg_plan = build_text_image_cfg_plan(
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            recipe=CfgRecipe.ADDITIVE_DELTAS,
            renorm=params.cfg_norm,
            renorm_min=params.cfg_renorm_min,
        )
        needs_text_uncond = Branch.TEXT_UNCOND in cfg_plan.branches
        if needs_text_uncond and st.tu.past is None:
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor("Mode A denoise is missing pre-staged text-unconditional CFG KV")
            st.tu = self._empty_img_start_prefix()
        elif needs_text_uncond:
            self._ensure_img_start(st.tu)
        needs_img_uncond = Branch.IMG_UNCOND in cfg_plan.branches
        if needs_img_uncond and st.iu.past is None:
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor("Mode A denoise is missing pre-staged image-unconditional CFG KV")
            st.iu = self._empty_img_start_prefix()
        elif needs_img_uncond:
            self._ensure_img_start(st.iu)
        return cond

    def _build_indexes(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        cond: TextCache,
        token_h: int,
        token_w: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        indexes_cond = self.model._build_t2i_image_indexes(
            token_h, token_w, cond.t_index + 1, device=device
        )
        indexes_tu = (
            self.model._build_t2i_image_indexes(token_h, token_w, st.tu.t_index + 1, device=device)
            if st.tu.past is not None
            else None
        )
        indexes_iu = (
            self.model._build_t2i_image_indexes(token_h, token_w, st.iu.t_index + 1, device=device)
            if st.iu.past is not None
            else None
        )
        return indexes_cond, indexes_tu, indexes_iu

    def _compute_noise_scale(self: TextImageDenoiseOwner, grid_h: int, grid_w: int) -> float:
        noise_scale = self.model.noise_scale
        mode = self.model.noise_scale_mode
        if mode in _NOISE_RESOLUTION_MODES:
            base = float(self.model.noise_scale_base_image_seq_len)
            seq_len_ratio = float(grid_h * grid_w) / (self.merge_size**2) / base
            noise_scale = seq_len_ratio**_NOISE_RESOLUTION_EXPONENT * float(noise_scale)
            if mode == NoiseScaleMode.DYNAMIC_SQRT:
                noise_scale = noise_scale**_NOISE_DYNAMIC_SQRT_EXPONENT
        return min(noise_scale, self.model.noise_scale_max_value)

    def _init_latent(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        params: _ImageParams,
        device: Any,
        noise_scale: float,
    ) -> torch.Tensor:
        if st.rng is None:
            seed = params.seed
            st.rng = torch.Generator(device=device).manual_seed(int(seed if seed is not None else 0))
        dtype = st.cond.last_logits.dtype
        return init_latent(
            (1, 3, params.height, params.width),
            rng=st.rng,
            device=device,
            dtype=dtype,
            scale=noise_scale,
        )

    def prepare_denoise_step(
        self: TextImageDenoiseOwner, req_id: int, state: Any, op: dict
    ) -> TextImageDenoiseStep:
        st = self._state(dict(op))
        # Mode A (tower disaggregation): the gen pool never ran the und text, so
        # rebuild st.cond from the conditioning KV the und pool published (carried
        # on op["locator"]). A no-op in Mode C / single-device.
        self._maybe_stage_conditioning_from_op(st, op)
        self._extend_cache_blocks(st.cond, op)
        if st.image_state is None:
            st.image_state = self._init_image_state(st, op)
        img = st.image_state
        step_i = int(op.get("timestep_idx") or 0)
        # Gen-tower feature extraction and timestep embedding run on the gen
        # coordinate's device (the gen modules are Pinned there); the tower
        # transport, not a dedicated stream, orders the und->gen handoff.
        device = getattr(self, "gen_device", self.device)
        t, t_next = img.schedule.pair(step_i, device=device, dtype=img.timesteps.dtype)
        z = patchify_batch(img.x_t, self.latent_downsample)
        image_input = patchify_batch(img.x_t, self.model.patch_size, channel_first=True)
        gen_vit = self.model.fm_modules["vision_model_mot_gen"]
        image_input = image_input.to(
            device=device,
            dtype=next(gen_vit.parameters()).dtype,
        )
        image_embeds = self.model.extract_feature(
            image_input.view(1 * img.grid_h * img.grid_w, -1),
            gen_model=True,
            grid_hw=img.grid_hw,
        ).view(1, img.token_h * img.token_w, -1)
        t_expanded = t.expand(img.token_h * img.token_w)
        timestep_embeddings = self.model.fm_modules["timestep_embedder"](t_expanded).view(
            1, img.token_h * img.token_w, -1
        )
        if self.model.add_noise_scale_embedding:
            ns = torch.full_like(t_expanded, img.noise_scale / self.model.noise_scale_max_value)
            timestep_embeddings += self.model.fm_modules["noise_scale_embedder"](ns).view(
                1, img.token_h * img.token_w, -1
            )
        image_embeds = image_embeds + timestep_embeddings
        total = int(img.schedule.num_steps)
        return TextImageDenoiseStep(
            req_id=int(req_id),
            state=state,
            op=op,
            latent=z,
            t=t,
            t_next=t_next,
            step_index=step_i,
            total_steps=total,
            cfg_text_scale=img.cfg_text_scale,
            cfg_img_scale=img.cfg_img_scale,
            cfg_interval=img.cfg_interval,
            cfg_renorm_type=img.cfg_norm,
            cfg_renorm_min=img.cfg_renorm_min,
            image_scale_applies_to_text=CfgRecipe.ADDITIVE_DELTAS,
            extra={
                "img": img,
                "image_embeds": image_embeds,
            },
        )

    def predict_denoise_velocity(
        self: TextImageDenoiseOwner, step: TextImageDenoiseStep, branch: str
    ) -> torch.Tensor:
        img = step.extra["img"]
        indexes, cache = self._denoise_branch_inputs(img, branch)
        return self._predict_v(
            img,
            step.extra["image_embeds"],
            indexes,
            cache,
            step.t,
            step.latent,
        )

    def predict_text_image_velocity_batch(
        self: TextImageDenoiseOwner,
        steps: Sequence[TextImageDenoiseStep],
        branches_by_step: Sequence[Sequence[str]],
    ) -> list[dict[str, torch.Tensor]]:
        results: list[dict[str, torch.Tensor]] = [dict() for _ in steps]
        rows: list[DenoiseRow] = []
        for step_index, (step, branches) in enumerate(zip(steps, branches_by_step)):
            img = step.extra["img"]
            for branch in branches:
                indexes, cache = self._denoise_branch_inputs(img, branch)
                if not isinstance(cache, PagedTextCache):
                    results[step_index][branch] = self.predict_denoise_velocity(step, branch)
                    continue
                rows.append(DenoiseRow(step_index, step, branch, img, indexes, cache))

        grouped: dict[tuple[Any, ...], list[DenoiseRow]] = {}
        for row in rows:
            key = self._batched_denoise_row_key(row)
            if key is None:
                results[row.step_index][row.branch] = self.predict_denoise_velocity(
                    row.step, row.branch
                )
                continue
            grouped.setdefault(key, []).append(row)

        for group in grouped.values():
            if len(group) == 1:
                row = group[0]
                results[row.step_index][row.branch] = self.predict_denoise_velocity(
                    row.step, row.branch
                )
                continue
            batched = self._predict_v_batched(group)
            for row_index, row in enumerate(group):
                results[row.step_index][row.branch] = batched[
                    row_index : row_index + 1
                ].contiguous()
        return results

    def _denoise_branch_inputs(self, img: ImageState, branch: str) -> tuple[torch.Tensor, Any]:
        # ``Branch`` is a ``str`` Enum, so this mapping resolves both ``Branch``
        # members and the equivalent bare strings ("cond"/"text_uncond"/
        # "img_uncond") to the same entry.
        branch_inputs: dict[Branch, tuple[torch.Tensor | None, Any]] = {
            Branch.COND: (img.indexes_cond, img.cond_cache),
            Branch.TEXT_UNCOND: (img.indexes_tu, img.tu_cache),
            Branch.IMG_UNCOND: (img.indexes_iu, img.iu_cache),
        }
        try:
            indexes, cache = branch_inputs[branch]  # type: ignore[index]
        except KeyError:
            raise model_execution_error(f"unknown denoise branch {branch!r}") from None
        if indexes is None or cache is None:
            raise model_execution_error("required CFG cache is not initialized")
        return indexes, cache

    def _batched_denoise_row_key(self, row: DenoiseRow) -> tuple[Any, ...] | None:
        step = row.step
        img = row.img
        indexes = row.indexes
        cache = row.cache
        pool = getattr(cache, "pool", None)
        image_embeds = step.extra["image_embeds"]
        if pool is None or indexes.ndim != 2 or indexes.shape[0] != 3:
            return None
        if not self._batched_paged_denoise_available(image_embeds, cache):
            return None
        if image_embeds.ndim != 3 or step.latent.ndim != 3:
            return None
        t_values = tuple(float(v) for v in step.t.detach().float().reshape(-1).cpu().tolist())
        return (
            id(pool),
            str(image_embeds.device),
            str(image_embeds.dtype),
            tuple(image_embeds.shape[1:]),
            str(step.latent.device),
            str(step.latent.dtype),
            tuple(step.latent.shape[1:]),
            tuple(indexes.shape),
            int(img.token_h),
            int(img.token_w),
            int(img.height),
            int(img.width),
            t_values,
        )

    def _batched_paged_denoise_available(
        self,
        image_embeds: torch.Tensor,
        cache: PagedTextCache,
    ) -> bool:
        if image_embeds.device.type != "cuda" or not image_embeds.is_cuda:
            return False
        pool = getattr(cache, "pool", None)
        block_size = int(getattr(pool, "block_size", 0) or 0)
        if pool is None or block_size <= 0:
            return False
        ctx = get_forward_context()
        preferred = ctx.attention_backend_name or getattr(self, "attention_backend", "auto")
        try:
            probe = image_embeds.new_empty((1, 1, 1, image_embeds.shape[-1]))
            return ops.can_run_attention(
                probe,
                probe,
                probe,
                regime=ops.AttentionRegime.DECODE,
                causal=True,
                scale=1.0,
                ctx=ctx,
                kv_cache=cache,
                block_table=image_embeds.new_empty((1, 1), dtype=torch.int32),
                cache_seqlens=image_embeds.new_empty((1,), dtype=torch.int32),
                override=preferred,
            )
        except Exception:
            return False

    def _predict_v_batched(
        self: TextImageDenoiseOwner,
        rows: Sequence[DenoiseRow],
    ) -> torch.Tensor:
        first = rows[0]
        img = first.img
        for row in rows:
            self._wait_gen_cache_ready(row.cache)
        image_embeds = torch.cat([row.step.extra["image_embeds"] for row in rows], dim=0)
        indexes = torch.stack([row.indexes for row in rows], dim=1).contiguous()
        cache = BatchedPagedTextCache([row.cache for row in rows])
        z = torch.cat([row.step.latent for row in rows], dim=0)
        return self._predict_v(
            img,
            image_embeds,
            indexes,
            cache,
            first.step.t,
            z,
        )

    def apply_denoise_update(
        self: TextImageDenoiseOwner, step: TextImageDenoiseStep, latent: torch.Tensor
    ) -> None:
        img = step.extra["img"]
        img.x_t = unpatchify_batch(
            latent,
            self.latent_downsample,
            height=img.height,
            width=img.width,
        )

    def _predict_v(
        self: TextImageDenoiseOwner,
        img: ImageState,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor | None,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if indexes is None or cache is None:
            raise model_execution_error("required CFG cache is not initialized")
        # B2: wait the snapshot's readiness before the gen tower reads the replica.
        self._wait_gen_cache_ready(cache)
        return self.model._t2i_predict_v(
            image_embeds,
            indexes,
            {"full_attention": None},
            cache,
            t,
            z,
            image_token_num=img.token_h * img.token_w,
            image_size=(img.width, img.height),
        )


class GeneratedImageCommitDriver:
    """Append a finished image into text caches and finalize the commit response."""

    def __init__(self, owner: InterleavedModelOwner) -> None:
        self.owner = owner

    def append_generated_image(self, cache: TextCache, image_state: Any) -> None:
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
        patch_size = self.owner.model.patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        flattened = (
            und_img[0]
            .view(channels, grid_h, patch_size, grid_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch_size**2)
        )
        vit_embeds = self.owner.model.extract_feature(
            flattened,
            grid_hw=image_state.grid_hw[:1].to(self.owner.device),
        ).unsqueeze(0)
        img_end = torch.tensor([[self.owner.img_end_id]], dtype=torch.long, device=self.owner.device)
        img_end_embed = self.owner.model.language_model.get_input_embeddings()(img_end)
        embeds = torch.cat([vit_embeds, img_end_embed], dim=1)
        num_image_tokens = vit_embeds.shape[1]

        abs_w, abs_h = build_abs_positions_from_grid_hw(
            image_state.grid_hw[:1] // int(1 / self.owner.model.downsample_ratio),
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
        outputs = self.owner.model.language_model(
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

    def commit_generated_image(self, op: dict[str, Any]) -> dict[str, Any]:
        st = self.owner._state(op)
        image_state = st.image_state
        if image_state is None:
            return {"req_id": op["req_id"]}
        png_b64 = tensor_to_png_b64(image_state.x_t)
        if getattr(self.owner, "_dataplane_handoff", None) is not None:
            locator = self.owner.publish_generated_latent_for_commit(image_state)
            st.image_state = None
            self.owner._release_image_state_caches(image_state)
            self.owner._release_scratch_cache(st.cond.past)
            self.owner._release_scratch_cache(st.tu.past)
            self.owner._release_scratch_cache(st.iu.past)
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
        if retain_images:
            self.owner._extend_cache_blocks(st.cond, op)
            self.owner._ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner._allocator_for_cache(st.cond.past)
            self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner._allocator_for_cache(st.tu.past)
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, png_b64)

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
        if retain_images:
            self.owner._extend_cache_blocks(st.cond, op)
            self.owner._ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner._allocator_for_cache(st.cond.past)
            self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner._allocator_for_cache(st.tu.past)
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, None)

    def _writeback_image_state(self, st: Any, op: dict[str, Any], latent: torch.Tensor) -> ImageState:
        ip = st.image or {}
        params = self.owner._parse_image_params(ip)
        height = int(params.height)
        width = int(params.width)
        token_h = height // self.owner.latent_downsample
        token_w = width // self.owner.latent_downsample
        grid_h = height // self.owner.model.patch_size
        grid_w = width // self.owner.model.patch_size
        device = self.owner.device
        latent_handle = int(op["req_id"])
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        self.owner.residency.latent.set(latent_handle, latent)
        return ImageState(
            latent_pool=self.owner.residency.latent,
            latent_handle=latent_handle,
            schedule=None,
            timesteps=torch.empty(0, device=device),
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            grid_hw=grid_hw,
            indexes_cond=None,
            indexes_tu=None,
            indexes_iu=None,
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
    ) -> dict[str, Any]:
        logits = st.cond.last_logits[:, -1, :].float()
        st.image_state = None
        self.owner._release_image_state_caches(image_state)
        self.owner._release_scratch_cache(st.iu.past)
        st.iu = TextCache()
        out = {
            "req_id": op["req_id"],
            "image_hw": [image_state.height, image_state.width],
            "logits": logits,
        }
        if png_b64 is not None:
            out["image_png_b64"] = png_b64
        return out
