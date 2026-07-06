"""System-owned commit driver for generated interleaved images.

Owns the "vision-token commit" flow: re-normalize the finished pixels,
re-encode them through the owner's image feature extractor, append the image
tokens plus the image-end marker into the conditional (and, when images are
retained, text-unconditional) paged text caches, and finalize the commit
response (PNG payload, next-token logits, cache/scratch release). Under a
dataplane tower handoff the latent is instead published by locator and the
writeback happens on the text side via :meth:`GeneratedImageCommitDriver.commit_writeback`.

Separate from :mod:`.interleaved_image_denoise` because commit mechanics are
the part most likely to differ across otherwise-similar denoise models; the
denoise engine does not depend on this module.
"""
from __future__ import annotations

from typing import Any, Protocol

import torch

from ..foundation.errors import invalid_descriptor, model_execution_error
from ..nn.vision import build_abs_positions_from_grid_hw
from ..runtime.image_utils import tensor_to_png_b64
from ..runtime.masks import build_commit_attention_mask
from .interleaved_image_denoise import ImageState
from .interleaved_text_stepper import InterleavedModelOwner, TextCache

__all__ = [
    'GeneratedImageCommitOwner',
    'GeneratedImageCommitDriver',
]

# ImageNet channel statistics for re-normalizing a generated image before ViT
# re-encoding at commit. Private module constants until a model with different
# encoder statistics adopts this driver.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GeneratedImageCommitOwner(InterleavedModelOwner, Protocol):
    """Collaborator surface the commit driver needs beyond the base owner.

    Extends :class:`~uniserve_worker.execution.interleaved_text_stepper.InterleavedModelOwner`
    with image-parameter access plus the dataplane commit hooks. The dataplane
    hooks are exercised only when ``_dataplane_handoff`` is not ``None``;
    single-device owners may implement them as raising stubs.
    """

    latent_downsample: int
    residency: Any                 # ResidencyManager; .latent backs ImageState.x_t
    _dataplane_handoff: Any | None

    def _parse_image_params(self, ip: dict) -> Any: ...
    def publish_generated_latent_for_commit(self, image_state: Any) -> Any: ...
    def fetch_commit_latent(self, locator: Any) -> torch.Tensor: ...
    def encode_commit_locator(self, locator: Any) -> str: ...


class GeneratedImageCommitDriver:
    """Append a finished image into text caches and finalize the commit response."""

    def __init__(self, owner: GeneratedImageCommitOwner) -> None:
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
        if retain_images:
            self.owner._extend_cache_blocks(st.cond, op)
            self.owner._ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.tu.past
                )
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
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.tu.past
                )
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, None)

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
        self.owner.residency.release_scratch_cache(st.iu.past)
        st.iu = TextCache()
        out = {
            "req_id": op["req_id"],
            "image_hw": [image_state.height, image_state.width],
            "logits": logits,
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
