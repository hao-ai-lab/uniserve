"""System-owned product encoding, materialization, and tower transfer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from uniserve_worker.contracts.forward_batch import EncodeRow
from uniserve_worker.contracts.forward_context import get_forward_context
from uniserve_worker.contracts.op_kinds import COMMIT_WRITEBACK, VIT_ENCODE
from uniserve_worker.execution.flow import FlowState
from uniserve_worker.execution.sequence import SequenceAdapter, SequenceCache, SequenceExecutor
from uniserve_worker.foundation.errors import invalid_descriptor, model_execution_error
from uniserve_worker.nn.diffusion import FlowMatchSchedule
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_flow_cfg_plan
from uniserve_worker.runtime.image_params import parse_text_image_generation_params
from uniserve_worker.runtime.image_utils import (
    pil_image_to_png_bytes,
    png_bytes_to_b64,
    tensor_to_png_bytes,
)
from uniserve_worker.runtime.masks import build_commit_attention_mask
from uniserve_worker.runtime.tower_handoff import (
    ConditioningSnapshot,
    DataPlaneTowerHandoff,
    LocalP2PTowerHandoff,
    TowerHandoff,
)
from uniserve_worker.runtime.transfer import Locator

# ---------------------
# Generated-image commit
# ---------------------

class ImageMaterializeAdapter(SequenceAdapter, Protocol):
    """Neural and geometry boundary required to materialize an image product.

    Extends :class:`~uniserve_worker.execution.sequence.SequenceAdapter` with
    the family's image geometry, pure tensor-space normalization, and neural
    feature extraction. Orchestration (cache staging, residency, product
    persistence) is owned by :class:`ImageMaterializer`.
    """

    latent_downsample: int
    img_end_id: int
    denoise_schedule_direction: Any
    denoise_schedule_shift_domain: Any

    def normalize_materialized_image(self, image: torch.Tensor) -> torch.Tensor: ...
    def sequence_position_indexes(
        self,
        grid_hw: torch.Tensor,
        temporal_indexes: torch.Tensor,
    ) -> torch.Tensor: ...
    def image_patch_size(self) -> int: ...
    def image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor: ...
    def flow_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: Any,
    ) -> torch.Tensor: ...
class ImageMaterializer:
    """Append a finished image into text caches and finalize the commit response."""

    def __init__(
        self,
        owner: ImageMaterializeAdapter,
        transfer: "ProductTransferSession",
        sequences: SequenceExecutor,
    ) -> None:
        self.owner = owner
        self.transfer = transfer
        self.sequences = sequences

    def commit(self, op: dict[str, Any], state: Any = None) -> dict[str, Any]:
        del state
        if op.get("kind") == COMMIT_WRITEBACK:
            return self.commit_writeback(op)
        return self.commit_generated_image(op)

    def append_generated_image(self, cache: SequenceCache, image_state: Any) -> int:
        if cache.past is None:
            raise model_execution_error(
                "cannot append generated image without an initialized text cache"
            )
        pred_img = self.transfer.prepare_commit_latent(image_state)
        und_img = self.owner.normalize_materialized_image(pred_img)
        channels, height, width = und_img[0].shape
        patch_size = self.owner.image_patch_size()
        grid_h = height // patch_size
        grid_w = width // patch_size
        flattened = (
            und_img[0]
            .view(channels, grid_h, patch_size, grid_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch_size**2)
        )
        vit_embeds = self.owner.image_features(
            flattened,
            grid_hw=image_state.grid_hw[:1].to(self.owner.device),
        ).unsqueeze(0)
        img_end = torch.tensor(
            [[self.owner.img_end_id]], dtype=torch.long, device=self.owner.device
        )
        img_end_embed = self.owner.sequence_embeddings(img_end)
        embeds = torch.cat([vit_embeds, img_end_embed], dim=1)
        num_image_tokens = vit_embeds.shape[1]

        past_len = cache.past.get_seq_length()
        target_len = num_image_tokens + 1
        t_indexes = torch.zeros(target_len, dtype=torch.long, device=self.owner.device)
        t_indexes[:num_image_tokens] = cache.t_index + 1
        t_indexes[num_image_tokens] = cache.t_index + 2
        indexes = self.owner.sequence_position_indexes(image_state.grid_hw[:1], t_indexes)
        mask = build_commit_attention_mask(
            num_image_tokens=num_image_tokens,
            past_len=past_len,
            device=self.owner.device,
        )
        outputs = self.owner.sequence_forward(
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
        st = self.sequences.state(op)
        image_state = st.image_state
        if image_state is None:
            return {"req_id": op["req_id"]}
        png = tensor_to_png_bytes(image_state.x_t) if _tp_rank(self.owner) == 0 else None
        if self.transfer.distributed:
            locator = self.transfer.publish_commit_latent(image_state)
            st.image_state = None
            latent_view = get_forward_context().latent_view
            if latent_view is None:
                raise invalid_descriptor("image commit requires an executor latent view")
            latent_view.release_state(image_state)
            self.owner.residency.release_scratch_cache(st.cond.past)
            self.owner.residency.release_scratch_cache(st.tu.past)
            self.owner.residency.release_scratch_cache(st.iu.past)
            st.cond = SequenceCache()
            st.tu = SequenceCache()
            st.iu = SequenceCache()
            self._store_frame(int(op["req_id"]), png)
            return {
                "req_id": op["req_id"],
                "image_png_b64": png_bytes_to_b64(png) if png is not None else None,
                "image_hw": [image_state.height, image_state.width],
                "locator": self.transfer.encode_commit_locator(locator),
            }
        retain_images = bool(st.image.get("retain_images", True))
        num_tokens = 0
        if retain_images:
            self.sequences.extend_cache_blocks(st.cond, op)
            self.sequences.ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            num_tokens = self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(st.tu.past)
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, png, num_tokens)

    def commit_writeback(self, op: dict[str, Any]) -> dict[str, Any]:
        st = self.sequences.state(op)
        image_state = st.image_state
        locator = op.get("locator")
        if not locator:
            raise invalid_descriptor("commit_writeback requires a commit latent locator")
        latent = self.transfer.fetch_commit_latent(locator)
        if image_state is None:
            image_state = self._writeback_image_state(st, op, latent)
            st.image_state = image_state
        image_state.x_t = latent
        retain_images = bool(st.image.get("retain_images", True))
        num_tokens = 0
        if retain_images:
            self.sequences.extend_cache_blocks(st.cond, op)
            self.sequences.ensure_host_cache(st.cond)
            if st.cond.past is not None:
                st.cond.past.allocate_blocks = self.owner.residency.allocator_for_cache(
                    st.cond.past
                )
            num_tokens = self.append_generated_image(st.cond, image_state)
            if st.tu.past is not None:
                st.tu.past.allocate_blocks = self.owner.residency.allocator_for_cache(st.tu.past)
                self.append_generated_image(st.tu, image_state)
        return self._finalize_commit(op, st, image_state, None, num_tokens)

    @staticmethod
    def _store_frame(req_id: int, png: bytes | None) -> None:
        if png is None:
            return
        product_view = get_forward_context().product_view
        if product_view is None:
            raise invalid_descriptor("image commit requires an executor product view")
        product_view.append_frame(int(req_id), png)

    def _writeback_image_state(
        self, st: Any, op: dict[str, Any], latent: torch.Tensor
    ) -> FlowState:
        ip = st.image or {}
        params = parse_text_image_generation_params(ip)
        height = int(params.height)
        width = int(params.width)
        token_h = height // self.owner.latent_downsample
        token_w = width // self.owner.latent_downsample
        patch_size = self.owner.image_patch_size()
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
        indexes_cond = self.owner.flow_indexes(
            token_h,
            token_w,
            st.cond.t_index + 1,
            device=device,
        )
        indexes_tu = (
            self.owner.flow_indexes(
                token_h,
                token_w,
                st.tu.t_index + 1,
                device=device,
            )
            if st.tu.past is not None
            else None
        )
        indexes_iu = (
            self.owner.flow_indexes(
                token_h,
                token_w,
                st.iu.t_index + 1,
                device=device,
            )
            if st.iu.past is not None
            else None
        )
        self.owner.residency.latent.set(latent_handle, latent)
        return FlowState(
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
        png: bytes | None,
        num_tokens: int,
    ) -> dict[str, Any]:
        logits = st.cond.last_logits[:, -1, :].float()
        st.image_state = None
        latent_view = get_forward_context().latent_view
        if latent_view is None:
            raise invalid_descriptor("image commit requires an executor latent view")
        latent_view.release_state(image_state)
        self.owner.residency.release_scratch_cache(st.iu.past)
        st.iu = SequenceCache()
        self._store_frame(int(op["req_id"]), png)
        out = {
            "req_id": op["req_id"],
            "image_hw": [image_state.height, image_state.width],
            "logits": logits,
            "num_tokens": int(num_tokens),
        }
        if png is not None:
            out["image_png_b64"] = png_bytes_to_b64(png)
        return out


def _tp_rank(owner: Any) -> int:
    mesh = getattr(owner, "mesh", None)
    try:
        return int(getattr(mesh, "tp_rank", 0))
    except (TypeError, ValueError):
        return 0


class GenerationCommitAdapter(Protocol):
    """Neural boundary for committing a generated image from generation state.

    ``vae_decode`` is the latent->pixels product route. ``commit_generated_kv``
    persists the generated latents into the request KV through the family's
    unified forward; it stays a model neural entry because the writeback is a
    forward pass of the same backbone, not a system cache operation.
    """

    def vae_decode(
        self,
        latent: torch.Tensor,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> Any: ...
    def commit_generated_kv(self, req_id: int, gen_state: Any, block_ids: Any) -> int: ...


class GenerationImageMaterializer:
    """Materialize a generated image: decode, retain KV, and persist the frame."""

    def __init__(self, owner: GenerationCommitAdapter) -> None:
        self.owner = owner

    def commit(self, op: dict[str, Any], state: Any = None) -> dict[str, Any]:
        req_id = int(op["req_id"])
        context = get_forward_context()
        latent_view = context.latent_view
        if latent_view is None:
            raise invalid_descriptor("image commit requires an executor latent view")
        gen_state = latent_view.state(req_id)
        if gen_state is None:
            return {"req_id": req_id}
        if state is not None:
            state.append_new_block_ids(op.get("new_block_ids"))
        image = self.owner.vae_decode(
            gen_state.x_t,
            height=int(gen_state.H),
            width=int(gen_state.W),
        )
        product_view = context.product_view
        if product_view is None:
            raise invalid_descriptor("image commit requires an executor product view")
        record = product_view.record(req_id)
        retain_images = bool((record.image or {}).get("retain_images", True))
        num_tokens = 0
        if retain_images:
            block_ids = list(getattr(state, "block_ids", []) or [])
            num_tokens = int(self.owner.commit_generated_kv(req_id, gen_state, block_ids))
        png = pil_image_to_png_bytes(image)
        latent_view.pop_state(req_id)
        product_view.append_frame(req_id, png)
        return {
            "req_id": req_id,
            "image_png_b64": png_bytes_to_b64(png),
            "image_hw": [int(gen_state.H), int(gen_state.W)],
            "num_tokens": num_tokens,
        }


class LatentImageMaterializer:
    """Default commit driver: the model exposes only latent->pixels decode.

    The model returns raw neural output (an image tensor, a PIL image, or a
    result mapping); the executor normalizes and encodes the final product.
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def commit(self, op: dict[str, Any], state: Any = None) -> Any:
        del op
        return self.owner.vae_decode(getattr(state, "latent", None))


def build_commit_materializer(adapter: Any) -> Any:
    """Build the family's commit driver from its declared adapter surface."""
    transfer = getattr(adapter, "tower_session", None)
    if isinstance(transfer, ProductTransferSession):
        return ImageMaterializer(adapter, transfer, sequences=adapter._text_driver())
    if callable(getattr(adapter, "commit_generated_kv", None)):
        return GenerationImageMaterializer(adapter)
    return LatentImageMaterializer(adapter)


# ---------------------
# Input-image ingest
# ---------------------


class ImageEncodeAdapter(Protocol):
    """Collaborator surface for ingesting understanding images."""

    device: Any

    def sequence_forward(
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
    def image_features(
        self, image_input: torch.Tensor, *, grid_hw: torch.Tensor, gen_model: bool = ...
    ) -> torch.Tensor: ...
    def sequence_position_indexes(
        self,
        grid_hw: torch.Tensor,
        temporal_indexes: torch.Tensor,
    ) -> torch.Tensor: ...


class ImageEncoder:
    """Ingest external images: encode, cache, and append vision-token KV spans.

    The encoder owns the understanding-ingest orchestration — cache-span
    staging through the system sequence executor, intermediate-product
    residency in the ``ProductStore``, and the KV append — while the adapter
    supplies only neural feature extraction and geometry.
    """

    def __init__(
        self,
        owner: ImageEncodeAdapter,
        *,
        sequences: SequenceExecutor | None = None,
        products: Any = None,
    ) -> None:
        self.owner = owner
        self.sequences = sequences
        self.products = products

    def encode_row(self, row: EncodeRow) -> dict[str, Any]:
        """Stage, encode (or replay), and ingest one understanding-image row."""
        ctx = row.ctx
        if ctx.kind != VIT_ENCODE:
            raise invalid_descriptor(f"unsupported understanding encode op {ctx.kind!r}")
        if ctx.temporal_index is None:
            raise invalid_descriptor("vit_encode requires the shared temporal index (cond_pos)")
        if self.sequences is None or self.products is None:
            raise invalid_descriptor("image ingest requires executor sequence and product stores")
        st = self.sequences.state({"req_id": ctx.req_id})
        self.sequences.extend_cache_span(
            st.cond,
            req_id=ctx.req_id,
            new_block_ids=ctx.new_block_ids,
            pos_range=ctx.pos_range,
        )
        self.sequences.ensure_host_cache(st.cond)

        if row.pixels is not None:
            if row.grid is None or ctx.image_hw is None:
                raise invalid_descriptor("vit_encode pixels require a patch grid and dimensions")
            image_hw = [int(value) for value in ctx.image_hw]
            vit_embeds = self.encode_understanding_image(row.pixels, row.grid).detach()
            grid_hw = row.grid
            self.products.put_intermediate(
                ctx.handle,
                {
                    "kind": VIT_ENCODE,
                    "vit_embeds": vit_embeds,
                    "grid_hw": grid_hw.detach(),
                    "image_hw": image_hw,
                },
            )
        else:
            cached = self.products.intermediate(ctx.handle)
            if not isinstance(cached, Mapping) or cached.get("kind") != VIT_ENCODE:
                raise invalid_descriptor("cached vit_encode handle is not resident")
            cached_vit_embeds = cached.get("vit_embeds")
            cached_grid_hw = cached.get("grid_hw")
            cached_image_hw = cached.get("image_hw")
            if not isinstance(cached_vit_embeds, torch.Tensor) or not isinstance(
                cached_grid_hw, torch.Tensor
            ):
                raise invalid_descriptor("cached vit_encode payload is incomplete")
            if (
                not isinstance(cached_image_hw, list)
                or len(cached_image_hw) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in cached_image_hw
                )
            ):
                raise invalid_descriptor("cached vit_encode dimensions are invalid")
            vit_embeds = cached_vit_embeds
            grid_hw = cached_grid_hw
            image_hw = [int(value) for value in cached_image_hw]

        num_tokens = self.ingest_understanding_embeddings(
            st.cond,
            vit_embeds,
            grid_hw,
            t_index=int(ctx.temporal_index),
        )
        return {
            "req_id": ctx.req_id,
            "encoder_handle": ctx.handle,
            "num_tokens": num_tokens,
            "image_hw": image_hw,
        }

    def ingest_understanding_image(
        self,
        cache: SequenceCache,
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
        return owner.image_features(
            flattened_patches.to(owner.device),
            grid_hw=grid_hw.to(owner.device),
        )

    def ingest_understanding_embeddings(
        self,
        cache: SequenceCache,
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

        t_indexes = torch.full((num_tokens,), int(t_index), dtype=torch.long, device=device)
        indexes = owner.sequence_position_indexes(grid_hw[:1].to(device), t_indexes)

        past_len = cache.past.get_seq_length()
        # Patches attend to the full prefix and to each other: an all-zeros
        # additive mask over [num_tokens, past + num_tokens].
        mask = torch.zeros(1, 1, num_tokens, past_len + num_tokens, device=device)

        outputs = owner.sequence_forward(
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


def build_understanding_encoder(adapter: Any, products: Any) -> ImageEncoder | None:
    """Build the shared ingest driver when the adapter declares its surface."""
    if not (
        callable(getattr(adapter, "image_features", None))
        and callable(getattr(adapter, "sequence_position_indexes", None))
        and callable(getattr(adapter, "_text_driver", None))
    ):
        return None
    return ImageEncoder(adapter, sequences=adapter._text_driver(), products=products)


# ---------------------
# Tower execution session
# ---------------------


class ProductTransferAdapter(Protocol):
    """Family boundary for product and recurrent-state transfer."""

    device: Any
    gen_device: Any
    img_start_id: int
    residency: Any

    def tower_binding(self) -> Any: ...
    def program_state(self, req_id: int) -> Any: ...
    def _ensure_img_start(self, cache: SequenceCache | None) -> None: ...
    def _empty_img_start_prefix(self) -> SequenceCache: ...
    def product_transfer_dtype(self) -> torch.dtype: ...


class ProductTransferSession:
    """Own product and recurrent-state transfer between execution towers.

    The session always carries an in-process staging handoff built over the
    family's declared tower geometry (``owner.tower_binding``). The worker's
    Mover selects the und↔gen crossing for the deployment edge and installs it
    through :meth:`use_tower_handoff`; a cross-process crossing additionally
    enables the data-plane publish/fetch surface.
    """

    def __init__(
        self,
        owner: ProductTransferAdapter,
    ) -> None:
        self.owner = owner
        self._handoff: TowerHandoff = LocalP2PTowerHandoff(owner.tower_binding)
        self._data_plane_handoff: DataPlaneTowerHandoff | None = None

    @property
    def distributed(self) -> bool:
        return self._data_plane_handoff is not None

    @property
    def tower_binding(self) -> Any:
        """The family's live tower geometry declaration (for the Mover)."""
        return self.owner.tower_binding

    def use_tower_handoff(self, handoff: TowerHandoff) -> None:
        """Install the Mover-selected und↔gen crossing for this worker's edge.

        An in-process crossing replaces the staging handoff; a cross-process
        crossing adds the data-plane publish/fetch surface while the in-process
        handoff keeps serving same-device staging.
        """
        if isinstance(handoff, DataPlaneTowerHandoff):
            self._data_plane_handoff = handoff
        else:
            self._handoff = handoff

    def wait_gen_cache_ready(self, cache: Any) -> None:
        self._handoff.await_ready(cache)

    def maybe_publish_conditioning(self, req_id: int, sampled_token_id: int) -> str | None:
        """und side: when text decode samples ``img_start``, publish ``st.cond``.

        Returns the wire locator (for ``SeqResult.locator``) the gen pool will
        fetch and rebuild ``st.cond`` from, or ``None`` outside a cross-process
        edge / a non-image token.
        """
        if self._data_plane_handoff is None:
            return None
        return self.publish_conditioning(
            self.owner.program_state(int(req_id)),
            sampled_token_id,
        )

    def publish_conditioning(self, state: Any, sampled_token_id: int) -> Any:
        handoff = self._data_plane_handoff
        if handoff is None:
            return None
        if int(sampled_token_id) != int(self.owner.img_start_id):
            return None
        if state.cond.past is None:
            return None
        self.owner._ensure_img_start(state.cond)
        params = parse_text_image_generation_params(state.image or {})
        cfg_plan = build_flow_cfg_plan(
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
        handoff = self._data_plane_handoff
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
        handoff = self._data_plane_handoff
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
            dtype=self.owner.product_transfer_dtype(),
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
        return self._handoff.stage_conditioning(cache)

    def prepare_commit_latent(self, image_state: Any) -> torch.Tensor:
        if self._data_plane_handoff is not None:
            return (
                image_state.x_t[0]
                .unsqueeze(0)
                .to(
                    device=self.owner.device,
                    dtype=torch.bfloat16,
                    non_blocking=True,
                )
            )
        return self._handoff.writeback_commit(
            image_state.x_t[0].unsqueeze(0),
            device=self.owner.device,
            dtype=torch.bfloat16,
        )

    def publish_commit_latent(self, image_state: Any) -> Any:
        if self._data_plane_handoff is None:
            return self.prepare_commit_latent(image_state)
        return self._data_plane_handoff.publish_commit_latent(image_state.x_t[0].unsqueeze(0))

    def fetch_commit_latent(self, locator: Any) -> torch.Tensor:
        if self._data_plane_handoff is None:
            raise RuntimeError("commit_writeback requires a data-plane handoff")
        if isinstance(locator, str):
            locator = Locator.from_wire_json(locator)
        if not isinstance(locator, Locator):
            raise invalid_descriptor("commit_writeback locator must be a typed data-plane Locator")
        latent = self._data_plane_handoff.data_plane.fetch(locator)
        return latent.to(device=self.owner.device, dtype=torch.bfloat16, non_blocking=True)

    @staticmethod
    def encode_commit_locator(locator: Any) -> str:
        if not isinstance(locator, Locator):
            raise invalid_descriptor("commit locator must be a typed data-plane Locator")
        return locator.to_wire_json()
