"""BAGEL UniModel entry backed by the shared runner."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

from uniserve_worker.execution.flow import FlowGraphExecution, FlowRow
from uniserve_worker.execution.runner import PreparedFlowStep, flow_cfg_branch_count
from uniserve_worker.execution.segment import SegmentExecutor
from uniserve_worker.execution.sequence import SequenceCache, SequenceExecutor
from uniserve_worker.runtime.paged_denoise import (
    PagedDenoiseBranchSet,
    can_run_paged_denoise_attention,
)

from ..contracts.forward_batch import ForwardBatch
from ..contracts.resource_plan import (
    AdapterResourcePolicy,
    CapsDescriptor,
    EncoderResourcePolicy,
    KvBlockResourcePolicy,
    LatentTokens,
    PerBranch,
    ResourcePlan,
    active_latent_capacity_tokens,
)
from ..foundation.errors import WorkerError, capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import (
    decode_graph_padding_block_count,
    get_execution_config,
)
from ..foundation.sizing import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BATCH_OPS,
    ceil_div,
    derive_num_blocks,
    derive_runtime_kv_capacity,
)
from ..loader.weight_utils import iter_weights, stacked_params_mapping_loop, tensor_shape
from ..nn import LinearBase, MLPConnector, ParallelLMHead, local_kv_head_count
from ..nn.decoder import KVCache, MoTDecoderLayer, MoTModel, Segment
from ..nn.diffusion import FlowMatchSchedule, ScheduleDirection, TimestepEmbedder, init_latent
from ..nn.diffusion.cfg import CfgRecipe
from ..nn.quant import (
    QuantizationConfig,
    get_current_kv_cache_dtype,
    kv_cache_bytes_per_token,
    use_quantization_config,
)
from ..nn.vae import AutoEncoder, default_ae_params
from ..nn.vision import (
    PositionEmbedding,
    SiglipNavitConfig,
    SiglipNavitEncoder,
    get_flattened_position_ids_extrapolate,
    patchify_batch,
)
from ..processors.bagel import BagelImageProcessor
from ..runtime.image_params import parse_text_image_generation_params
from ..runtime.image_utils import pil_image_to_png_b64
from ..runtime.kv_pool import PagedKVPool
from ..runtime.lora import MergeOnLoadLoRA
from ..runtime.paged_text_cache import PagedTextCache, copy_paged_text_cache_span
from ..runtime.request_state import RequestState
from ..runtime.residency import (
    DEFAULT_ENCODER_CACHE_BUDGET,
    GenResidencySpec,
    KvCacheSpec,
    ResidencyManager,
    encoder_handle_from_mm_hash,
)
from .cache_registrations import bagel_cache_registration
from .registry import UniModelBase

__all__ = [
    "LLMConfig",
    "BagelConfig",
    "GenState",
    "BagelTextRequestState",
    "BagelForUnifiedGeneration",
    "EntryClass",
]

logger = logging.getLogger(__name__)

_BAGEL_RMS_NORM_EPS = 1e-6
_BAGEL_ROPE_THETA = 1_000_000.0
_BAGEL_VIT_LAYER_NORM_EPS = 1e-6
_BAGEL_IMAGE_MARKER_TOKENS = 2


def _weights_file(model_dir: str) -> str:
    for name in ("ema.safetensors", "model.safetensors"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no ema.safetensors or model.safetensors under {model_dir}")


@dataclass
class LLMConfig:
    """Language-model hyperparameters for the BAGEL stack."""

    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    rms_norm_eps: float = _BAGEL_RMS_NORM_EPS
    rope_theta: float = _BAGEL_ROPE_THETA
    qk_norm: bool = True
    bos_token_id: int = 151644
    eos_token_id: int = 151645

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class BagelConfig:
    """Top-level BAGEL model configuration (LLM, ViT, VAE, and latent settings)."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    visual_gen: bool = True
    visual_und: bool = True
    start_of_image_id: int = 151652
    end_of_image_id: int = 151653
    vae_z_channels: int = 16
    vae_downsample: int = 8
    latent_patch_size: int = 2
    max_latent_size: int = 32
    timestep_shift: float = 1.0
    vit_hidden_size: int = 1152
    vit_intermediate_size: int = 4304
    vit_num_hidden_layers: int = 26
    vit_num_attention_heads: int = 16
    vit_patch_size: int = 14
    vit_image_size: int = 980
    vit_layer_norm_eps: float = _BAGEL_VIT_LAYER_NORM_EPS
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"

    @property
    def latent_downsample(self) -> int:
        return self.vae_downsample * self.latent_patch_size

    @property
    def latent_token_capacity(self) -> int:
        return self.max_latent_size * self.max_latent_size

    @property
    def vit_token_capacity(self) -> int:
        return (self.vit_image_size // self.vit_patch_size) ** 2

    @property
    def latent_channel(self) -> int:
        return self.vae_z_channels

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size**2 * self.latent_channel

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BagelConfig":
        try:
            with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            logger.warning(
                "BAGEL config.json missing or invalid under %s (%s); "
                "falling back to default model dimensions",
                model_dir,
                exc,
            )
            raw = {}

        def side(name: str) -> dict[str, Any]:
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                return {}
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        if "llm_config" not in raw:
            raw = dict(raw)
            raw["llm_config"] = side("llm_config.json")
            raw["vit_config"] = side("vit_config.json")
            raw.setdefault("vae_config", side("vae_config.json"))

        llm_raw = raw["llm_config"]
        llm = LLMConfig(
            hidden_size=llm_raw["hidden_size"],
            intermediate_size=llm_raw["intermediate_size"],
            num_hidden_layers=llm_raw["num_hidden_layers"],
            num_attention_heads=llm_raw["num_attention_heads"],
            num_key_value_heads=llm_raw["num_key_value_heads"],
            vocab_size=llm_raw["vocab_size"],
            rms_norm_eps=llm_raw.get("rms_norm_eps", _BAGEL_RMS_NORM_EPS),
            rope_theta=llm_raw.get("rope_theta", 1e6),
            qk_norm=llm_raw.get("qk_norm", True),
            bos_token_id=llm_raw.get("bos_token_id", 151644),
            eos_token_id=llm_raw.get("eos_token_id", 151645),
        )
        vae = raw.get("vae_config", {})
        vit = raw.get("vit_config", {})
        max_latent = raw.get("max_latent_size", 32)
        try:
            rows = tensor_shape(_weights_file(model_dir), "latent_pos_embed.pos_embed")[0]
            max_latent = int(math.isqrt(rows))
        except Exception as exc:
            logger.warning(
                "BAGEL could not infer max_latent_size from latent_pos_embed (%s); "
                "using configured value %s",
                exc,
                max_latent,
            )
        return cls(
            llm=llm,
            visual_gen=raw.get("visual_gen", True),
            visual_und=raw.get("visual_und", True),
            start_of_image_id=raw.get("start_of_image_id", 151652),
            end_of_image_id=raw.get("end_of_image_id", 151653),
            vae_z_channels=vae.get("z_channels", 16),
            vae_downsample=vae.get("downsample", 8),
            latent_patch_size=raw.get("latent_patch_size", 2),
            max_latent_size=max_latent,
            timestep_shift=raw.get("timestep_shift", 1.0),
            vit_hidden_size=vit.get("hidden_size", 1152),
            vit_intermediate_size=vit.get("intermediate_size", 4304),
            vit_num_hidden_layers=vit.get("num_hidden_layers", 27) - 1,
            vit_num_attention_heads=vit.get("num_attention_heads", 16),
            vit_patch_size=vit.get("patch_size", 14),
            vit_image_size=vit.get("image_size", 980),
            vit_layer_norm_eps=vit.get("layer_norm_eps", _BAGEL_VIT_LAYER_NORM_EPS),
            vit_max_num_patch_per_side=raw.get("vit_max_num_patch_per_side", 70),
            connector_act=raw.get("connector_act", "gelu_pytorch_tanh"),
        )


class _BagelGraph(nn.Module):
    """BAGEL neural graph: MoT language model, VAE, ViT, and flow-matching connectors."""

    def __init__(self, cfg: BagelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.llm.hidden_size
        # Install checkpoint quantization context during layer construction.
        self._quant_config = QuantizationConfig.from_model_config(cfg)
        with use_quantization_config(self._quant_config):
            self.lm = MoTModel(cfg.llm)
            self.lm_head = ParallelLMHead(hidden, cfg.llm.vocab_size, bias=False)
            self.vae2llm = LinearBase(cfg.patch_latent_dim, hidden)
            self.llm2vae = LinearBase(hidden, cfg.patch_latent_dim)
            self.time_embedder = TimestepEmbedder(hidden)
            self.latent_pos_embed = PositionEmbedding(
                cfg.max_latent_size, hidden, init_sincos=False
            )
            self.vae = AutoEncoder(default_ae_params())
            self.vit_model = SiglipNavitEncoder(
                SiglipNavitConfig(
                    patch_size=cfg.vit_patch_size,
                    hidden_size=cfg.vit_hidden_size,
                    image_size=cfg.vit_image_size,
                    num_attention_heads=cfg.vit_num_attention_heads,
                    intermediate_size=cfg.vit_intermediate_size,
                    num_hidden_layers=cfg.vit_num_hidden_layers,
                    layer_norm_eps=cfg.vit_layer_norm_eps,
                )
            )
            self.connector = MLPConnector(cfg.vit_hidden_size, hidden, cfg.connector_act)
            self.vit_pos_embed = PositionEmbedding(
                cfg.vit_max_num_patch_per_side, hidden, init_sincos=False
            )
        self._gen_graph_layouts: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.llm.num_hidden_layers

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    def new_cache(self) -> KVCache:
        return KVCache(self.num_layers)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.embed_tokens(ids)

    def build_und_segment(self, token_ids, start_pos, cache, update=True) -> Segment:
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        n_tokens = ids.shape[0]
        positions = torch.arange(start_pos, start_pos + n_tokens, device=self.device)
        embeds = self.embed_tokens(ids).to(torch.bfloat16)
        is_gen = torch.zeros(n_tokens, dtype=torch.bool, device=self.device)
        return Segment(
            embeds=embeds,
            positions=positions,
            is_gen=is_gen,
            cache=cache,
            causal=True,
            update_cache=update,
        )

    def gen_segment_embeds(self, num_vae, vae_pos_ids, x_t, timestep) -> torch.Tensor:
        """Marker/VAE-latent/timestep embeddings for one gen segment, ``[num_vae+2, hidden]``.

        Shared by graph denoise and image-commit paths so marker and latent
        embeddings follow one model contract.
        """
        hidden = self.cfg.llm.hidden_size
        total = int(num_vae) + 2
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=self.device,
        )
        marker_emb = self.embed_tokens(marker_ids).to(torch.bfloat16)
        x_t = x_t.to(device=self.device, dtype=torch.bfloat16)
        timesteps = torch.full((int(num_vae),), float(timestep), device=self.device)
        vae_emb = (
            self.vae2llm(x_t)
            + self.time_embedder(timesteps)
            + self.latent_pos_embed(vae_pos_ids.to(self.device))
        ).to(torch.bfloat16)
        embeds = torch.empty(total, hidden, dtype=torch.bfloat16, device=self.device)
        embeds[0] = marker_emb[0]
        embeds[1 : 1 + int(num_vae)] = vae_emb
        embeds[1 + int(num_vae)] = marker_emb[1]
        return embeds

    def gen_segment_is_gen(self, num_vae) -> torch.Tensor:
        """Gen-segment modality pattern: markers are text-modality, latents gen."""
        is_gen = torch.zeros(int(num_vae) + 2, dtype=torch.bool, device=self.device)
        is_gen[1 : 1 + int(num_vae)] = True
        return is_gen

    def gen_segment_graph_layout(
        self,
        batch_size: int,
        num_vae: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        num_vae = int(num_vae)
        key = (batch_size, num_vae, str(self.device))
        cached = self._gen_graph_layouts.get(key)
        if cached is not None:
            return cached
        total = num_vae + _BAGEL_IMAGE_MARKER_TOKENS
        is_gen = self.gen_segment_is_gen(num_vae)
        row_offsets = torch.arange(batch_size, device=self.device, dtype=torch.long) * total
        marker_offsets = torch.tensor(
            [0, total - 1],
            device=self.device,
            dtype=torch.long,
        )
        text_idx = (row_offsets.unsqueeze(1) + marker_offsets.unsqueeze(0)).reshape(-1)
        layout = (is_gen, text_idx)
        self._gen_graph_layouts[key] = layout
        return layout

    def build_gen_segment(
        self, num_vae, vae_pos_ids, x_t, timestep, position_id, cache, update=False
    ) -> Segment:
        total = int(num_vae) + 2
        embeds = self.gen_segment_embeds(num_vae, vae_pos_ids, x_t, timestep)
        positions = torch.full((total,), int(position_id), dtype=torch.long, device=self.device)
        return Segment(
            embeds=embeds,
            positions=positions,
            is_gen=self.gen_segment_is_gen(num_vae),
            cache=cache,
            causal=False,
            update_cache=update,
        )

    @torch.no_grad()
    def run(self, segments) -> list[torch.Tensor]:
        return self.lm.forward_segments(segments)

    @torch.no_grad()
    def logits(self, hidden_last_row: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_last_row)

    @torch.no_grad()
    def velocity_from_hidden(self, hidden, num_vae) -> torch.Tensor:
        return self.llm2vae(hidden[1 : 1 + int(num_vae)].to(torch.bfloat16))

    def latent_position_ids(self, height: int, width: int) -> torch.Tensor:
        return get_flattened_position_ids_extrapolate(
            height,
            width,
            self.cfg.latent_downsample,
            self.cfg.max_latent_size,
        )

    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        return height // self.cfg.latent_downsample, width // self.cfg.latent_downsample

    @torch.no_grad()
    def vit_encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.vit_encode_batch(image_tensor.unsqueeze(0))[0]

    @torch.no_grad()
    def vit_encode_batch(self, image_tensors: torch.Tensor) -> torch.Tensor:
        if image_tensors.ndim != 4:
            raise invalid_descriptor("BAGEL batched ViT encode expects NCHW pixels")
        image_tensors = image_tensors.to(self.device)
        batch, _channels, height, width = image_tensors.shape
        patch = self.cfg.vit_patch_size
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            patch,
            self.cfg.vit_max_num_patch_per_side,
        ).to(self.device)
        patches = (
            patchify_batch(image_tensors, patch)
            .reshape(
                -1,
                patch * patch * int(image_tensors.shape[1]),
            )
            .to(self.device, torch.bfloat16)
        )
        tokens_per_image = (height // patch) * (width // patch)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * tokens_per_image,
            tokens_per_image,
            dtype=torch.int32,
            device=self.device,
        )
        packed_pos_ids = pos_ids.repeat(batch)
        vit_out = self.vit_model(
            patches,
            {"position_ids": packed_pos_ids, "cu_seqlens": cu_seqlens},
        )
        emb = self.connector(vit_out) + self.vit_pos_embed(packed_pos_ids)
        return emb.reshape(batch, tokens_per_image, -1).to(torch.bfloat16)

    @torch.no_grad()
    def vae_encode_clean(self, image_tensor: torch.Tensor):
        latents, pos_ids, hw = self.vae_encode_clean_batch(image_tensor.unsqueeze(0))
        return latents[0], pos_ids, hw

    @torch.no_grad()
    def vae_encode_clean_batch(
        self,
        image_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if image_tensors.ndim != 4:
            raise invalid_descriptor("BAGEL batched VAE encode expects NCHW pixels")
        image_tensors = image_tensors.to(self.device)
        _batch, _channels, height, width = image_tensors.shape
        vae_dtype = next(self.vae.parameters()).dtype
        latent_images = self.vae.encode(image_tensors.to(vae_dtype))
        patch = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        h = height // self.cfg.latent_downsample
        w = width // self.cfg.latent_downsample
        latents = latent_images[:, :, : h * patch, : w * patch].reshape(
            int(latent_images.shape[0]),
            channels,
            h,
            patch,
            w,
            patch,
        )
        latents = torch.einsum("nchpwq->nhwpqc", latents).reshape(
            int(latent_images.shape[0]),
            -1,
            patch * patch * channels,
        )
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            self.cfg.latent_downsample,
            self.cfg.max_latent_size,
        ).to(self.device)
        return latents.to(torch.bfloat16), pos_ids, (h, w)

    def build_und_image_segment(self, vit_embeds, position_id, cache, update=True) -> Segment:
        n_tokens = vit_embeds.shape[0]
        hidden = self.cfg.llm.hidden_size
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=self.device,
        )
        marker_emb = self.embed_tokens(marker_ids).to(torch.bfloat16)
        total = n_tokens + 2
        embeds = torch.empty(total, hidden, dtype=torch.bfloat16, device=self.device)
        embeds[0] = marker_emb[0]
        embeds[1 : 1 + n_tokens] = vit_embeds.to(self.device)
        embeds[1 + n_tokens] = marker_emb[1]
        positions = torch.full((total,), int(position_id), dtype=torch.long, device=self.device)
        is_gen = torch.zeros(total, dtype=torch.bool, device=self.device)
        return Segment(
            embeds=embeds,
            positions=positions,
            is_gen=is_gen,
            cache=cache,
            causal=False,
            update_cache=update,
        )

    @torch.no_grad()
    def vae_decode(self, latent: torch.Tensor, height: int, width: int) -> Image.Image:
        h, w = self.latent_hw(height, width)
        patch = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        latent = latent.reshape(1, h, w, patch, patch, channels)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, channels, h * patch, w * patch)
        latent = latent.to(next(self.vae.parameters()).dtype)
        img = self.vae.decode(latent)
        img = (img * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(img.to(torch.uint8).cpu().numpy())

    def load_weights(self, weights, *, dtype: torch.dtype = torch.bfloat16) -> None:
        exact = {
            "language_model.model.embed_tokens.weight": "lm.embed_tokens.weight",
            "language_model.model.norm.weight": "lm.norm.weight",
            "language_model.model.norm_moe_gen.weight": "lm.norm_moe_gen.weight",
            "language_model.lm_head.weight": "lm_head.weight",
            "vae2llm.weight": "vae2llm.weight",
            "vae2llm.bias": "vae2llm.bias",
            "llm2vae.weight": "llm2vae.weight",
            "llm2vae.bias": "llm2vae.bias",
            "time_embedder.mlp.0.weight": "time_embedder.mlp.0.weight",
            "time_embedder.mlp.0.bias": "time_embedder.mlp.0.bias",
            "time_embedder.mlp.2.weight": "time_embedder.mlp.2.weight",
            "time_embedder.mlp.2.bias": "time_embedder.mlp.2.bias",
            "latent_pos_embed.pos_embed": "latent_pos_embed.pos_embed",
        }
        stacked: list[tuple[str, str, str | int]] = [
            ("qkv_proj_moe_gen", "q_proj_moe_gen", "q"),
            ("qkv_proj_moe_gen", "k_proj_moe_gen", "k"),
            ("qkv_proj_moe_gen", "v_proj_moe_gen", "v"),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        def map_name(name: str) -> str | None:
            if name in exact:
                return exact[name]
            if name.startswith("language_model.model.layers."):
                mapped = name.replace("language_model.model.layers.", "lm.layers.", 1)
                return mapped.replace(".self_attn.", ".")
            if name.startswith("vit_model.vision_model.embeddings."):
                return name.replace("vit_model.vision_model.embeddings.", "vit_model.", 1)
            if name.startswith("vit_model.vision_model.encoder."):
                mapped = name.replace("vit_model.vision_model.", "vit_model.", 1)
                mapped = mapped.replace(".mlp.fc1.", ".mlp.0.")
                return mapped.replace(".mlp.fc2.", ".mlp.2.")
            if name.startswith("vit_model.vision_model.post_layernorm."):
                return name.replace(
                    "vit_model.vision_model.post_layernorm.",
                    "vit_model.encoder.post_layernorm.",
                    1,
                )
            if name.startswith(("connector.", "vit_pos_embed.")):
                return name
            return None

        all_weights = list(weights)
        loaded, ignored = stacked_params_mapping_loop(
            self,
            all_weights,
            stacked,
            name_mapper=map_name,
            dtype=dtype,
        )
        expected = {name for name, _ in self.named_parameters() if not name.startswith("vae.")}
        missing = sorted(expected - loaded)
        if missing:
            raise capability_mismatch(
                "BAGEL weight load mismatch: "
                f"missing={missing[:8]} ({len(missing)}) ignored={ignored[:8]}"
            )
        logger.info("loaded BAGEL graph weights (%d params)", len(loaded))

    def load_vae_weights(self, weights) -> None:
        state_dict = dict(weights)
        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        real_missing = [name for name in missing if "reg" not in name]
        if real_missing or unexpected:
            raise capability_mismatch(
                "BAGEL VAE weight load mismatch: "
                f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
            )
        logger.info("loaded BAGEL VAE weights (%d tensors)", len(state_dict))


def _load_bagel(model_dir: str, device: str = "cuda") -> _BagelGraph:
    cfg = BagelConfig.from_pretrained(model_dir)
    model = _BagelGraph(cfg).eval()
    model.load_weights(iter_weights([Path(_weights_file(model_dir))]))
    model.load_vae_weights(iter_weights([Path(model_dir) / "ae.safetensors"]))
    model.to(device=device, dtype=torch.bfloat16)
    return model


@dataclass(frozen=True)
class _LoadedBagelRuntime:
    model: _BagelGraph
    pool: PagedKVPool
    image_processor: BagelImageProcessor


@dataclass(slots=True)
class GenState:
    """Mutable image-generation state for one BAGEL denoise/commit cycle."""

    x_t: torch.Tensor
    vae_pos_ids: torch.Tensor
    num_vae: int
    H: int
    W: int
    schedule: FlowMatchSchedule
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_renorm_type: str
    cfg_renorm_min: float
    cfg_interval: tuple[float, float]
    cond_pos: int
    cond_branch_kvlen: int = 0
    cfg_pos: int = 0
    paged_branches: PagedDenoiseBranchSet | None = None
    graph_image: _BagelDenoiseGraphImage | None = None


@dataclass(slots=True)
class _BagelDenoiseGraphImage:
    token_h: int
    token_w: int
    height: int
    width: int
    cond_cache: PagedTextCache | None = None
    tu_cache: PagedTextCache | None = None
    iu_cache: PagedTextCache | None = None
    indexes: dict[str, torch.Tensor] = field(default_factory=dict)


# BAGEL's start-of-image marker string (token id 151652 in the Qwen2 vocab).
# The worker never tokenizes it (encode ops embed the marker directly); the
# shared text driver takes it as configuration.
_BAGEL_IMG_START_TOKEN = "<|vision_start|>"

# Scheduler-visible denoise scratch capacity in tokens. Each image stages cond
# and text-uncond prefixes plus one transient gen span per physical branch. The
# physical pool also reserves one request-KV-pool-sized region for text rows in
# shared packed forwards; that staging reserve is not additional denoise
# admission capacity.
_BAGEL_SCRATCH_CAPACITY_TOKENS = 65536


@dataclass
class BagelTextRequestState:
    """Per-request state for system sequence execution.

    ``cond`` is the single conditional text branch; its ``block_ids`` list is
    shared (same object) with the runner ``RequestState.block_ids`` so the
    driver's text ops and BAGEL's encode/denoise/commit ops ingest host
    ``new_block_ids`` into one list and every path sees every block.
    """

    cond: SequenceCache = field(default_factory=SequenceCache)


class BagelForUnifiedGeneration(UniModelBase):
    """BAGEL unified text/image model with VAE denoise and ViT/VAE encode paths."""

    family = "bagel"
    architectures = ("BagelForUnifiedGeneration", "BAGEL", "bagel")
    supported_ops = (
        "prefill_und",
        "decode_und",
        "denoise_gen",
        "commit_gen",
        "vit_encode",
        "vae_encode",
    )
    cache_registration_factory = staticmethod(bagel_cache_registration)
    supported_controls = ("free_encoder", "load_lora", "unload_lora")
    adapter_mode = "engine_wide"
    resource_plan = ResourcePlan(
        kv_block=KvBlockResourcePolicy.PER_BLOCK,
        encoder_output=EncoderResourcePolicy.PER_HANDLE,
        image_latent=LatentTokens(downsample=16),
        scratch=PerBranch(),
        adapter=AdapterResourcePolicy.PER_ADAPTER,
    )

    def velocity_parameterization(self) -> str:
        return "velocity"

    # Encoder-output cache capacity reported to the host scheduler.
    ENCODER_CACHE_BUDGET = DEFAULT_ENCODER_CACHE_BUDGET

    @classmethod
    def recognizes(cls, model_path: str | Path) -> bool:
        root = Path(model_path)
        return ((root / "ema.safetensors").exists() or (root / "model.safetensors").exists()) and (
            root / "ae.safetensors"
        ).exists()

    def __init__(
        self,
        config: Any | None = None,
        *,
        model: _BagelGraph | None = None,
        image_processor: BagelImageProcessor | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.model = model
        self.block_size = int(block_size)
        self.kv_token_capacity = int(kv_token_capacity) if kv_token_capacity is not None else None
        self.attention_backend = attention_backend or "auto"
        self.device = str(device)
        self.cfg = (
            model.cfg
            if model is not None
            else (config if isinstance(config, BagelConfig) else BagelConfig())
        )
        self.resource_plan = self._build_resource_plan()
        self.image_processor = image_processor or (
            BagelImageProcessor() if model is not None else None
        )
        self.states: dict[int, RequestState] = {}
        self.generation_session = TextImageGenerationSession(self)
        # Interleaved-text-driver owner surface: per-request driver states plus
        # the marker/eos ids the driver reads as configuration. BAGEL has no
        # worker-side tokenizer (the host tokenizes). The scratch pool holds
        # the denoise CFG-branch prefixes and their transient gen rows.
        self.reqs: dict[int, BagelTextRequestState] = {}
        self.tokenizer = None
        self.scratch_pool: PagedKVPool | None = None
        self._scratch_blocks = 0
        self.eos_id = int(self.cfg.llm.eos_token_id)
        self.img_start_id = int(self.cfg.start_of_image_id)
        self.img_end_id = int(self.cfg.end_of_image_id)
        self._shared_text_driver: SequenceExecutor | None = None
        self.pool: PagedKVPool | None = None
        self.residency = ResidencyManager(encoder_cache_budget=self.ENCODER_CACHE_BUDGET)
        self.lora: MergeOnLoadLoRA | None = None
        # engine_wide LoRA merges/unmerges mutate shared model weights in place;
        # serialize load/unload so a concurrent control op cannot interleave a
        # half-applied delta with another adapter's merge.
        self._lora_lock = threading.Lock()
        c = self.cfg.llm
        self.kv_cache_dtype = get_current_kv_cache_dtype(config)
        self.bytes_per_token = self._kv_bytes_per_token(torch.bfloat16)
        if self.model is not None:
            self.kv_token_capacity = int(
                derive_runtime_kv_capacity(
                    device=self.device,
                    block_size=self.block_size,
                    kv_token_capacity=self.kv_token_capacity,
                    bytes_per_token=self.bytes_per_token,
                    memory_fraction=get_execution_config().kv_memory_fraction,
                    floor=64,
                ).token_capacity
            )
            self.num_blocks = derive_num_blocks(self.block_size, self.kv_token_capacity, floor=64)
            self.residency = self._build_residency(c)
            self.pool = self.residency.kv
            self.scratch_pool = self.residency.scratch
            self.bytes_per_token = self._kv_bytes_per_token(torch.bfloat16)
            self.lora = MergeOnLoadLoRA(self.model)
        else:
            self.num_blocks = derive_num_blocks(self.block_size, self.kv_token_capacity, floor=64)
        self.flow_graph_execution = FlowGraphExecution(self)
        self.segment_executor = SegmentExecutor(self)

    def _build_resource_plan(self) -> ResourcePlan:
        return ResourcePlan(
            kv_block=KvBlockResourcePolicy.PER_BLOCK,
            encoder_output=EncoderResourcePolicy.PER_HANDLE,
            image_latent=LatentTokens(downsample=int(self.cfg.latent_downsample)),
            scratch=PerBranch(),
            adapter=AdapterResourcePolicy.PER_ADAPTER,
        )

    def _denoise_scratch_num_blocks(self, block_size: int | None = None) -> int:
        return ceil_div(_BAGEL_SCRATCH_CAPACITY_TOKENS, int(block_size or self.block_size))

    def _scratch_num_blocks(self, block_size: int | None = None) -> int:
        # Every live text token can require one same-pool packed-forward mirror.
        # Host KV admission bounds their aggregate by ``num_blocks``; adding
        # that exact capacity preserves the independently admitted denoise
        # reservation without allowing it to consume text-staging headroom.
        return self._denoise_scratch_num_blocks(block_size) + int(self.num_blocks)

    def _build_residency(self, cfg: LLMConfig) -> ResidencyManager:
        # System-owned residency: the worker-owned ResidencyManager constructs
        # and owns the KV pool; the model declares only geometry/sizing here.
        # The scratch pool holds denoise CFG branches and same-pool text mirrors
        # used by shared packed text/denoise forwards.
        self._scratch_blocks = self._scratch_num_blocks()
        return ResidencyManager.build_gen(
            GenResidencySpec(
                kv=KvCacheSpec(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=local_kv_head_count(cfg.num_key_value_heads),
                    head_dim=cfg.head_dim,
                    dtype=torch.bfloat16,
                    store_dtype=self._kv_store_dtype_for(torch.bfloat16),
                ),
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                device=self.device,
                scratch_num_blocks=self._scratch_blocks,
                reserved_tail_blocks=decode_graph_padding_block_count(self.block_size),
                encoder_cache_budget=self.ENCODER_CACHE_BUDGET,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        **_kwargs: Any,
    ) -> "BagelForUnifiedGeneration":
        model = _load_bagel(model_path, device=device)
        return cls(
            model.cfg,
            model=model,
            device=device,
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            image_processor=BagelImageProcessor(),
            attention_backend=attention_backend,
        )

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> CapsDescriptor:
        block = int(block_size or self.block_size)
        cap = kv_token_capacity if kv_token_capacity is not None else self.kv_token_capacity
        if self.model is not None:
            cap = int(cap or self.kv_token_capacity or self.block_size * self.num_blocks)
            physical_blocks = int(self.num_blocks)
        else:
            physical_blocks = derive_num_blocks(block, cap, floor=64)
        padding_blocks = decode_graph_padding_block_count(block)
        num_blocks = max(1, physical_blocks - padding_blocks)
        c = self.cfg.llm
        # Report only schedulable denoise capacity. The physical pool's separate
        # text-staging reserve mirrors already-admitted request KV and must not
        # increase host denoise admission.
        scratch_blocks = self._denoise_scratch_num_blocks(block)
        return CapsDescriptor(
            block_size=block,
            num_blocks=num_blocks,
            num_layers=c.num_hidden_layers,
            scratch_capacity_tokens=int(scratch_blocks * block),
            max_latent_size=active_latent_capacity_tokens(
                self.cfg.latent_token_capacity,
                cap,
            ),
            latent_downsample=self.cfg.latent_downsample,
            max_vae_grid_tokens=(self.cfg.latent_token_capacity + _BAGEL_IMAGE_MARKER_TOKENS),
            max_vit_grid_tokens=(self.cfg.vit_token_capacity + _BAGEL_IMAGE_MARKER_TOKENS),
            commit_marker_tokens=_BAGEL_IMAGE_MARKER_TOKENS,
            gen_rope_advance=2,
            max_cfg_branches=3,
            bytes_per_token=self._kv_bytes_per_token(torch.bfloat16),
            max_batch_ops=DEFAULT_MAX_BATCH_OPS,
            attention_backend=self.attention_backend,
            kv_dtype=self._kv_dtype_name_for(torch.bfloat16),
            encoder_cache_budget=self.ENCODER_CACHE_BUDGET,
        )

    def _kv_bytes_per_token(self, compute_dtype: torch.dtype) -> int:
        c = self.cfg.llm
        return kv_cache_bytes_per_token(
            num_kv_heads=local_kv_head_count(c.num_key_value_heads),
            head_dim=c.head_dim,
            num_layers=c.num_hidden_layers,
            compute_dtype=compute_dtype,
            store_dtype=self.kv_cache_dtype,
        )

    def on_new_request(self, req_id: int, state: RequestState) -> None:
        r = int(req_id)
        self.states[r] = state
        self.reqs.pop(r, None)
        self.program_state(r)
        self.generation_session.begin_request(
            r,
            sampling=state.sampling,
            image=state.image,
            neg_token_ids=state.neg_token_ids,
            lora_id=state.lora_id,
        )

    def drop_request(self, req_id: int) -> None:
        r = int(req_id)
        state = self.states.pop(r, None)
        text_state = self.reqs.pop(r, None)
        if text_state is not None:
            self.segment_executor.release_staging(text_state.cond.past)
        self.generation_session.release_request(r, request_state=state)

    def free_encoder(self, handles) -> None:
        # Encoder-output residency is system-owned: the handle→embedding store lives
        # on the ResidencyManager, not the model.
        for h in handles or []:
            self.residency.encoder.pop(int(h))

    def load_lora(self, lora_id, lora_path) -> None:
        if self.lora is None:
            raise capability_mismatch("BAGEL model weights are not loaded")
        with self._lora_lock:
            count = self.lora.load(lora_id, lora_path)
        logger.info("merged LoRA adapter %s into %d parameters", lora_id, count)

    def unload_lora(self, lora_id) -> None:
        if self.lora is None:
            raise capability_mismatch("BAGEL model weights are not loaded")
        with self._lora_lock:
            count = self.lora.unload(lora_id)
        if count:
            logger.info("unmerged LoRA adapter %s", lora_id)

    def _record(self, req_id: int) -> dict:
        return self.generation_session.record(int(req_id))

    def _state(self, req_id: int) -> RequestState:
        req_id = int(req_id)
        state = self.states.get(req_id)
        if state is None:
            state = RequestState()
            state.seed = req_id
            state.rng = torch.Generator(device="cpu").manual_seed(req_id)
            self.states[req_id] = state
        return state

    def _length(self, req_id: int) -> int:
        return self._state(req_id).kv_length()

    def _set_length(self, req_id: int, value: int) -> None:
        self._state(req_id).set_kv_length(value)

    def _gen_state(self, req_id: int) -> GenState | None:
        return self.generation_session.generation_state(int(req_id))

    def _set_gen_state(self, req_id: int, value: GenState) -> None:
        self.generation_session.set_generation_state(int(req_id), value)

    def _pop_gen_state(self, req_id: int) -> GenState | None:
        return self.generation_session.release_generated_state(int(req_id))

    def _release_paged_denoise_branches(self, gs: GenState) -> None:
        if gs.graph_image is not None:
            gs.graph_image = None
        self.generation_session.release_paged_branches(gs)

    def _extend_blocks(self, op) -> list[int]:
        state = self._state(int(op["req_id"]))
        state.append_new_block_ids(op.get("new_block_ids"))
        return state.block_ids

    # ---- sequence adapter surface ------------------------------------------

    @property
    def kv_pool(self) -> PagedKVPool | None:
        # Sequence execution uses this request KV pool.
        return self.pool

    @property
    def num_layers(self) -> int:
        return int(self.cfg.llm.num_hidden_layers)

    def _text_driver(self) -> SequenceExecutor:
        driver = self._shared_text_driver
        if driver is None:
            driver = SequenceExecutor(
                self,
                request_state_factory=BagelTextRequestState,
                image_start_token=_BAGEL_IMG_START_TOKEN,
            )
            self._shared_text_driver = driver
        return driver

    def _extend_cache_blocks(self, cache: SequenceCache, op: dict[str, Any]) -> None:
        self._text_driver().extend_cache_blocks(cache, op)

    def _ensure_host_cache(self, cache: SequenceCache) -> None:
        self._text_driver().ensure_host_cache(cache)

    def program_state(self, req_id: int) -> BagelTextRequestState:
        req_id = int(req_id)
        st = self.reqs.get(req_id)
        if st is None:
            st = BagelTextRequestState()
            # Share one block list between the driver's text cache and the
            # runner request state (see BagelTextRequestState docstring).
            st.cond.block_ids = self._state(req_id).block_ids
            self.reqs[req_id] = st
        return st

    def sequence_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._ensure_loaded().model.embed_tokens(input_ids).to(torch.bfloat16)

    def packed_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._ensure_loaded().model.embed_tokens(input_ids).to(torch.bfloat16)

    def _text_indexes(
        self,
        start: int,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target = self.device if device is None else device
        positions = torch.arange(int(start), int(start) + int(seq_len), device=target)
        spatial = torch.zeros(int(seq_len), dtype=torch.long, device=target)
        return torch.stack((positions, spatial, spatial), dim=0)

    def packed_decoder_forward(
        self,
        input_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        indexes: torch.Tensor,
        forward_stream: Any,
        kv_view: Any,
    ) -> torch.Tensor:
        return self._ensure_loaded().model.lm.forward_packed_visible(
            input_embeds,
            route_indicators=route_indicators,
            indexes=indexes,
            forward_stream=forward_stream,
            kv_view=kv_view,
        )

    def packed_text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._ensure_loaded().model.logits(hidden_states)

    def packed_hidden_to_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        del t, latent, image_size
        num_vae = int(image_token_num) - _BAGEL_IMAGE_MARKER_TOKENS
        if num_vae <= 0 or tuple(hidden_states.shape[:2]) != (1, int(image_token_num)):
            raise invalid_descriptor("BAGEL packed hidden states do not match image geometry")
        return self._ensure_loaded().model.velocity_from_hidden(hidden_states[0], num_vae)

    def segment_graph_attention(self) -> Any:
        layers = self._ensure_loaded().model.lm.layers
        if not layers:
            raise capability_mismatch("BAGEL packed graph requires at least one decoder layer")
        return layers[0].attn

    def packed_denoise_indicators(
        self,
        step: PreparedFlowStep,
        q_len: int,
    ) -> torch.Tensor:
        generation = step.extra["gs"]
        indicators, _text_indexes = self._ensure_loaded().model.gen_segment_graph_layout(
            1,
            int(generation.num_vae),
        )
        if int(indicators.numel()) != int(q_len):
            raise invalid_descriptor("BAGEL denoise modality mask does not match token geometry")
        return indicators

    def packed_route_indices(
        self,
        forward_stream: Any,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_indices: list[int] = []
        gen_indices: list[int] = []
        offset = 0
        for segment in forward_stream.segments:
            q_len = int(segment.q_len)
            if q_len <= 0:
                raise invalid_descriptor("BAGEL packed segments must contain tokens")
            if segment.modality == "und":
                text_indices.extend(range(offset, offset + q_len))
            elif segment.modality == "gen":
                if q_len <= _BAGEL_IMAGE_MARKER_TOKENS:
                    raise invalid_descriptor("BAGEL generation segment is missing latent tokens")
                text_indices.extend((offset, offset + q_len - 1))
                gen_indices.extend(range(offset + 1, offset + q_len - 1))
            else:
                raise invalid_descriptor("BAGEL packed segment has an unsupported modality")
            offset += q_len
        return (
            torch.tensor(text_indices, dtype=torch.long, device=device),
            torch.tensor(gen_indices, dtype=torch.long, device=device),
        )

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
        return_all_logits: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run the MoT understanding expert stack over the shared paged text KV.

        The sequence executor sends strictly-causal token spans here
        (BAGEL is 1-D rope, so only ``indexes[0]`` is consumed; the spatial rows
        are zero). Its block-causal / past-visible masks are equivalent to plain
        causal attention, so ``attention_mask`` is not materialized.
        """
        del attention_mask, use_cache, text_only_rope, causal_paged_update
        m = self._ensure_loaded().model
        if input_ids is None and inputs_embeds is None:
            raise invalid_descriptor(
                "BAGEL sequence forward requires exactly one of input_ids or inputs_embeds"
            )
        if input_ids is not None and inputs_embeds is not None:
            raise invalid_descriptor(
                "BAGEL sequence forward requires exactly one of input_ids or inputs_embeds"
            )
        if indexes is None and cache_position is not None:
            indexes = cache_position.reshape(1, -1)
        if indexes is None or past_key_values is None:
            raise invalid_descriptor("BAGEL sequence forward requires indexes and a paged cache")
        if inputs_embeds is None:
            if input_ids is None:
                raise invalid_descriptor("BAGEL text input ids are missing")
            inputs_embeds = m.embed_tokens(input_ids).to(torch.bfloat16)
        batch, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        if batch > 1:
            if seq_len != 1:
                raise invalid_descriptor(
                    "BAGEL batched sequence forward requires one token per row"
                )
            hidden = m.lm.forward_paged_text_batch(
                inputs_embeds,
                indexes[0].reshape(-1),
                past_key_values,
            )
        else:
            hidden = m.lm.forward_paged_text(
                inputs_embeds.reshape(seq_len, -1),
                indexes[0].reshape(-1),
                past_key_values,
            ).view(batch, seq_len, -1)
        logits = (
            m.logits(hidden.reshape(batch * seq_len, -1)).view(batch, seq_len, -1)
            if return_all_logits
            else m.logits(hidden[:, -1, :]).unsqueeze(1)
        )
        return CausalLMOutputWithPast(
            logits=cast(Any, logits),
            past_key_values=past_key_values,
        )

    def query_geometry(self) -> tuple[int, float, torch.dtype]:
        """Query-side geometry for the system decode-graph FlashInfer planner.

        KV-side geometry (heads / head dim / page size / dtype) is read off the
        shared pool by the system adapter; the query side comes from the MoT
        text expert (tensor-parallel-local head count, softmax scale, and the
        bf16 dtype ``project_qkv`` emits).
        """
        layer = cast(MoTDecoderLayer, self._ensure_loaded().model.lm.layers[0])
        return int(layer.n_heads), float(layer.scale), torch.bfloat16

    @staticmethod
    def _encoder_handle(mm_hash: Any) -> int:
        return encoder_handle_from_mm_hash(mm_hash)

    def encode_many(self, ops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not ops:
            return []
        loaded = self._ensure_loaded()
        m = loaded.model
        pool = loaded.pool
        image_processor = loaded.image_processor
        driver = self._text_driver()
        items: list[dict[str, Any]] = []
        feature_groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
        for raw_op in ops:
            op = dict(raw_op)
            req_id = int(op["req_id"])
            kind = str(op["kind"])
            if kind not in {"vae_encode", "vit_encode"}:
                raise invalid_descriptor(f"unsupported image encode kind: {kind}")
            state = self.program_state(req_id)
            driver.extend_cache_blocks(state.cond, op)
            driver.ensure_host_cache(state.cond)
            base_len = int(state.cond.past.length)
            item: dict[str, Any] = {
                "req_id": req_id,
                "kind": kind,
                "state": state,
                "base_len": base_len,
                "view": pool.view(state.cond.block_ids, base_len),
                "rope": int(op["cond_pos"]),
            }
            image_b64 = op.get("image_b64")
            if image_b64:
                preprocessed = image_processor.prepare_from_b64(image_b64)
                image_hw = [preprocessed.size[1], preprocessed.size[0]]
                handle = self._encoder_handle(op.get("mm_hash"))
                tensor = (
                    image_processor.vae_tensor(preprocessed)
                    if kind == "vae_encode"
                    else image_processor.vit_tensor(preprocessed)
                )
                item.update({"image_hw": image_hw, "handle": handle, "tensor": tensor})
                feature_groups.setdefault((kind, tuple(int(v) for v in tensor.shape)), []).append(
                    item
                )
            else:
                handle = op.get("image_in")
                if not isinstance(handle, int) or isinstance(handle, bool):
                    raise invalid_descriptor("cached image encode requires an encoder handle")
                payload = self.residency.encoder.get(handle)
                if not isinstance(payload, Mapping) or payload.get("kind") != kind:
                    raise invalid_descriptor("cached image encode handle is not resident")
                cached_image_hw = payload.get("image_hw")
                if (
                    not isinstance(cached_image_hw, list)
                    or len(cached_image_hw) != 2
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in cached_image_hw
                    )
                ):
                    raise invalid_descriptor("cached image encode dimensions are invalid")
                item.update(
                    {
                        "image_hw": [int(value) for value in cached_image_hw],
                        "handle": handle,
                        "payload": payload,
                    }
                )
            items.append(item)

        for (kind, _shape), group in feature_groups.items():
            tensors = torch.stack([item["tensor"] for item in group], dim=0)
            if kind == "vae_encode":
                clean_latents, position_ids, _ = m.vae_encode_clean_batch(tensors)
                for index, item in enumerate(group):
                    payload = {
                        "kind": kind,
                        "clean_lat": clean_latents[index].detach(),
                        "vpos": position_ids.detach(),
                        "image_hw": item["image_hw"],
                    }
                    item["payload"] = payload
                    self.residency.encoder.put(item["handle"], payload)
            else:
                embeddings = m.vit_encode_batch(tensors)
                for index, item in enumerate(group):
                    payload = {
                        "kind": kind,
                        "vemb": embeddings[index].detach(),
                        "image_hw": item["image_hw"],
                    }
                    item["payload"] = payload
                    self.residency.encoder.put(item["handle"], payload)

        segments: list[Segment] = []
        for item in items:
            kind = item["kind"]
            payload = item["payload"]
            if kind == "vae_encode":
                clean_lat = payload.get("clean_lat")
                position_ids = payload.get("vpos")
                if not isinstance(clean_lat, torch.Tensor) or not isinstance(
                    position_ids, torch.Tensor
                ):
                    raise invalid_descriptor("cached VAE output is incomplete")
                token_count = int(clean_lat.shape[0])
                segment = m.build_gen_segment(
                    token_count,
                    position_ids,
                    clean_lat,
                    0.0,
                    item["rope"],
                    item["view"],
                    update=True,
                )
            else:
                embeddings = payload.get("vemb")
                if not isinstance(embeddings, torch.Tensor):
                    raise invalid_descriptor("cached ViT output is incomplete")
                token_count = int(embeddings.shape[0])
                segment = m.build_und_image_segment(
                    embeddings,
                    item["rope"],
                    item["view"],
                    update=True,
                )
            item["added"] = token_count + _BAGEL_IMAGE_MARKER_TOKENS
            segments.append(segment)

        hidden_rows = m.run(segments)
        if len(hidden_rows) != len(items):
            raise invalid_descriptor("BAGEL batched image encode returned the wrong row count")
        outputs: list[dict[str, Any]] = []
        for item, hidden in zip(items, hidden_rows, strict=True):
            req_id = int(item["req_id"])
            added = int(item["added"])
            new_len = int(item["base_len"]) + added
            self._sync_text_cache_after_image(
                req_id,
                length=new_len,
                last_position=int(item["rope"]),
            )
            sampling = self._state(req_id).sampling
            if (
                sampling.get("return_prompt_logprobs")
                or int(sampling.get("n_prompt_logprobs", 0) or 0) > 0
            ):
                item["state"].cond.last_logits = m.logits(hidden[-1:]).unsqueeze(0)
            self._set_length(req_id, new_len)
            record = self._record(req_id)
            record["dims"] = item["image_hw"]
            if item["kind"] == "vit_encode":
                record["context_image_feedback"] = True
                record["text_branch_kvlen"] = new_len
                record["text_branch_pos"] = int(item["rope"]) + 1
            outputs.append(
                {
                    "req_id": req_id,
                    "encoder_handle": int(item["handle"]),
                    "num_tokens": added,
                    "image_hw": item["image_hw"],
                }
            )
        return outputs

    def run_encode(self, op: Mapping[str, Any]) -> dict[str, Any]:
        return self.encode_many((op,))[0]

    def prompt_predecessor_logits(self, req_id: int) -> torch.Tensor | None:
        return self.program_state(int(req_id)).cond.last_logits

    def encode_image(
        self,
        pixels: Any = None,
        grid: Any = None,
        *,
        op: Mapping[str, Any] | None = None,
    ) -> Any:
        del pixels, grid
        if op is None:
            raise invalid_descriptor("BAGEL image encode requires an op descriptor")
        return self.run_encode(dict(op))

    def encode_latents(
        self,
        pixels: Any = None,
        grid: Any = None,
        *,
        op: Mapping[str, Any] | None = None,
    ) -> Any:
        del pixels, grid
        if op is None:
            raise invalid_descriptor("BAGEL latent encode requires an op descriptor")
        return self.run_encode(dict(op))

    def run_text_logits_batch(self, ops):
        """Sequence prefill/decode through the system executor.

        Batching and the one-token decode CUDA graph are system-owned by the
        driver; BAGEL contributes only the MoT und-expert forward
        (``sequence_forward``). The host's ``pos_range`` stays
        authoritative for every op's rope position — matching the pre-driver
        Segment path, since BAGEL's 1-D positions do not advance across image
        spans the way KV length does — and the request-state KV-length mirror
        is refreshed from the text cache afterwards for the encode/denoise/
        commit paths that read it.
        """
        self._ensure_loaded()
        op_list = self._prepare_text_logits_batch(ops)
        out = self._text_driver().run_text_logits_batch(op_list)
        self._sync_text_cache_lengths(op_list)
        return out

    def try_run_graph_logits_batch(self, ops):
        """Return text logits only when the shared CUDA graph covers the batch."""
        self._ensure_loaded()
        op_list = self._prepare_text_logits_batch(ops)
        out = self._text_driver().try_run_graph_logits_batch(op_list)
        if out is None:
            return None
        self._sync_text_cache_lengths(op_list)
        return out

    def _prepare_text_logits_batch(self, ops):
        op_list = [dict(op) for op in ops]
        for op in op_list:
            st = self.program_state(int(op["req_id"]))
            pos_range = op.get("pos_range")
            if st.cond.past is not None and pos_range:
                st.cond.t_index = int(pos_range[0]) - 1
        return op_list

    def _sync_text_cache_lengths(self, ops) -> None:
        for op in ops:
            r = int(op["req_id"])
            st = self.program_state(r)
            if st.cond.past is not None:
                self._set_length(r, int(st.cond.past.length))

    def run_text_logits(self, op):
        return self.run_text_logits_batch([dict(op)])[0]

    def _init_generation_noise(
        self,
        shape: tuple[int, ...] | list[int],
        *,
        seed: int | None,
    ) -> torch.Tensor:
        """Sample BAGEL's request-private CPU-FP32 initial-noise stream."""
        effective_seed = int(seed if seed is not None else 0)
        return init_latent(
            shape,
            rng=torch.Generator(device="cpu").manual_seed(effective_seed),
            device=self.device,
            dtype=torch.float32,
            source_device="cpu",
            source_dtype=torch.float32,
        )

    def _init_gen(self, op):
        m = self._ensure_loaded().model
        state = self._state(int(op["req_id"]))
        rec = self._record(op["req_id"])
        ip = rec.get("image") or {}
        cfg = op.get("cfg") if isinstance(op.get("cfg"), dict) else {}
        cond_pos = int(op["cond_pos"])
        dims = rec.get("dims")
        parse_ip = dict(ip)
        if dims is not None:
            parse_ip["height"] = int(dims[0])
            parse_ip["width"] = int(dims[1])
        params = parse_text_image_generation_params(
            parse_ip,
            cfg=cfg,
            timestep_shift_default=m.cfg.timestep_shift,
        )
        height = int(params.height)
        width = int(params.width)
        h, w = m.latent_hw(height, width)
        num_vae = h * w
        vae_pos_ids = m.latent_position_ids(height, width).to(self.device)
        x_t = self._init_generation_noise(
            (num_vae, m.cfg.patch_latent_dim),
            seed=params.seed if params.seed is not None else state.seed,
        )
        if bool(rec.get("context_image_feedback")):
            raise capability_mismatch(
                "BAGEL context-image generation has no graph-ready CFG branch layout"
            )
        gs = GenState(
            x_t=x_t,
            vae_pos_ids=vae_pos_ids,
            num_vae=num_vae,
            H=height,
            W=width,
            schedule=FlowMatchSchedule(
                num_steps=int(params.steps),
                shift=float(params.timestep_shift),
                direction=ScheduleDirection.DESCENDING,
            ),
            cfg_text_scale=float(params.cfg_text),
            cfg_img_scale=float(params.cfg_img),
            cfg_renorm_type=str(params.cfg_norm),
            cfg_renorm_min=float(params.cfg_renorm_min),
            cfg_interval=(float(params.cfg_interval[0]), float(params.cfg_interval[1])),
            cond_pos=cond_pos,
            cond_branch_kvlen=cond_pos,
        )
        neg = list(rec.get("neg_token_ids") or [])
        gs.cfg_pos = len(neg) if (gs.cfg_text_scale > 1.0 and neg) else 0
        self._init_paged_denoise_branches(op, gs, neg)
        self._set_gen_state(op["req_id"], gs)

    def _init_paged_denoise_branches(self, op, gs: GenState, neg: list[int]) -> None:
        """Stage the t2i CFG branch prefixes into scratch-paged caches.

        The batched denoise path runs all active branches as rows of ONE
        gen-expert forward over a shared batched paged cache, which requires
        every row — the cond prefix included — to live in one KV pool, with
        each step's transient gen rows written past the row's fixed prefix.
        The request pool satisfies neither constraint (the neg branch cannot
        share it, and pure-t2i requests carry no host-allocated capacity for
        the ``num_vae + 2`` transient span), so the cond prefix KV
        ``[0, cond_kvlen)`` is copied once per image into the scratch pool —
        the same staging SenseNova's tower handoff performs on its
        trivial-tower path. The neg-prompt (text-uncond) branch prefills
        directly into scratch through the shared paged und-stack forward.

        Graph serving requires the branch caches to share one paged scratch
        pool. Missing capacity or backend support is a capability error.
        """
        if self.scratch_pool is None:
            raise capability_mismatch("BAGEL denoise requires a scratch KV pool")
        allocate_blocks = self.residency.allocator_for_pool(self.scratch_pool)
        if not callable(allocate_blocks):
            raise capability_mismatch("BAGEL denoise scratch KV allocation is unavailable")
        loaded = self._ensure_loaded()
        m = loaded.model
        pool = loaded.pool
        total_gen = int(gs.num_vae) + 2
        cond_len = int(gs.cond_branch_kvlen)
        caches: dict[str, PagedTextCache] = {}
        positions: dict[str, int] = {}
        try:
            cond_cache = PagedTextCache(
                self.scratch_pool,
                [],
                num_layers=self.num_layers,
                allocate_blocks=allocate_blocks,
            )
            caches["cond"] = cond_cache
            if not can_run_paged_denoise_attention(
                cond_cache,
                prototype=self.scratch_pool.k,
                query_width=self.cfg.llm.head_dim,
                query_tokens=total_gen,
                batch_size=1,
                attention_backend=self.attention_backend,
            ):
                raise capability_mismatch(
                    "BAGEL denoise attention backend does not support the graph row geometry"
                )
            cond_cache.ensure_capacity(cond_len + total_gen)
            if cond_len:
                copy_paged_text_cache_span(
                    pool.view(self._state(int(op["req_id"])).block_ids, cond_len),
                    cond_cache,
                    start=0,
                    length=cond_len,
                    num_layers=self.num_layers,
                )
            cond_cache.length = cond_len
            positions["cond"] = int(gs.cond_pos)
            if gs.cfg_img_scale > 1.0:
                caches["img_uncond"] = cond_cache
                positions["img_uncond"] = int(gs.cond_pos)
            if gs.cfg_text_scale > 1.0:
                tu_cache = PagedTextCache(
                    self.scratch_pool,
                    [],
                    num_layers=self.num_layers,
                    allocate_blocks=allocate_blocks,
                )
                caches["text_uncond"] = tu_cache
                tu_cache.ensure_capacity(len(neg) + total_gen)
                if neg:
                    ids = torch.as_tensor(neg, dtype=torch.long, device=self.device)
                    m.lm.forward_paged_text(
                        m.embed_tokens(ids).to(torch.bfloat16),
                        torch.arange(len(neg), device=self.device),
                        tu_cache,
                    )
                positions["text_uncond"] = int(gs.cfg_pos)
        except Exception as exc:
            released: set[int] = set()
            for cache in caches.values():
                if id(cache) in released:
                    continue
                released.add(id(cache))
                self.residency.release_scratch_cache(cache)
            if isinstance(exc, WorkerError):
                raise
            raise capability_mismatch("BAGEL denoise branch staging failed") from exc
        gs.paged_branches = PagedDenoiseBranchSet(caches=caches, positions=positions)
        gs.graph_image = _BagelDenoiseGraphImage(
            token_h=1,
            token_w=int(gs.num_vae) + _BAGEL_IMAGE_MARKER_TOKENS,
            height=int(gs.H),
            width=int(gs.W),
            cond_cache=caches.get("cond"),
            tu_cache=caches.get("text_uncond"),
            iu_cache=caches.get("img_uncond"),
            indexes={
                name: torch.stack(
                    (
                        torch.full(
                            (total_gen,),
                            int(positions[name]),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.zeros(total_gen, dtype=torch.long, device=self.device),
                        torch.zeros(total_gen, dtype=torch.long, device=self.device),
                    ),
                    dim=0,
                )
                for name in caches
            },
        )

    def prepare_flow_step(self, req_id: int, state, op: dict) -> PreparedFlowStep:
        r = int(req_id)
        self._extend_blocks(op)
        if int(op.get("timestep_idx") or 0) == 0 and self._gen_state(r) is None:
            self._init_gen(op)
        gs = self._gen_state(r)
        if gs is None:
            raise invalid_descriptor("denoise generation state is not initialized")
        i = int(op.get("timestep_idx") or 0)
        t, t_next = gs.schedule.pair(i, device=self.device, dtype=torch.float32)
        total_steps = int(gs.schedule.num_steps)
        if gs.paged_branches is None or gs.graph_image is None:
            raise capability_mismatch("BAGEL denoise requires graph-ready paged branch caches")
        image_embeds = (
            self._ensure_loaded()
            .model.gen_segment_embeds(
                int(gs.num_vae),
                gs.vae_pos_ids,
                gs.x_t,
                float(t.detach().float().item()),
            )
            .unsqueeze(0)
        )
        return PreparedFlowStep(
            req_id=r,
            state=state,
            op=op,
            latent=gs.x_t,
            t=t,
            t_next=t_next,
            step_index=i,
            total_steps=total_steps,
            cfg_text_scale=float(gs.cfg_text_scale),
            cfg_img_scale=float(gs.cfg_img_scale),
            cfg_interval=(float(gs.cfg_interval[0]), float(gs.cfg_interval[1])),
            cfg_renorm_type=str(gs.cfg_renorm_type),
            cfg_renorm_min=float(gs.cfg_renorm_min),
            cfg_branch_count=flow_cfg_branch_count(op),
            image_scale_applies_to_text=CfgRecipe.IMAGE_OVER_TEXT,
            extra={"gs": gs, "img": gs.graph_image, "image_embeds": image_embeds},
        )

    def prepare_flow(
        self,
        state: Any,
        op: Mapping[str, Any],
    ) -> PreparedFlowStep:
        return self.prepare_flow_step(int(op["req_id"]), state, dict(op))

    def predict_velocity(
        self,
        ctx: PreparedFlowStep,
        t: torch.Tensor,
        latent: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        del t, latent
        return self.predict_flow_velocity(ctx, branch)

    def _denoise_branch_inputs(
        self,
        image: _BagelDenoiseGraphImage,
        branch: Any,
    ) -> tuple[torch.Tensor, PagedTextCache]:
        name = str(getattr(branch, "value", branch))
        cache_by_name = {
            "cond": image.cond_cache,
            "text_uncond": image.tu_cache,
            "img_uncond": image.iu_cache,
        }
        cache = cache_by_name.get(name)
        indexes = image.indexes.get(name)
        if cache is None or indexes is None:
            raise invalid_descriptor(f"BAGEL denoise branch {name!r} is not initialized")
        return indexes, cache

    def predict_flow_velocity_batch(
        self,
        steps,
        branches_by_step,
        *,
        graph_mode: str = "auto",
    ):
        """Execute compatible request and CFG rows through CUDA graphs."""
        self._ensure_loaded()
        if graph_mode == "eager":
            raise capability_mismatch("BAGEL denoise execution requires CUDA graphs")
        graphed = self._predict_flow_velocity_graph(steps, branches_by_step)
        if graphed is None and graph_mode == "require":
            return None
        if graphed is None:
            raise capability_mismatch("BAGEL denoise CUDA graph did not cover the batch")
        return graphed

    def _predict_flow_velocity_graph(
        self,
        steps,
        branches_by_step,
    ) -> list[dict[str, torch.Tensor]] | None:
        m = self._ensure_loaded().model
        results: list[dict[str, torch.Tensor]] = [dict() for _ in steps]
        groups: dict[tuple[Any, ...], list[tuple[int, str, FlowRow]]] = {}
        for step_index, (step, branches) in enumerate(zip(steps, branches_by_step, strict=True)):
            gs = step.extra["gs"]
            paged_branches = gs.paged_branches
            graph_image = getattr(gs, "graph_image", None)
            names = tuple(str(getattr(branch, "value", branch)) for branch in branches)
            if paged_branches is None or graph_image is None or not paged_branches.has_all(names):
                return None
            total = int(gs.num_vae) + _BAGEL_IMAGE_MARKER_TOKENS
            embeds = m.gen_segment_embeds(
                int(gs.num_vae),
                gs.vae_pos_ids,
                step.latent,
                float(step.t.detach().float().item()),
            )
            positions = paged_branches.positions_tensor(
                names,
                device=self.device,
                width=total,
            )
            step.extra["image_embeds"] = embeds.unsqueeze(0)
            step.extra["img"] = graph_image
            first_cache = paged_branches.caches[names[0]]
            signature = (
                id(first_cache.pool),
                tuple(int(dim) for dim in embeds.shape),
                str(embeds.device),
                str(embeds.dtype),
                tuple(int(dim) for dim in step.latent.shape),
                str(step.latent.device),
                str(step.latent.dtype),
                tuple(int(dim) for dim in positions.shape[1:]),
                str(positions.dtype),
            )
            group = groups.setdefault(signature, [])
            for branch_index, name in enumerate(names):
                group.append(
                    (
                        step_index,
                        name,
                        FlowRow(
                            step_index=step_index,
                            step=step,
                            branch=name,
                            img=graph_image,
                            indexes=positions[branch_index],
                            cache=paged_branches.caches[name],
                        ),
                    )
                )

        for group in groups.values():
            rows = [entry[2] for entry in group]
            group_num_vae = int(rows[0].img.token_w) - _BAGEL_IMAGE_MARKER_TOKENS
            m.gen_segment_graph_layout(len(rows), group_num_vae)
            velocity = self.flow_graph_execution.maybe_run_graph(rows)
            if not isinstance(velocity, torch.Tensor):
                return None
            if int(velocity.shape[0]) != len(group):
                raise invalid_descriptor("BAGEL denoise graph rows must align with CFG rows")
            for row_index, (step_index, name, _row) in enumerate(group):
                results[step_index][name] = velocity[row_index]
        return results

    def flow_predict_velocity(
        self,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: Any,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int],
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del attention_mask, t, z, image_size
        m = self._ensure_loaded().model
        num_vae = int(image_token_num) - _BAGEL_IMAGE_MARKER_TOKENS
        if num_vae <= 0:
            raise invalid_descriptor("BAGEL denoise graph requires latent tokens")
        if indexes.ndim != 2:
            raise invalid_descriptor("BAGEL denoise graph positions must be token-by-row")
        positions = indexes.transpose(0, 1).contiguous()
        is_gen, text_idx = m.gen_segment_graph_layout(int(image_embeds.shape[0]), num_vae)
        hidden = m.lm.forward_paged_gen_batch(
            image_embeds,
            positions,
            is_gen,
            cache,
            text_idx=text_idx,
        )
        velocity = m.llm2vae(hidden[:, 1 : 1 + num_vae].to(torch.bfloat16))
        if return_hidden:
            return velocity, hidden
        return velocity

    def predict_flow_velocity(self, step: PreparedFlowStep, branch: str) -> torch.Tensor:
        result = self.predict_flow_velocity_batch(
            [step],
            [(branch,)],
            graph_mode="require",
        )
        if result is None:
            raise capability_mismatch("BAGEL denoise CUDA graph did not cover the branch")
        return result[0][branch]

    def apply_flow_update(self, step: PreparedFlowStep, latent: torch.Tensor) -> None:
        gs = step.extra["gs"]
        gs.x_t = latent.to(dtype=gs.x_t.dtype, device=gs.x_t.device)

    def accept_flow_update(self, ctx: PreparedFlowStep, latent: torch.Tensor) -> None:
        self.apply_flow_update(ctx, latent)

    def _sync_text_cache_after_image(self, req_id: int, *, length: int, last_position: int) -> None:
        """Advance the driver's text cache past an image KV span written outside it.

        Encode/commit write image KV through raw pool views; the shared text
        driver must resume text from the post-image KV length and rope
        position (block ids are already shared, so only length/position move).
        """
        st = self.program_state(int(req_id))
        self._text_driver().ensure_host_cache(st.cond)
        st.cond.past.length = int(length)
        st.cond.t_index = int(last_position)

    def commit_generated_image(self, req_id: int, state, op) -> dict:
        return self._commit_generated_image(dict(op))

    def decode_image(
        self,
        latent: Any,
        *,
        req_id: int | None = None,
        state: Any = None,
        op: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del latent, state
        if req_id is None or op is None:
            raise invalid_descriptor("BAGEL image commit requires req_id and op")
        return self._commit_generated_image(dict(op))

    def _commit_generated_image(self, op):
        loaded = self._ensure_loaded()
        m = loaded.model
        pool = loaded.pool
        r = op["req_id"]
        gs = self._gen_state(r)
        if gs is None:
            return {"req_id": r}
        self._extend_blocks(op)
        img = m.vae_decode(gs.x_t, gs.H, gs.W)
        rec = self._record(r)
        retain_images = bool((rec.get("image") or {}).get("retain_images", True))
        added = 0
        if retain_images:
            # Program continuation: persist the generated latents into the
            # request KV so following text conditions on the image. The engine
            # allocates these blocks only when retention is requested (pure
            # image mode ends at the commit and skips both).
            view = pool.view(self._state(r).block_ids, gs.cond_pos)
            commit_seg = m.build_gen_segment(
                gs.num_vae, gs.vae_pos_ids, gs.x_t, 0.0, gs.cond_pos, view, update=True
            )
            m.run([commit_seg])
            self._set_length(r, gs.cond_pos + gs.num_vae + 2)
            # gen_rope_advance=2: following text continues at cond_pos + 2.
            self._sync_text_cache_after_image(
                r, length=gs.cond_pos + gs.num_vae + 2, last_position=gs.cond_pos + 1
            )
            added = gs.num_vae + 2
        b64 = pil_image_to_png_b64(img)
        self._pop_gen_state(r)
        return {
            "req_id": r,
            "image_png_b64": b64,
            "image_hw": [gs.H, gs.W],
            "num_tokens": added,
        }

    @torch.no_grad()
    def forward(
        self, batch: ForwardBatch
    ) -> Any:
        from ..contracts.forward_context import get_forward_context

        self._ensure_loaded()
        options = get_forward_context().execution_options
        with self._autocast():
            return self.segment_executor.execute(
                batch,
                request_states=self.states,
                defer_text_cpu_results=bool(
                    getattr(options, "defer_text_cpu_results", False)
                ),
            )

    @torch.no_grad()
    def forward_text(self, batch: ForwardBatch) -> torch.Tensor:
        self._ensure_loaded()
        if len(batch.ops) != 1:
            raise invalid_descriptor("BAGEL tensor text forward requires one operation")
        with self._autocast():
            logits = self.run_text_logits(dict(batch.ops[0]))
        if not isinstance(logits, torch.Tensor):
            raise invalid_descriptor("BAGEL text forward must return logits")
        return logits

    def _ensure_loaded(self) -> _LoadedBagelRuntime:
        model = self.model
        pool = self.pool
        image_processor = self.image_processor
        if model is None or pool is None or image_processor is None:
            raise capability_mismatch("BAGEL model weights are not loaded")
        return _LoadedBagelRuntime(
            model=model,
            pool=pool,
            image_processor=image_processor,
        )

    def _autocast(self):
        self._ensure_loaded()
        device = str(self.device)
        if device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()


EntryClass = BagelForUnifiedGeneration


# ---------------------
# Text-image generation workflow session (family-owned request lifecycle)
# ---------------------


class TextImageGenerationSession:
    """Owns request lifecycle state for text-image generation flows."""

    def __init__(self, owner: Any, req_id: int | None = None) -> None:
        self.owner = owner
        self.req_id = None if req_id is None else int(req_id)
        self.records: dict[int, dict[str, Any]] = {}
        self.generated: dict[int, Any] = {}

    def begin_request(
        self,
        req_id: int,
        *,
        sampling: dict[str, Any] | None = None,
        image: dict[str, Any] | None = None,
        neg_token_ids: list[int] | None = None,
        lora_id: Any = None,
    ) -> dict[str, Any]:
        record = {
            "sampling": dict(sampling or {}),
            "image": dict(image or {}),
            "neg_token_ids": list(neg_token_ids or []),
            "lora_id": lora_id,
            "dims": None,
        }
        self.records[int(req_id)] = record
        self.release_generated_state(req_id)
        return record

    def record(self, req_id: int) -> dict[str, Any]:
        return self.records.setdefault(int(req_id), {})

    def generation_state(self, req_id: int) -> Any | None:
        return self.generated.get(int(req_id))

    def set_generation_state(self, req_id: int, value: Any) -> None:
        self.generated[int(req_id)] = value

    def release_generated_state(self, req_id: int) -> Any | None:
        state = self.generated.pop(int(req_id), None)
        if state is not None:
            self.release_paged_branches(state)
        return state

    def release_request(self, req_id: int, *, request_state: Any | None = None) -> None:
        target = int(req_id)
        self.release_generated_state(target)
        self.records.pop(target, None)
        if request_state is not None:
            request_state.kv_lengths.pop("default", None)

    def release_paged_branches(self, state: Any) -> None:
        branches = getattr(state, "paged_branches", None)
        if branches is None:
            return
        branches.release(self.owner.residency)
        state.paged_branches = None

    def start(self, op: dict[str, Any]) -> Any:
        state = getattr(self.owner, "_state", None)
        if callable(state):
            return state(int(op["req_id"]))
        return None

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        encode = getattr(self.owner, "encode_image", None)
        if callable(encode):
            return encode(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose image encoding")

    def prepare_flow(self, state: Any, op: dict[str, Any]) -> Any:
        prepare = getattr(self.owner, "prepare_flow", None)
        if callable(prepare):
            return prepare(state, op)
        raise RuntimeError("generation session owner does not expose denoise preparation")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        predict = getattr(self.owner, "predict_velocity", None)
        if callable(predict):
            return predict(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose denoise prediction")

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        decode = getattr(self.owner, "decode_image", None)
        if callable(decode):
            return decode(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose image commit")

    def release(self, req_id: int | None = None) -> None:
        target = self.req_id if req_id is None else int(req_id)
        if target is not None:
            self.release_request(target)
