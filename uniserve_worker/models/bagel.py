"""BAGEL UniModel entry backed by the shared runner."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..contracts.batches import UniForwardBatch
from ..contracts.resource_plan import (
    AdapterResourcePolicy,
    CapsDescriptor,
    EncoderResourcePolicy,
    KvBlockResourcePolicy,
    LatentTokens,
    PerBranch,
    ResourcePlan,
)
from ..execution.denoise_driver import TextImageDenoiseStep, text_image_cfg_branch_count
from ..execution.interleaved_text_stepper import InterleavedTextCacheDriver, TextCache
from ..execution.model_base import UniModelBase
from ..execution.paged_denoise import PagedDenoiseBranchSet, can_run_paged_denoise_attention
from ..execution.text_image_generation_session import TextImageGenerationSession
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import get_worker_config
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
    patchify,
)
from ..processors.bagel import BagelImageProcessor
from ..runtime.image_params import parse_text_image_generation_params
from ..runtime.image_utils import pil_image_to_png_b64
from ..runtime.kv_pool import PagedKVPool
from ..runtime.lora import MergeOnLoadLoRA
from ..runtime.paged_text_cache import (
    PagedTextCache,
    copy_paged_text_cache_span,
)
from ..runtime.request_state import RequestState
from ..runtime.residency import (
    GenResidencySpec,
    KvCacheSpec,
    ResidencyManager,
    encoder_handle_from_mm_hash,
)

__all__ = [
    'LLMConfig',
    'BagelConfig',
    'GenState',
    'BagelTextRequestState',
    'BagelForUnifiedGeneration',
    'EntryClass',
]

logger = logging.getLogger(__name__)

_BAGEL_RMS_NORM_EPS = 1e-6
_BAGEL_ROPE_THETA = 1_000_000.0
_BAGEL_VIT_LAYER_NORM_EPS = 1e-6


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
            self.latent_pos_embed = PositionEmbedding(cfg.max_latent_size, hidden, init_sincos=False)
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
            self.vit_pos_embed = PositionEmbedding(cfg.vit_max_num_patch_per_side, hidden, init_sincos=False)

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

        Shared by the eager per-branch :meth:`build_gen_segment` path and the
        batched-rows denoise path so the two are byte-identical by construction.
        """
        hidden = self.cfg.llm.hidden_size
        total = int(num_vae) + 2
        marker_ids = torch.tensor(
            [self.cfg.start_of_image_id, self.cfg.end_of_image_id],
            dtype=torch.long,
            device=self.device,
        )
        marker_emb = self.embed_tokens(marker_ids).to(torch.bfloat16)
        x_t = x_t.to(self.device)
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

    def build_gen_segment(self, num_vae, vae_pos_ids, x_t, timestep, position_id, cache, update=False) -> Segment:
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
        image_tensor = image_tensor.to(self.device)
        height, width = image_tensor.shape[1], image_tensor.shape[2]
        patch = self.cfg.vit_patch_size
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            patch,
            self.cfg.vit_max_num_patch_per_side,
        ).to(self.device)
        patches = patchify(image_tensor, patch).to(self.device, torch.bfloat16)
        cu_seqlens = torch.tensor([0, patches.shape[0]], dtype=torch.int32, device=self.device)
        vit_out = self.vit_model(
            patches,
            {"position_ids": pos_ids, "cu_seqlens": cu_seqlens},
        )
        emb = self.connector(vit_out) + self.vit_pos_embed(pos_ids)
        return emb.to(torch.bfloat16)

    @torch.no_grad()
    def vae_encode_clean(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.to(self.device)
        height, width = image_tensor.shape[1], image_tensor.shape[2]
        vae_dtype = next(self.vae.parameters()).dtype
        latent_image = self.vae.encode(image_tensor.unsqueeze(0).to(vae_dtype))[0]
        patch = self.cfg.latent_patch_size
        channels = self.cfg.latent_channel
        h = height // self.cfg.latent_downsample
        w = width // self.cfg.latent_downsample
        latent = latent_image[:, : h * patch, : w * patch].reshape(channels, h, patch, w, patch)
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, patch * patch * channels)
        pos_ids = get_flattened_position_ids_extrapolate(
            height,
            width,
            self.cfg.latent_downsample,
            self.cfg.max_latent_size,
        ).to(self.device)
        return latent.to(torch.bfloat16), pos_ids, (h, w)

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
    """Mutable image-generation state for one BAGEL denoise/commit cycle.

    ``paged_branches`` (when set) holds scratch-paged CFG branch prefixes and
    their batched-row cache memo. ``None`` keeps the per-branch eager segment
    path (context-image feedback, no scratch pool, or an ineligible attention
    backend).
    """

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
    uses_context_image_feedback: bool
    cond_branch_kvlen: int = 0
    text_branch_pos: int = 0
    text_branch_kvlen: int = 0
    cfg_cache: KVCache | None = None
    cfg_pos: int = 0
    cfg_img_cache: KVCache | None = None
    cfg_img_pos: int = 0
    paged_branches: PagedDenoiseBranchSet | None = None


# BAGEL's start-of-image marker string (token id 151652 in the Qwen2 vocab).
# The worker never tokenizes it (encode ops embed the marker directly); the
# shared text driver takes it as configuration.
_BAGEL_IMG_START_TOKEN = "<|vision_start|>"

# Scratch (per-CFG-branch) KV pool capacity in tokens. Sized for several
# concurrent denoises: each t2i image stages cond + text-uncond prefixes plus
# one transient gen span (``num_vae + 2`` <= 4098 at the 64x64 latent cap) per
# branch, ~9k tokens for a 1024x1024 image. The caps descriptor advertises
# exactly this pool's capacity.
_BAGEL_SCRATCH_CAPACITY_TOKENS = 65536


@dataclass
class BagelTextRequestState:
    """Per-request text-cache state for the shared interleaved text driver.

    ``cond`` is the single conditional text branch; its ``block_ids`` list is
    shared (same object) with the runner ``RequestState.block_ids`` so the
    driver's text ops and BAGEL's encode/denoise/commit ops ingest host
    ``new_block_ids`` into one list and every path sees every block.
    """

    cond: TextCache = field(default_factory=TextCache)


class BagelForUnifiedGeneration(UniModelBase):
    """BAGEL unified text/image model with VAE denoise and ViT/VAE encode paths."""

    architectures = ("BagelForUnifiedGeneration", "BAGEL", "bagel")
    supported_ops = (
        "prefill_und",
        "decode_und",
        "denoise_gen",
        "commit_gen",
        "vit_encode",
        "vae_encode",
    )
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
    ENCODER_CACHE_BUDGET = 256

    @classmethod
    def recognizes(cls, model_path: str | Path) -> bool:
        root = Path(model_path)
        return (
            (root / "ema.safetensors").exists() or (root / "model.safetensors").exists()
        ) and (root / "ae.safetensors").exists()

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
        self.cfg = model.cfg if model is not None else (
            config if isinstance(config, BagelConfig) else BagelConfig()
        )
        self.resource_plan = self._build_resource_plan()
        self.image_processor = image_processor or (BagelImageProcessor() if model is not None else None)
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
        self._shared_text_driver: InterleavedTextCacheDriver | None = None
        self.pool: PagedKVPool | None = None
        self.residency = ResidencyManager()
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
                    memory_fraction=get_worker_config().kv_memory_fraction,
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
            self.num_blocks = derive_num_blocks(
                self.block_size, self.kv_token_capacity, floor=64
            )

    def _build_resource_plan(self) -> ResourcePlan:
        return ResourcePlan(
            kv_block=KvBlockResourcePolicy.PER_BLOCK,
            encoder_output=EncoderResourcePolicy.PER_HANDLE,
            image_latent=LatentTokens(downsample=int(self.cfg.latent_downsample)),
            scratch=PerBranch(),
            adapter=AdapterResourcePolicy.PER_ADAPTER,
        )

    def _scratch_num_blocks(self, block_size: int | None = None) -> int:
        return ceil_div(_BAGEL_SCRATCH_CAPACITY_TOKENS, int(block_size or self.block_size))

    def _build_residency(self, cfg: LLMConfig) -> ResidencyManager:
        # System-owned residency: the worker-owned ResidencyManager constructs
        # and owns the KV pool; the model declares only geometry/sizing here.
        # The scratch pool holds the denoise CFG-branch prefix KV (staged cond
        # + neg-prompt branches) and their per-step transient gen rows.
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
            num_blocks = int(self.num_blocks)
        else:
            num_blocks = derive_num_blocks(block, cap, floor=64)
        c = self.cfg.llm
        # Report the actual per-CFG-branch scratch pool capacity in tokens
        # (the built pool's blocks when loaded, the planned sizing otherwise).
        scratch_blocks = self._scratch_blocks if self.model is not None else self._scratch_num_blocks(block)
        return CapsDescriptor(
            block_size=block,
            num_blocks=num_blocks,
            num_layers=c.num_hidden_layers,
            scratch_capacity_tokens=int(scratch_blocks * block),
            max_latent_size=self.cfg.latent_token_capacity,
            latent_downsample=self.cfg.latent_downsample,
            max_vae_grid_tokens=self.cfg.latent_token_capacity,
            commit_marker_tokens=2,
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
        self.interleaved_image_state(r)
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
        self.reqs.pop(r, None)
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
        self.generation_session.release_paged_branches(gs)

    def _extend_blocks(self, op) -> list[int]:
        state = self._state(int(op["req_id"]))
        state.append_new_block_ids(op.get("new_block_ids"))
        return state.block_ids

    # ---- shared interleaved text driver (owner surface) --------------------

    @property
    def kv_pool(self) -> PagedKVPool | None:
        # The interleaved text driver's name for the request KV pool.
        return self.pool

    @property
    def num_layers(self) -> int:
        return int(self.cfg.llm.num_hidden_layers)

    def _text_driver(self) -> InterleavedTextCacheDriver:
        driver = self._shared_text_driver
        if driver is None:
            driver = InterleavedTextCacheDriver(
                self,
                request_state_factory=BagelTextRequestState,
                image_start_token=_BAGEL_IMG_START_TOKEN,
            )
            self._shared_text_driver = driver
        return driver

    def interleaved_image_state(self, req_id: int) -> BagelTextRequestState:
        req_id = int(req_id)
        st = self.reqs.get(req_id)
        if st is None:
            st = BagelTextRequestState()
            # Share one block list between the driver's text cache and the
            # runner request state (see BagelTextRequestState docstring).
            st.cond.block_ids = self._state(req_id).block_ids
            self.reqs[req_id] = st
        return st

    def interleaved_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._ensure_loaded().model.embed_tokens(input_ids).to(torch.bfloat16)

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
        return_all_logits: bool = False,
    ) -> CausalLMOutputWithPast:
        """Run the MoT understanding expert stack over the shared paged text KV.

        The interleaved text driver only ever sends strictly-causal pure-text
        spans here (BAGEL is 1-D rope, so only ``indexes[0]`` is consumed; the
        spatial rows are zero). Its block-causal / past-visible masks are
        therefore equivalent to plain causal attention — exactly what the
        paged extend/decode path computes and what the eager
        ``Segment(causal=True)`` path this replaces computed — so
        ``attention_mask`` is intentionally not materialized.
        """
        del attention_mask, use_cache, text_only_rope, causal_paged_update
        m = self._ensure_loaded().model
        if input_ids is None and inputs_embeds is None:
            raise invalid_descriptor(
                "BAGEL interleaved text forward requires exactly one of input_ids or inputs_embeds"
            )
        if input_ids is not None and inputs_embeds is not None:
            raise invalid_descriptor(
                "BAGEL interleaved text forward requires exactly one of input_ids or inputs_embeds"
            )
        if indexes is None and cache_position is not None:
            indexes = cache_position.reshape(1, -1)
        if indexes is None or past_key_values is None:
            raise invalid_descriptor(
                "BAGEL interleaved text forward requires indexes and a paged cache"
            )
        if inputs_embeds is None:
            if input_ids is None:
                raise invalid_descriptor("BAGEL text input ids are missing")
            inputs_embeds = m.embed_tokens(input_ids).to(torch.bfloat16)
        batch, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        hidden = m.lm.forward_paged_text(
            inputs_embeds.reshape(batch * seq_len, -1),
            indexes[0].reshape(-1),
            past_key_values,
        )
        hidden = hidden.view(batch, seq_len, -1)
        logits = (
            m.logits(hidden.reshape(batch * seq_len, -1)).view(batch, seq_len, -1)
            if return_all_logits
            else m.logits(hidden[:, -1, :]).unsqueeze(1)
        )
        return CausalLMOutputWithPast(
            logits=cast(Any, logits),
            past_key_values=past_key_values,
        )

    def text_decode_graph_query_geometry(self) -> tuple[int, float, torch.dtype]:
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

    def run_encode(self, op):
        loaded = self._ensure_loaded()
        m = loaded.model
        pool = loaded.pool
        image_processor = loaded.image_processor
        r = int(op["req_id"])
        # Encode writes image KV into the same paged text cache the shared
        # driver serves text from: extend blocks / base length through the
        # driver's TextCache so both paths agree on one residency state.
        driver = self._text_driver()
        st = self.interleaved_image_state(r)
        driver.extend_cache_blocks(st.cond, op)
        driver.ensure_host_cache(st.cond)
        base_len = int(st.cond.past.length)
        view = pool.view(st.cond.block_ids, base_len)
        rope = int(op["cond_pos"])
        kind = str(op["kind"])
        image_b64 = op.get("image_b64")
        if image_b64:
            pre = image_processor.prepare_from_b64(image_b64)
            image_hw = [pre.size[1], pre.size[0]]
            handle = self._encoder_handle(op.get("mm_hash"))
            if kind == "vae_encode":
                clean_lat, vpos, _ = m.vae_encode_clean(image_processor.vae_tensor(pre))
                cached_payload = {
                    "kind": kind,
                    "clean_lat": clean_lat.detach(),
                    "vpos": vpos.detach(),
                    "image_hw": image_hw,
                }
            elif kind == "vit_encode":
                vemb = m.vit_encode(image_processor.vit_tensor(pre)).detach()
                cached_payload = {
                    "kind": kind,
                    "vemb": vemb,
                    "image_hw": image_hw,
                }
            else:
                raise invalid_descriptor(f"unsupported image encode kind: {kind}")
            self.residency.encoder.put(handle, cached_payload)
        else:
            cached_handle = op.get("image_in")
            if not isinstance(cached_handle, int) or isinstance(cached_handle, bool):
                raise invalid_descriptor("cached image encode requires an encoder handle")
            cached_payload = self.residency.encoder.get(cached_handle)
            if not isinstance(cached_payload, Mapping) or cached_payload.get("kind") != kind:
                raise invalid_descriptor("cached image encode handle is not resident")
            cached_image_hw = cached_payload.get("image_hw")
            if (
                not isinstance(cached_image_hw, list)
                or len(cached_image_hw) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in cached_image_hw
                )
            ):
                raise invalid_descriptor("cached image encode dimensions are invalid")
            image_hw = [int(value) for value in cached_image_hw]
            handle = cached_handle

        if kind == "vae_encode":
            clean_lat = cached_payload.get("clean_lat")
            vpos = cached_payload.get("vpos")
            if not isinstance(clean_lat, torch.Tensor) or not isinstance(vpos, torch.Tensor):
                raise invalid_descriptor("cached VAE output is incomplete")
            n = clean_lat.shape[0]
            seg = m.build_gen_segment(n, vpos, clean_lat, 0.0, rope, view, update=True)
            hidden = m.run([seg])[0]
            added = n + 2
        elif kind == "vit_encode":
            cached_vemb = cached_payload.get("vemb")
            if not isinstance(cached_vemb, torch.Tensor):
                raise invalid_descriptor("cached ViT output is incomplete")
            vemb = cached_vemb
            n = vemb.shape[0]
            hidden = m.run([m.build_und_image_segment(vemb, rope, view, update=True)])[0]
            added = n + 2
        else:
            raise invalid_descriptor(f"unsupported image encode kind: {kind}")
        new_len = base_len + added
        # The image span consumed `added` KV slots but a single rope position;
        # advance the driver's text cache for both so following text ops
        # continue from the right KV length and position.
        self._sync_text_cache_after_image(r, length=new_len, last_position=rope)
        sampling = self._state(r).sampling
        if (
            sampling.get("return_prompt_logprobs")
            or int(sampling.get("n_prompt_logprobs", 0) or 0) > 0
        ):
            st.cond.last_logits = m.logits(hidden[-1:]).unsqueeze(0)
        self._set_length(r, new_len)
        rec = self._record(r)
        rec["dims"] = image_hw
        if kind == "vit_encode":
            rec["context_image_feedback"] = True
            rec["text_branch_kvlen"] = new_len
            rec["text_branch_pos"] = rope + 1
        return {"req_id": r, "encoder_handle": handle, "num_tokens": added,
                "image_hw": image_hw}

    def prompt_predecessor_logits(self, req_id: int) -> torch.Tensor | None:
        return self.interleaved_image_state(int(req_id)).cond.last_logits

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
        """Text prefill/decode through the shared interleaved text driver.

        Batching and the one-token decode CUDA graph are system-owned by the
        driver; BAGEL contributes only the MoT und-expert forward
        (``interleaved_text_forward``). The host's ``pos_range`` stays
        authoritative for every op's rope position — matching the pre-driver
        Segment path, since BAGEL's 1-D positions do not advance across image
        spans the way KV length does — and the request-state KV-length mirror
        is refreshed from the text cache afterwards for the encode/denoise/
        commit paths that read it.
        """
        self._ensure_loaded()
        op_list = [dict(op) for op in ops]
        for op in op_list:
            st = self.interleaved_image_state(int(op["req_id"]))
            pos_range = op.get("pos_range")
            if st.cond.past is not None and pos_range:
                st.cond.t_index = int(pos_range[0]) - 1
        out = self._text_driver().run_text_logits_batch(op_list)
        for op in op_list:
            r = int(op["req_id"])
            st = self.interleaved_image_state(r)
            if st.cond.past is not None:
                self._set_length(r, int(st.cond.past.length))
        return out

    def run_text_logits(self, op):
        return self.run_text_logits_batch([dict(op)])[0]

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
        g = state.device_rng(self.device)
        x_t = init_latent(
            (num_vae, m.cfg.patch_latent_dim),
            rng=g,
            device=self.device,
            dtype=torch.bfloat16,
        )
        uses_context_image_feedback = bool(rec.get("context_image_feedback"))
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
            uses_context_image_feedback=uses_context_image_feedback,
        )
        if gs.uses_context_image_feedback:
            gs.cond_branch_kvlen = int(self._length(op["req_id"]) or gs.cond_pos)
            gs.text_branch_pos = int(rec["text_branch_pos"])
            gs.text_branch_kvlen = int(rec["text_branch_kvlen"])
            gs.cfg_img_cache = m.new_cache()
            neg = rec.get("neg_token_ids") or []
            m.run([m.build_und_segment(neg, 0, gs.cfg_img_cache, update=True)])
            gs.cfg_img_pos = len(neg)
            gs.cfg_cache = None
            gs.cfg_pos = 0
        else:
            gs.cond_branch_kvlen = gs.cond_pos
            gs.text_branch_pos = gs.text_branch_kvlen = 0
            gs.cfg_img_cache = None
            gs.cfg_img_pos = 0
            gs.cfg_cache = None
            neg = list(rec.get("neg_token_ids") or [])
            gs.cfg_pos = len(neg) if (gs.cfg_text_scale > 1.0 and neg) else 0
            self._init_paged_denoise_branches(op, gs, neg)
            if gs.paged_branches is None:
                # Eager per-branch fallback: transient in-RAM KV branches
                # driven through the segment API (pre-scratch-pool semantics).
                gs.cfg_cache = m.new_cache()
                if gs.cfg_text_scale > 1.0 and neg:
                    m.run([m.build_und_segment(neg, 0, gs.cfg_cache, update=True)])
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

        Any ineligibility (no scratch pool, context-image feedback branches,
        quantized KV store, a backend without paged attention, or scratch
        exhaustion) leaves ``gs.paged_branches`` ``None`` and the eager
        per-branch segment path fully authoritative.
        """
        if self.scratch_pool is None or gs.uses_context_image_feedback:
            return
        allocate_blocks = self.residency.allocator_for_pool(self.scratch_pool)
        if not callable(allocate_blocks):
            return
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
                attention_backend=self.attention_backend,
            ):
                self.residency.release_scratch_cache(cond_cache)
                return
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
        except RuntimeError:
            # Worker-local scratch exhaustion is recoverable: release and keep
            # the eager path authoritative.
            for cache in caches.values():
                self.residency.release_scratch_cache(cache)
            logger.warning(
                "BAGEL scratch staging for denoise CFG branches failed; "
                "falling back to the per-branch segment path",
                exc_info=True,
            )
            return
        gs.paged_branches = PagedDenoiseBranchSet(caches=caches, positions=positions)

    def prepare_denoise_step(self, req_id: int, state, op: dict) -> TextImageDenoiseStep:
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
        return TextImageDenoiseStep(
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
            cfg_branch_count=text_image_cfg_branch_count(op),
            image_scale_applies_to_text=CfgRecipe.IMAGE_OVER_TEXT,
            extra={"gs": gs},
        )

    def prepare_denoise(
        self,
        state: Any,
        op: Mapping[str, Any],
    ) -> TextImageDenoiseStep:
        return self.prepare_denoise_step(int(op["req_id"]), state, dict(op))

    def predict_velocity(
        self,
        ctx: TextImageDenoiseStep,
        t: torch.Tensor,
        latent: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        del t, latent
        return self.predict_denoise_velocity(ctx, branch)

    def predict_text_image_velocity_batch(self, steps, branches_by_step):
        """Batched denoise: all CFG branches of a step as rows of ONE gen forward.

        The denoise driver prefers this over per-branch ``predict_velocity``
        calls. Steps whose branches are not fully staged in scratch-paged
        caches (context-image feedback, or no scratch/backend support) run
        the per-branch path with its exact existing semantics.
        """
        self._ensure_loaded()
        results: list[dict[str, torch.Tensor]] = []
        for step, branches in zip(steps, branches_by_step):
            gs = step.extra["gs"]
            paged_branches = gs.paged_branches
            names = tuple(branches)
            if paged_branches is not None and paged_branches.has_all(names):
                results.append(self._predict_denoise_velocity_rows(step, gs, names))
            else:
                results.append(
                    {branch: self.predict_denoise_velocity(step, branch) for branch in names}
                )
        return results

    def _predict_denoise_velocity_rows(
        self, step: TextImageDenoiseStep, gs: GenState, branches: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        """Run the given CFG branches as rows of one batched MoT gen forward.

        Per-row math matches the per-branch ``build_gen_segment`` path exactly
        (shared embedding builder, same modality routing / rope / bidirectional
        visibility over [branch prefix + gen rows]); only the attention kernel
        changes from the dense per-layer prefix gather to the batched
        transient paged-varlen read, and the branch GEMMs run once as rows.
        """
        m = self._ensure_loaded().model
        paged_branches = gs.paged_branches
        if paged_branches is None:
            raise invalid_descriptor("paged denoise branches are not initialized")
        num_vae = int(gs.num_vae)
        total = num_vae + 2
        embeds = m.gen_segment_embeds(
            num_vae, gs.vae_pos_ids, step.latent, float(step.t.detach().float().item())
        )
        rows = len(branches)
        positions = paged_branches.positions_tensor(
            branches,
            device=self.device,
            width=total,
        )
        batched = paged_branches.batched_cache(branches)
        hidden = m.lm.forward_paged_gen_batch(
            embeds.unsqueeze(0).expand(rows, total, embeds.shape[-1]),
            positions,
            m.gen_segment_is_gen(num_vae),
            batched,
        )
        velocity = m.llm2vae(hidden[:, 1 : 1 + num_vae].to(torch.bfloat16))
        return {branch: velocity[row] for row, branch in enumerate(branches)}

    def predict_denoise_velocity(self, step: TextImageDenoiseStep, branch: str) -> torch.Tensor:
        loaded = self._ensure_loaded()
        m = loaded.model
        pool = loaded.pool
        gs = step.extra["gs"]
        paged_branches = gs.paged_branches
        if paged_branches is not None and paged_branches.has_all((branch,)):
            # Single-row batched path: the staged scratch branches are the only
            # prefill of the CFG prefixes, so per-branch prediction rides the
            # same substrate (B=1) rather than a diverging in-RAM copy.
            return self._predict_denoise_velocity_rows(step, gs, (branch,))[branch]
        if branch == "cond":
            cache = pool.view(self._state(step.req_id).block_ids, gs.cond_branch_kvlen)
            position = gs.cond_pos
        elif branch == "text_uncond":
            if gs.uses_context_image_feedback:
                cache = pool.view(self._state(step.req_id).block_ids, gs.text_branch_kvlen)
                position = gs.text_branch_pos
            else:
                cache = gs.cfg_cache
                position = gs.cfg_pos
        elif branch == "img_uncond":
            cache = gs.cfg_img_cache
            position = gs.cfg_img_pos
        else:
            raise invalid_descriptor(f"unknown denoise branch {branch!r}")
        if cache is None:
            raise invalid_descriptor(f"denoise branch {branch!r} is not initialized")
        seg = m.build_gen_segment(
            gs.num_vae,
            gs.vae_pos_ids,
            step.latent,
            float(step.t.detach().float().item()),
            position,
            cache,
        )
        hidden = m.run([seg])[0]
        return m.velocity_from_hidden(hidden, gs.num_vae)

    def apply_denoise_update(self, step: TextImageDenoiseStep, latent: torch.Tensor) -> None:
        gs = step.extra["gs"]
        gs.x_t = latent.to(dtype=gs.x_t.dtype, device=gs.x_t.device)

    def accept_denoise_update(self, ctx: TextImageDenoiseStep, latent: torch.Tensor) -> None:
        self.apply_denoise_update(ctx, latent)

    def _sync_text_cache_after_image(self, req_id: int, *, length: int, last_position: int) -> None:
        """Advance the driver's text cache past an image KV span written outside it.

        Encode/commit write image KV through raw pool views; the shared text
        driver must resume text from the post-image KV length and rope
        position (block ids are already shared, so only length/position move).
        """
        st = self.interleaved_image_state(int(req_id))
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
        image_processor = loaded.image_processor
        r = op["req_id"]
        gs = self._gen_state(r)
        if gs is None:
            return {"req_id": r}
        self._extend_blocks(op)
        img = m.vae_decode(gs.x_t, gs.H, gs.W)
        if gs.uses_context_image_feedback:
            base = self._length(r) or gs.cond_branch_kvlen
            rope = gs.cond_pos
            pre = image_processor.resize_for_vae(img)
            clean_lat, vpos, _ = m.vae_encode_clean(image_processor.vae_tensor(pre))
            nv = clean_lat.shape[0]
            v1 = pool.view(self._state(r).block_ids, base)
            s1 = m.build_gen_segment(nv, vpos, clean_lat, 0.0, rope, v1, update=True)
            m.run([s1])
            base += nv + 2
            vemb = m.vit_encode(image_processor.vit_tensor(pre))
            nt = vemb.shape[0]
            v2 = pool.view(self._state(r).block_ids, base)
            m.run([m.build_und_image_segment(vemb, rope + 1, v2, update=True)])
            base += nt + 2
            self._set_length(r, base)
            # Keep the shared text driver's cache authoritative: the committed
            # image spans consumed KV up to `base` and rope positions
            # rope / rope + 1; following text continues at rope + 2.
            self._sync_text_cache_after_image(r, length=base, last_position=rope + 1)
            added = (nv + 2) + (nt + 2)
            b64 = pil_image_to_png_b64(img)
            self._pop_gen_state(r)
            return {"req_id": r, "image_png_b64": b64, "image_hw": [gs.H, gs.W],
                    "num_tokens": added}
        rec = self._record(r)
        retain_images = bool((rec.get("image") or {}).get("retain_images", True))
        added = 0
        if retain_images:
            # Interleave continuation: persist the generated latents into the
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
        self,
        input_ids: Any,
        positions: Any | None = None,
        *,
        kv: Any = None,
        mode: str | None = None,
        input_embeds: torch.Tensor | None = None,
        op: dict[str, Any] | None = None,
        request_state: RequestState | None = None,
    ) -> Any:
        loaded = self._ensure_loaded()
        with self._autocast():
            if isinstance(input_ids, UniForwardBatch):
                batch = input_ids
                raise capability_mismatch(f"BAGEL direct batch forward is unsupported for {batch.mode}")
            del loaded, positions, kv, mode, input_embeds, request_state
            if op is None:
                raise invalid_descriptor("BAGEL text forward requires the source op")
            return self.run_text_logits(dict(op))

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
