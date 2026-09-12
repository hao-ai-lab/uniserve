"""Tensor-parallel Qwen3-VL text conditioner for MiniMax H3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLConfig

from ...nn.decoder.qwen import Qwen3Config, Qwen3Model
from ...nn.layer import LayerConfig
from ...nn.mesh import DeviceMesh
from ...nn.quant.config import LinearPrecision, QuantizationConfig
from .packing import TEXT_TAG, VIDEO_TAG
from .vision import H3VisionModel

__all__ = ["H3TextEncoderConfig", "MiniMaxH3TextEncoder"]


@dataclass(frozen=True, slots=True)
class H3TextEncoderConfig:
    """Defines the H3 text encoder's vocabulary and tensor geometry.

    The configuration fixes hidden width, attention heads, layer count, and rotary
    settings.
    """

    vocab_size: int = 151_936
    hidden_size: int = 5_120
    intermediate_size: int = 25_600
    checkpoint_layers: int = 64
    retained_layers: int = 50
    heads: int = 64
    kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    norm_eps: float = 1e-6
    max_text_rows: int = 1_024


class MiniMaxH3TextEncoder(nn.Module):
    """The Qwen3-VL language path through checkpoint hidden state 50."""

    architecture = "Qwen3VLForConditionalGeneration"

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        max_text_rows: int,
        parameter_device: torch.device | str = "meta",
        dtype: torch.dtype = torch.bfloat16,
        linear_precision: LinearPrecision = "bf16",
        checkpoint_path: Path | None = None,
        config: H3TextEncoderConfig | None = None,
    ) -> None:
        """Configure a bounded BF16 text-conditioning path on the supplied mesh."""

        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("the H3 text encoder uses bfloat16 activations")
        if linear_precision not in ("bf16", "fp8", "nvfp4"):
            raise ValueError(f"unsupported H3 text encoder linear precision {linear_precision!r}")
        self.config = config or H3TextEncoderConfig(max_text_rows=int(max_text_rows))
        self.processor = None
        self.visual = None
        self.vision_config = None
        if checkpoint_path is not None:
            self.processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True)
            self.vision_config = Qwen3VLConfig.from_pretrained(
                checkpoint_path, local_files_only=True
            )
            if self.vision_config.vision_config.out_hidden_size != self.config.hidden_size:
                raise ValueError("Qwen vision output width must match the language hidden width")
            with torch.device(parameter_device):
                self.visual = H3VisionModel(self.vision_config.vision_config)
        if any(
            width % mesh.size("tp")
            for width in (self.config.heads, self.config.kv_heads, self.config.intermediate_size)
        ):
            raise ValueError("H3 text encoder TP must divide query/KV heads and MLP width")
        self.mesh = mesh
        decoder_config = Qwen3Config(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.retained_layers,
            num_attention_heads=self.config.heads,
            num_key_value_heads=self.config.kv_heads,
            head_dim=self.config.head_dim,
            hidden_act="silu",
            rms_norm_eps=self.config.norm_eps,
            rope_theta=self.config.rope_theta,
            max_position_embeddings=self.config.max_text_rows,
            attention_bias=False,
            tie_word_embeddings=False,
            num_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=self.config.intermediate_size,
        )
        quantization = QuantizationConfig(
            method="unquantized" if linear_precision == "bf16" else linear_precision
        )
        attention_quantization = QuantizationConfig(
            method="nvfp4" if linear_precision == "nvfp4" else "unquantized"
        )
        with torch.device(parameter_device):
            self.language_model = Qwen3Model(
                decoder_config,
                layer_config=LayerConfig(mesh.get_group("tp"), quantization, "language_model"),
                attention_quantization=attention_quantization,
                mlp_input_dtype=torch.float8_e4m3fn if linear_precision == "fp8" else None,
                normalize_output=False,
            )

    def forward(
        self, token_ids: torch.Tensor, images: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Validate a single bounded prompt and produce its text-conditioning states."""

        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("H3 text conditioning requires one token sequence")
        if token_ids.shape[1] < 1 or token_ids.shape[1] > self.config.max_text_rows:
            raise ValueError(
                f"H3 prompt token count must be between 1 and {self.config.max_text_rows}"
            )
        if images:
            return self.encode_presentation(token_ids, images)[0]
        positions = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device)
        return self.language_model(token_ids, positions)

    def numerical_entry(self, token_ids: torch.Tensor, *images: torch.Tensor) -> torch.Tensor:
        """Adapt the numerical runner's flat tensor inputs to ordered decoded images."""
        return self(token_ids, list(images) if images else None)

    def prepare_images(self, images: list[torch.Tensor]):
        """Process decoded HWC uint8 RGB rasters with the checkpoint's Qwen processor."""
        if self.processor is None:
            raise ValueError("image conditioning requires the checkpoint processor")
        for image in images:
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
                raise ValueError("H3 decoded images must be HWC uint8 RGB")
        return self.processor.image_processor(
            images=[image.cpu().numpy() for image in images], return_tensors="pt"
        )

    def encode_presentation(
        self, token_ids: torch.Tensor, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return layer-50 presentation states and modality tags in causal input order.

        Labels and vision spans precede prompt tokens, exactly as in FastVideo.
        These are Qwen conditioning rows, not the separately encoded VAE reference rows.
        """
        if self.visual is None or self.vision_config is None or self.processor is None:
            raise ValueError("image conditioning requires loaded Qwen vision weights")
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or not images:
            raise ValueError("image presentation requires one prompt and nonempty images")
        processed = self.prepare_images(images)
        grid = processed["image_grid_thw"]
        merge = self.vision_config.vision_config.spatial_merge_size
        if (
            grid.shape != (len(images), 3)
            or bool((grid <= 0).any())
            or bool((grid[:, 1:] % merge != 0).any())
        ):
            raise ValueError("invalid Qwen image grid")
        ids, tags, positions = [], [], []
        offset = 0
        for index, (frames, height, width) in enumerate(grid.tolist()):
            height, width = height // merge, width // merge
            count = frames * height * width
            label = self.processor.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)[
                "input_ids"
            ]
            prefix = label + [self.vision_config.vision_start_token_id]
            ids.extend(
                prefix
                + [self.vision_config.image_token_id] * count
                + [self.vision_config.vision_end_token_id]
            )
            tags.extend([TEXT_TAG] * len(label) + [VIDEO_TAG] * (count + 2))
            positions.append(torch.arange(len(prefix)).expand(3, -1) + offset)
            axes = torch.meshgrid(
                torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij"
            )
            visual_positions = torch.stack([axis.flatten() for axis in axes]) + offset + len(prefix)
            positions.append(visual_positions)
            offset = int(visual_positions.max()) + 1
            positions.append(torch.full((3, 1), offset))
            offset += 1
        prompt = token_ids[0].tolist()
        ids.extend(prompt)
        tags.extend([TEXT_TAG] * len(prompt))
        positions.append(torch.arange(len(prompt)).expand(3, -1) + offset)
        if not prompt or len(ids) > self.config.max_text_rows:
            raise ValueError("H3 image presentation exceeds conditioning capacity")
        device = token_ids.device
        presentation = torch.tensor([ids], device=device, dtype=torch.long)
        mask = presentation == self.vision_config.image_token_id
        features, deepstack = self.visual(
            processed["pixel_values"].to(
                device=device, dtype=self.visual.patch_embed.proj.weight.dtype
            ),
            grid.to(device),
        )
        expected = (int(mask.sum()), self.config.hidden_size)
        if (
            tuple(features.shape) != expected
            or len(deepstack) != len(self.vision_config.vision_config.deepstack_visual_indexes)
            or any(tuple(value.shape) != expected for value in deepstack)
        ):
            raise ValueError("Qwen vision/DeepStack shapes do not match image placeholders")
        embeds = self.language_model.embed_tokens(presentation)
        embeds = embeds.masked_scatter(mask.unsqueeze(-1), features.to(embeds))
        # Qwen3-VL interleaves height/width frequencies into the temporal base.
        coordinates = torch.cat(positions, dim=1).to(device)
        frequencies = coordinates.float()[:, :, None] * self.language_model.rotary.inv_freq.to(
            device
        )
        interleaved = frequencies[0].clone()
        sections = self.vision_config.text_config.rope_parameters["mrope_section"]
        for axis in (1, 2):
            interleaved[:, axis : sections[axis] * 3 : 3] = frequencies[
                axis, :, axis : sections[axis] * 3 : 3
            ]
        hidden = self.language_model(
            presentation,
            coordinates[0],
            input_embeds=embeds,
            position_embeddings=(interleaved.cos(), interleaved.sin()),
            visual_mask=mask,
            deepstack_features=deepstack,
        )
        return hidden, torch.tensor(tags, dtype=torch.long, device=device)
