"""Validated H3 component formats and fixed precision preset values."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...nn.quant.config import LinearPrecision

SUPPORTED_PRECISIONS: Mapping[str, tuple[LinearPrecision, ...]] = {
    "transformer.attention": ("bf16", "fp8", "nvfp4"),
    "transformer.mlp": ("bf16", "fp8", "mxfp8", "nvfp4"),
    "text_encoder": ("bf16", "fp8", "nvfp4"),
    "video_vae": ("fp16", "bf16", "nvfp4"),
}

PRECISION_PRESETS: Mapping[str, Mapping[str, LinearPrecision]] = {
    "quality": {
        "transformer.attention": "bf16",
        "transformer.mlp": "bf16",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "balanced": {
        "transformer.attention": "bf16",
        "transformer.mlp": "bf16",
        "text_encoder": "bf16",
        "video_vae": "nvfp4",
    },
    "performance": {
        "transformer.attention": "bf16",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "nvfp4",
    },
    "maximum": {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "mxfp8",
        "text_encoder": "fp8",
        "video_vae": "nvfp4",
    },
}

PRECISION_SHORTHANDS: Mapping[str, Mapping[str, LinearPrecision]] = {
    "bf16": PRECISION_PRESETS["quality"],
    "fp8": {
        "transformer.attention": "fp8",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "mxfp8": {
        "transformer.attention": "bf16",
        "transformer.mlp": "mxfp8",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "nvfp4": {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "nvfp4",
        "text_encoder": "nvfp4",
        "video_vae": "nvfp4",
    },
}


FASTH3_LADDER = (1000, 750, 500, 250)
FASTH3_SHIFTS = (12.0, 3.0)
FASTH3_TIME_SCALE = 1000.0

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
FASTH3_REVISION = "5ea076f35b84da4c3c82217112fa733d8eea2ae1"


def resolve_h3_contract(root: Path) -> dict[str, object]:
    """Validate the supported full VSA checkpoint before allocating components.

    The manifest's training indices do not select the inference schedule. The
    pinned FastVideo basic_fasth3 recipe uses five uniformly spaced grid points;
    its explicit DMD-index override is a different numerical protocol.
    """

    path = root / "fastvideo_inference.json"
    if not path.is_file():
        raise ValueError(
            "MiniMax H3 requires fastvideo_inference.json from the full FastH3 VSA "
            "checkpoint; base partitions and adapter-only checkpoints are unsupported"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_id": FASTH3_MODEL_ID,
        "checkpoint_content_sha256": "b36987515e4c75fa4c7aaa632a7842c829ea141b235358a54d782b51230497b3",
        "checkpoint_metadata_sha256": "dcad0fbee2a7c7e75e53435f4fd98fccf3138844883874edf057962ab48fa428",
        "fastvideo_commit": "48a047c05ff4138f20cfa33351499c6ec5945f5d",
        "task": "t2av",
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_tile_size": 64,
        "vsa_sparsity": 0.9,
    }
    if not isinstance(manifest, dict):
        raise ValueError("fastvideo_inference.json must contain an object")
    for name, value in expected.items():
        if type(manifest.get(name)) is not type(value) or manifest.get(name) != value:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be {value!r}, "
                f"got {manifest.get(name)!r}; use {FASTH3_MODEL_ID}@{FASTH3_REVISION}"
            )
    # A local copy has a declared content identity, not independently verified
    # Hub revision provenance. Report the latter only for a snapshot directory.
    revision = root.name if root.parent.name == "snapshots" else None
    return {
        "family": "minimax-h3",
        "variant": "fasth3",
        "model_id": manifest["model_id"],
        "revision": revision,
        "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
        "attention": "vsa",
        "sparsity": 0.9,
        "tasks": ["t2va"],
        "inference_grid": [1.0, 0.75, 0.5, 0.25, 0.0],
        "sigma_shifts": list(FASTH3_SHIFTS),
        "denoise_steps": len(FASTH3_LADDER),
        "width": 1344,
        "height": 768,
        "fps": 24,
        "audio_rate": 32000,
        "precision_presets": list(PRECISION_PRESETS),
    }


@dataclass(frozen=True, slots=True)
class H3TransformerConfig:
    """Defines H3 multimodal width, layer, attention, expert, modulation, and sparse-video geometry."""

    hidden_size: int = 5376
    heads: int = 56
    head_dim: int = 128
    layers: int = 50
    refiner_layers: int = 2
    ffn_dim: int = 14336
    video_channels: int = 24
    audio_channels: int = 32
    text_dim: int = 5120
    frequency_dim: int = 256
    time_hidden_dim: int = 5376
    time_dim: int = 2688
    rope_frequency_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
