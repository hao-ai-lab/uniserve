"""Validated H3 component formats and fixed precision preset values."""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypedDict

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
FASTH3_8STEP_MODEL_ID = "FastVideo/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10"

# Published identities are immutable. Custom exports use their own model_id.
FASTH3_8STEP_RELEASE = {
    "checkpoint_content_sha256": "516323fa396fa5dff4e82669d4e9a08a5791692a3d3b98ff6bc3de3fc6a33d11",
    "checkpoint_metadata_sha256": "ca9f2d609c05742ba465d24989981ec02cca26acb6ca2f163dc0f6dc8d11c27b",
    "fastvideo_commit": "24bbe7fddd05ca6f2c34b3dbed06ac1c75b72086",
    "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
    "video_scheduler_shift": 10.0,
    "audio_scheduler_shift": 3.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.8,
    "sequence_parallel_size": 4,
}


class H3Contract(TypedDict):
    """Normalized checkpoint settings shared by catalog, runtime and capabilities."""

    family: str
    variant: str
    model_id: str
    revision: str | None
    checkpoint_content_sha256: str
    checkpoint_metadata_sha256: str
    fastvideo_commit: str
    attention: str
    attention_backend: str
    sparsity: float
    tile_size: int
    guidance_scale: float
    sequence_parallel_size: int
    ladder: list[int]
    inference_grid: list[float]
    sigma_shifts: list[float]
    denoise_steps: int
    tasks: list[str]
    width: int
    height: int
    fps: int
    audio_rate: int
    precision_presets: list[str]


def resolve_h3_contract(root: Path, contract_path: Path | None = None) -> H3Contract:
    """Resolve a declared distilled recipe before allocating components.

    An explicit sidecar (argument or UNISERVE_H3_CONTRACT) must name its local
    checkpoint_root when no embedded manifest exists. Published manifests may
    not be overridden. UNISERVE_H3_VARIANT selects base or ref from a pinned
    repository containing both denoisers; it cannot override a distilled recipe.
    Hashes are exporter declarations, not rehashed weights.
    Only the pinned four-step release uses its historical uniform grid rather
    than the explicit DMD-index protocol selected by custom checkpoints.
    """

    variant = os.environ.get("UNISERVE_H3_VARIANT")
    if variant not in (None, "base", "ref"):
        raise ValueError("UNISERVE_H3_VARIANT must be base or ref")
    selected = contract_path or os.environ.get("UNISERVE_H3_CONTRACT")
    path = Path(selected) if selected else root / "fastvideo_inference.json"
    if not path.is_file():
        if selected:
            raise ValueError(f"H3 contract sidecar does not exist: {path}")
        from .base_contract import resolve_base_h3_contract

        if not (root / "modular_model_index.json").is_file():
            raise ValueError(
                "MiniMax H3 requires a pinned base H3 root or an explicit "
                "UNISERVE_H3_CONTRACT sidecar for a full local export"
            )
        # Full base repositories contain both denoisers. A process may select
        # one explicitly; otherwise retain checkpoint-based reference discovery.
        reference = variant == "ref" if variant else (root / "transformer_ref").exists()
        return resolve_base_h3_contract(root, reference=reference)
    if variant is not None:
        raise ValueError("UNISERVE_H3_VARIANT cannot override a distilled contract")
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
    pinned = manifest.get("model_id") == FASTH3_MODEL_ID
    if pinned:
        for name, value in expected.items():
            if type(manifest.get(name)) is not type(value) or manifest.get(name) != value:
                raise ValueError(
                    f"unsupported FastH3 checkpoint: {name} must be {value!r}, "
                    f"got {manifest.get(name)!r}; use {FASTH3_MODEL_ID}@{FASTH3_REVISION}"
                )
    else:
        _validate_manifest(manifest)
        if manifest["model_id"] == FASTH3_8STEP_MODEL_ID:
            for name, value in FASTH3_8STEP_RELEASE.items():
                if manifest.get(name) != value:
                    raise ValueError(f"published eight-step checkpoint {name} must be {value!r}")
    if selected:
        # A sidecar cannot relabel a published release or silently replace its recipe.
        embedded = root / "fastvideo_inference.json"
        if embedded.is_file() and json.loads(embedded.read_text()) != manifest:
            raise ValueError("sidecar contract disagrees with fastvideo_inference.json")
        if not embedded.is_file():
            if manifest.get("checkpoint_root") != str(root.resolve()):
                raise ValueError("sidecar checkpoint_root must equal the resolved checkpoint root")
    ladder = FASTH3_LADDER if pinned else tuple(manifest["dmd_denoising_steps"])
    shifts = (
        FASTH3_SHIFTS
        if pinned
        else (manifest["video_scheduler_shift"], manifest["audio_scheduler_shift"])
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
        "attention_backend": "VIDEO_SPARSE_ATTN" if pinned else manifest["attention_backend"],
        "sparsity": manifest["vsa_sparsity"],
        "tile_size": manifest["vsa_tile_size"],
        "guidance_scale": manifest["guidance_scale"],
        "sequence_parallel_size": manifest.get("sequence_parallel_size", 4),
        "checkpoint_metadata_sha256": manifest["checkpoint_metadata_sha256"],
        "fastvideo_commit": manifest["fastvideo_commit"],
        "ladder": list(ladder),
        "tasks": ["t2va"],
        "inference_grid": [*(value / FASTH3_TIME_SCALE for value in ladder), 0.0],
        "sigma_shifts": list(shifts),
        "denoise_steps": len(ladder),
        "width": 1344,
        "height": 768,
        "fps": 24,
        "audio_rate": 32000,
        "precision_presets": list(PRECISION_PRESETS),
    }


def _validate_manifest(manifest: dict) -> None:
    """Validate explicit distilled T2AV recipes; never infer missing numerical settings."""

    for key, value in {
        "schema_version": "fasth3-inference-contract-v1",
        "task": "t2av",
        "guidance_scale": 1.0,
        "vsa_tile_size": 64,
    }.items():
        if manifest.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if type(manifest.get("guidance_scale")) not in (float, int):
        raise ValueError("guidance_scale must be numeric")
    if type(manifest.get("vsa_tile_size")) is not int:
        raise ValueError("vsa_tile_size must be an integer")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"].strip():
        raise ValueError("model_id must be a nonempty identity")
    for key, length in (
        ("checkpoint_content_sha256", 64),
        ("checkpoint_metadata_sha256", 64),
        ("fastvideo_commit", 40),
    ):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", manifest[key]
        ):
            raise ValueError(f"{key} must be a lowercase hexadecimal digest")
    if manifest.get("attention_backend") not in {"VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"}:
        raise ValueError("attention_backend must select a supported VSA backend")
    for key in ("video_scheduler_shift", "audio_scheduler_shift", "vsa_sparsity"):
        value = manifest.get(key)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key == "vsa_sparsity":
            if not 0 <= value < 1:
                raise ValueError("vsa_sparsity must be in [0, 1)")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
    ladder = manifest.get("dmd_denoising_steps")
    if (
        not isinstance(ladder, list)
        or not ladder
        or any(type(value) is not int or not 0 < value <= 1000 for value in ladder)
        or any(left <= right for left, right in zip(ladder, ladder[1:]))
    ):
        raise ValueError(
            "dmd_denoising_steps must be strictly descending integer indices in (0, 1000]"
        )
    for key, expected_count in (
        ("transformer_forwards", len(ladder)),
        ("num_inference_steps", len(ladder) + 1),
    ):
        if type(manifest.get(key)) is not int or manifest[key] != expected_count:
            raise ValueError(f"{key} disagrees with dmd_denoising_steps")
    if (
        type(manifest.get("sequence_parallel_size")) is not int
        or manifest["sequence_parallel_size"] < 1
    ):
        raise ValueError("sequence_parallel_size must be a positive integer")


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
