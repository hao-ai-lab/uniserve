"""Pinned undistilled MiniMax-H3 T2VA checkpoint identity and recipe."""

import json
from pathlib import Path

import torch

BASE_H3_MODEL_ID = "MiniMaxAI/MiniMax-H3"
BASE_H3_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
BASE_H3_GRID_POINTS = 50
BASE_H3_SHIFTS = (12.0, 3.0)
FASTVIDEO_BASE_COMMIT = "a943220c115228ade5d57b3bab9a6a87fd600a10"


def resolve_base_h3_contract(root: Path) -> dict[str, object]:
    """Validate top-level component provenance and complete indexed shards.

    Local downloads require Hugging Face revision receipts for every consumed
    file. These establish declared provenance, not recomputed weight hashes.
    Snapshot roots additionally require the pinned repository cache namespace.
    Nested task exports and distilled manifests are not base checkpoints.
    """

    if (root / "fastvideo_inference.json").exists():
        raise ValueError("base H3 must not contain a distilled inference manifest")
    snapshot = (
        root.name == BASE_H3_REVISION
        and root.parent.name == "snapshots"
        and root.parent.parent.name == "models--MiniMaxAI--MiniMax-H3"
    )

    def require_file(relative: str) -> None:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"base H3 requires {relative}")
        if not snapshot:
            receipt = root / ".cache/huggingface/download" / (relative + ".metadata")
            lines = receipt.read_text().splitlines() if receipt.is_file() else []
            if not lines or lines[0] != BASE_H3_REVISION:
                raise ValueError(f"base H3 {relative} requires revision {BASE_H3_REVISION}")

    require_file("modular_model_index.json")
    index = json.loads((root / "modular_model_index.json").read_text())
    if index.get("_class_name") != "MiniMaxH3ModularPipeline":
        raise ValueError("base H3 requires the MiniMaxH3ModularPipeline root")
    for component in ("transformer", "text_encoder", "vae", "audio_vae"):
        require_file(f"{component}/config.json")
        directory = root / component
        indices = list(directory.glob("*.safetensors.index.json"))
        if indices:
            if len(indices) != 1:
                raise ValueError(f"base H3 {component} requires one weight index")
            index = indices[0]
            require_file(f"{component}/{index.name}")
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"base H3 {component} has an invalid weight index")
            shards = set(mapping.values())
        else:
            shards = {path.name for path in directory.glob("*.safetensors")}
        if not shards or (component == "transformer" and len(shards) != 14):
            raise ValueError(f"base H3 {component} has an invalid shard count")
        for shard in shards:
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError(f"base H3 {component} has an unsafe shard path")
            require_file(f"{component}/{shard}")
    for relative in (
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        require_file(relative)
    for component, shift in zip(("scheduler", "audio_scheduler"), BASE_H3_SHIFTS, strict=True):
        config = json.loads((root / component / "scheduler_config.json").read_text())
        if config.get("shift") != shift:
            raise ValueError(f"base H3 {component} requires shift {shift}")
    return {
        "family": "minimax-h3",
        "variant": "base",
        "model_id": BASE_H3_MODEL_ID,
        "revision": BASE_H3_REVISION,
        "identity_validation": "huggingface_revision_provenance",
        "fastvideo_commit": FASTVIDEO_BASE_COMMIT,
        "attention": "dense",
        "attention_backend": "FLASH_ATTN",
        "sparsity": 0.0,
        "tile_size": 64,
        "tasks": ["t2va"],
        "num_inference_steps": BASE_H3_GRID_POINTS,
        "inference_grid": torch.linspace(
            1.0, 0.0, BASE_H3_GRID_POINTS, dtype=torch.float32
        ).tolist(),
        "denoise_steps": BASE_H3_GRID_POINTS - 1,
        "sigma_shifts": list(BASE_H3_SHIFTS),
        "guidance_scale": 1.0,
        "width": 1344,
        "height": 768,
        "fps": 24,
        "audio_rate": 32000,
        "precision_presets": ["quality", "balanced", "performance", "maximum"],
    }
