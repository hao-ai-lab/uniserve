"""Resolve checkpoint identity and inspect the installed serving prerequisites."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..loader.config import LoadConfig
from ..loader.source import read_model_config, resolve_model_root
from .catalog import resolve_catalog_entry


def inspect_model(model: str, *, download: bool = False, revision: str | None = None) -> dict:
    """Resolve catalog metadata and validate the H3 variant without GPU weights."""

    from ..models.minimax_h3.config import FASTH3_MODEL_ID, FASTH3_REVISION, resolve_h3_contract
    from ..models.minimax_h3.weights import require_h3_checkpoint

    if model == FASTH3_MODEL_ID and revision is None:
        revision = FASTH3_REVISION
    root, repository = resolve_model_root(model, LoadConfig(revision=revision))
    config = read_model_config(root)
    entry = resolve_catalog_entry(config["architectures"])
    descriptions = {
        "MiniMaxH3Transformer3DModel": "minimax-h3",
        "Qwen3ForCausalLM": "qwen3",
        "BagelForConditionalGeneration": "bagel",
        "NEOChatModel": "sensenova",
    }
    if repository and entry.architecture == "MiniMaxH3Transformer3DModel":
        from huggingface_hub import snapshot_download

        logging.info("download: resolving checkpoint %s@%s", repository, revision or root.name)
        root = Path(
            snapshot_download(
                repository,
                revision=revision or root.name,
                allow_patterns=list(entry.sidecars),
            )
        )
    contract = None
    if entry.architecture == "MiniMaxH3Transformer3DModel":
        require_h3_checkpoint(root)
        contract = resolve_h3_contract(root)
        if repository and download:
            root = Path(snapshot_download(repository, revision=root.name))
    return {
        "description": descriptions[entry.architecture],
        "model_path": str(root) if contract is not None else model,
        "repository": repository,
        "contract": contract,
        "python": sys.executable,
    }


def doctor(model: dict, ranks: int) -> dict:
    """Check native ABI, CUDA architecture, peer access and codecs without model allocation."""

    import torch

    from ..backends.attention import video_sparse_sm100
    from ..media.mux import require_media_codecs
    from ..models.minimax_h3.config import PRECISION_PRESETS

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("H3 requires the locked Python 3.12 environment; run uv sync --extra h3")
    from .. import _uniserve_ipc  # noqa: F401

    nvcc = shutil.which("nvcc") or str(
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
    )
    toolkit = subprocess.check_output([nvcc, "--version"], text=True).strip()
    if "release 13." not in toolkit:
        raise RuntimeError("H3 requires the CUDA 13 toolkit")
    if ranks < 1:
        raise RuntimeError("worker-ranks must be positive")
    if torch.cuda.device_count() < ranks:
        raise RuntimeError(
            f"requested {ranks} GPUs, but only {torch.cuda.device_count()} are visible"
        )
    devices = []
    for rank in range(ranks):
        if torch.cuda.get_device_capability(rank) != (10, 0):
            raise RuntimeError(f"GPU {rank} does not support the H3 sm_100a provider")
        free, total = torch.cuda.mem_get_info(rank)
        devices.append(
            {
                "rank": rank,
                "name": torch.cuda.get_device_name(rank),
                "free_bytes": free,
                "total_bytes": total,
            }
        )
        for peer in range(ranks):
            if peer != rank and not torch.cuda.can_device_access_peer(rank, peer):
                raise RuntimeError(f"CUDA peer access unavailable: {rank} -> {peer}")
    if not video_sparse_sm100.available():
        raise RuntimeError(
            "required SM100a sparse-attention provider is unavailable"
        ) from video_sparse_sm100.import_error()
    require_media_codecs("libx264", "aac")
    root = Path(model["model_path"])
    component_bytes = {}
    for name in ("transformer", "text_encoder", "vae", "audio_vae"):
        paths = list((root / name).glob("*.safetensors"))
        component_bytes[name] = sum(path.stat().st_size for path in paths) if paths else None
    if model.get("repository") and any(value is None for value in component_bytes.values()):
        from huggingface_hub import HfApi

        metadata = HfApi().model_info(model["repository"], revision=root.name, files_metadata=True)
        for name in component_bytes:
            sizes = [
                file.size
                for file in metadata.siblings
                if file.rfilename.startswith(name + "/") and file.rfilename.endswith(".safetensors")
            ]
            if sizes and all(size is not None for size in sizes):
                component_bytes[name] = sum(sizes)
    complete_sizes = all(value is not None for value in component_bytes.values())
    weight_ceiling = (
        component_bytes["transformer"]
        + component_bytes["text_encoder"] // ranks
        + component_bytes["vae"]
        + component_bytes["audio_vae"]
        if complete_sizes
        else None
    )
    return {
        **model,
        "toolkit": toolkit,
        "devices": devices,
        "codecs": ["libx264", "aac"],
        "native_sparse_attention": "available",
        "precision": PRECISION_PRESETS["balanced"],
        "capacity_estimate": {
            "checkpoint_component_bytes": component_bytes,
            "unquantized_weight_bytes_per_rank": weight_ceiling,
            "basis": "replicated denoiser and decoders; encoder TP within the supplied rank budget",
            "limitations": "weight-only estimate before balanced quantization; activation, transfer, graph and allocator storage require additional memory",
            "resident_requests": 2,
            "max_video_seconds": 15,
            "max_prompt_tokens": 16384,
        },
        "allocation": "not attempted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--worker-ranks", type=int, default=1)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    result = inspect_model(args.model, download=args.download, revision=args.revision)
    if args.doctor:
        result = doctor(result, args.worker_ranks)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
