"""Resolve checkpoint identity and inspect installed serving prerequisites."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from uniserve.loading import Config as IOConfig
from uniserve_models import loading as models


def inspect_model(
    model: str, *, download: bool = False, revision: str | None = None
) -> dict:
    """Inspect the normalized checkpoint metadata used for public loading."""
    from uniserve_models.minimax_h3 import Config as H3Config
    from uniserve_models.minimax_h3.config import (
        FASTH3_MODEL_ID,
        FASTH3_REVISION,
    )

    if model == FASTH3_MODEL_ID and revision is None:
        revision = FASTH3_REVISION

    source = models.read_config(
        model,
        io=IOConfig(revision=revision),
        modules=None if download else frozenset(),
    )
    descriptions = {
        "uniserve_models.minimax_h3": "minimax-h3",
        "uniserve_models.qwen3": "qwen3",
        "uniserve_models.bagel": "bagel",
        "uniserve_models.sensenova_u1": "sensenova",
    }

    local = Path(model).expanduser()
    repository = None if local.exists() else model
    checkpoint_info = None
    root = local.parent if local.is_file() else local

    if isinstance(source.model, H3Config):
        if repository is not None:
            if source.tokenizer is None:
                raise ValueError(
                    "H3 inspection requires its resolved tokenizer directory"
                )
            root = source.tokenizer.parent

        architecture = source.model
        diffusion, output = architecture.diffusion, architecture.output
        manifest = json.loads((root / "fastvideo_inference.json").read_text())
        # Inspection reports checkpoint identity; numerical architecture
        # configs retain only fields consumed by the network and its
        # mathematical recipes.
        checkpoint_info = {
            "family": "minimax-h3",
            "variant": "fasth3",
            "model_id": manifest["model_id"],
            "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
            "attention": "vsa",
            "sparsity": 0.9,
            "tasks": ["t2va"],
            "inference_grid": [
                *(step / diffusion.time_scale for step in diffusion.ladder),
                0.0,
            ],
            "sigma_shifts": [diffusion.video_shift, diffusion.audio_shift],
            "denoise_steps": len(diffusion.ladder),
            "width": output.frame_size.width,
            "height": output.frame_size.height,
            "fps": output.frame_rate,
            "audio_rate": output.sample_rate,
            "precision_presets": list(source.precisions),
            "revision": root.name if root.parent.name == "snapshots" else None,
        }
    return {
        "description": descriptions[
            source.model_class.__module__.rsplit(".", 1)[0]
        ],
        "model_path": str(root) if checkpoint_info is not None else model,
        "repository": repository,
        # ``contract`` is the established Rust/Python inspection message field.
        "contract": checkpoint_info,
        "python": sys.executable,
    }


def doctor(model: dict, ranks: int) -> dict:
    """Check native prerequisites without model allocation.

    Covers native ABI, CUDA architecture, peer access and codecs.
    """
    import torch

    from uniserve.runtime.backends.attention import vsa
    from uniserve_models.minimax_h3 import precisions

    from ..media.mux import require_media_codecs

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "H3 requires the locked Python 3.12 environment; "
            "run uv sync --extra h3"
        )
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
            f"requested {ranks} GPUs, but only "
            f"{torch.cuda.device_count()} are visible"
        )

    devices = []
    for rank in range(ranks):
        provider = vsa.resolve("auto", device=torch.device("cuda", rank))
        free, total = torch.cuda.mem_get_info(rank)
        devices.append(
            {
                "rank": rank,
                "name": torch.cuda.get_device_name(rank),
                "sparse_attention": type(provider).__module__.rsplit(".", 1)[
                    -1
                ],
                "free_bytes": free,
                "total_bytes": total,
            }
        )
        for peer in range(ranks):
            if peer != rank and not torch.cuda.can_device_access_peer(
                rank, peer
            ):
                raise RuntimeError(
                    f"CUDA peer access unavailable: {rank} -> {peer}"
                )

    require_media_codecs("libx264", "aac")

    root = Path(model["model_path"])
    component_bytes = {}
    for name in ("transformer", "text_encoder", "vae", "audio_vae"):
        paths = list((root / name).glob("*.safetensors"))
        component_bytes[name] = (
            sum(path.stat().st_size for path in paths) if paths else None
        )

    # Remote checkpoints report component sizes from repository metadata when
    # the weight files have not been downloaded yet.
    if model.get("repository") and any(
        value is None for value in component_bytes.values()
    ):
        from huggingface_hub import HfApi

        metadata = HfApi().model_info(
            model["repository"], revision=root.name, files_metadata=True
        )
        for name in component_bytes:
            sizes = [
                file.size
                for file in metadata.siblings or ()
                if file.rfilename.startswith(name + "/")
                and file.rfilename.endswith(".safetensors")
            ]
            if sizes and all(size is not None for size in sizes):
                component_bytes[name] = sum(
                    size for size in sizes if size is not None
                )

    known_sizes = {
        name: size for name, size in component_bytes.items() if size is not None
    }
    weight_ceiling = (
        known_sizes["transformer"]
        + known_sizes["text_encoder"] // ranks
        + known_sizes["vae"]
        + known_sizes["audio_vae"]
        if len(known_sizes) == len(component_bytes)
        else None
    )

    return {
        **model,
        "toolkit": toolkit,
        "devices": devices,
        "codecs": ["libx264", "aac"],
        "native_sparse_attention": "available",
        "precision": {
            "dtype": str(precisions["balanced"].dtype),
            "dtypes": {
                path: str(dtype)
                for path, dtype in precisions["balanced"].dtypes.items()
            },
            "quantization": {
                path: None
                if value is None
                else {
                    "weight": value.weight.format,
                    "activation": value.activation.format,
                }
                for path, value in precisions["balanced"].quantization.items()
            },
        },
        "capacity_estimate": {
            "checkpoint_component_bytes": component_bytes,
            "unquantized_weight_bytes_per_rank": weight_ceiling,
            "basis": (
                "replicated denoiser and decoders; encoder TP within "
                "the supplied rank budget"
            ),
            "limitations": (
                "weight-only estimate before balanced quantization; "
                "activation, transfer, graph and allocator storage "
                "require additional memory"
            ),
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

    result = inspect_model(
        args.model, download=args.download, revision=args.revision
    )
    if args.doctor:
        result = doctor(result, args.worker_ranks)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
