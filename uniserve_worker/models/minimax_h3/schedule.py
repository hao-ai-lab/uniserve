"""Checkpoint-trained FastH3 coordinates in video/audio modality order."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ...nn.diffusion.schedule import DiffusionSchedule

FASTH3_LADDER = (1000, 750, 500, 250)
FASTH3_SHIFTS = (12.0, 3.0)
FASTH3_TIME_SCALE = 1000.0


def load_fasth3_schedule(root: Path, device: torch.device) -> DiffusionSchedule:
    """Materialize the checkpoint's trained ladder before modulation preparation.

    Checkpoints carrying an inference contract define explicit DMD jump points
    and FP32 shift arithmetic. Checkpoints without that sidecar use the uniform
    four-interval grid declared by their FastH3 scheduler.
    """

    path = root / "fastvideo_inference.json"
    if not path.is_file():
        return DiffusionSchedule.build(
            FASTH3_LADDER, FASTH3_SHIFTS, scale=FASTH3_TIME_SCALE, device=device
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("FastH3 inference contract must contain an object")
    required = {
        "schema_version": "fasth3-inference-contract-v1",
        "task": "t2av",
        "transformer_forwards": 4,
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_tile_size": 64,
        "vsa_sparsity": 0.9,
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise ValueError(f"FastH3 inference contract {field} must be {expected!r}")
    ladder = contract.get("dmd_denoising_steps")
    if (
        not isinstance(ladder, list)
        or len(ladder) != 4
        or any(type(value) is not int or not 0 < value <= FASTH3_TIME_SCALE for value in ladder)
        or any(left <= right for left, right in zip(ladder, ladder[1:]))
    ):
        raise ValueError("FastH3 DMD ladder requires four descending integer points in (0, 1000]")

    # The published DMD scheduler shifts a CPU FP32 grid before device transfer.
    # Preserve those rounding points in both noise and clean-time coordinates.
    base = torch.tensor(
        [value / FASTH3_TIME_SCALE for value in ladder] + [0.0],
        dtype=torch.float32,
        device="cpu",
    )
    sigmas = tuple(shift * base / (1.0 + (shift - 1.0) * base) for shift in FASTH3_SHIFTS)
    return DiffusionSchedule(
        tuple(values.to(device=device) for values in sigmas),
        tuple((1.0 - values[:-1]).to(device=device) for values in sigmas),
    )
