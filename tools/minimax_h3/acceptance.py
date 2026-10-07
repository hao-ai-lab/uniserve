"""Judge UniServe's MiniMax-H3 trajectories by the two-level acceptance rule.

The rule (``docs/minimax_h3/evaluation.md``) compares UniServe with the
valid implementations of each reference: the reference run on its canonical
exact attention kernel and on the alternative exact kernels, with identical
inputs and noise, at seeds 42, 0, 1 and 2.

* Per-step level: on sampled denoising steps, each implementation predicts
  once from the canonical run's own input sample. A cell is one (step,
  seed); ``K`` counts the cells where UniServe's deviation from the canonical
  prediction exceeds the largest deviation of the alternative kernels. The
  modality fails when ``K`` reaches the one-sided 1% critical value of
  ``Binomial(N, 1/2)``.
* Free-running level: the final generated sample's relative L2 against the
  canonical run. The alternatives' values over the seeds are the valid
  samples; the modality fails when UniServe exceeds their maximum at every
  seed.

A workload passes when both levels pass for video and audio. Decoded-media
metrics (PSNR, SSIM, audio spectral cosine) of the free runs are reported
alongside, not judged.

Writes ``<acceptance>/<workload>/verdict.json`` per workload and
``<acceptance>/verdicts.json`` with every workload, and prints a summary
table. Usage: ``acceptance.py --artifacts artifacts/minimax_h3``.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from trajectory_metrics import compare

SEEDS = (42, 0, 1, 2)
MODALITIES = ("video", "audio")
# One-sided significance of the per-step count test.
ALPHA = 0.01


@dataclass(frozen=True)
class Family:
    """The valid implementations of one reference and where their runs are.

    Attributes:
        canonical: ``reference/<impl>`` of the canonical kernel's runs.
        alternatives: ``reference/<impl>`` of each alternative kernel's
            free-running runs, which replay the canonical noise.
        teacher: ``reference/<impl>`` of each alternative kernel's
            teacher-forced runs, or ``None`` when the per-step deviations
            come from ``kernel_floor.py`` (``kernel_steps.json``).
        steps: Sampled denoising steps of the per-step level.
    """

    canonical: str
    alternatives: tuple[str, ...]
    teacher: tuple[str, ...] | None
    steps: tuple[int, ...]


BASE_STEPS = tuple(range(0, 49, 4))
DIFFUSERS = Family(
    "diffusers",
    ("diffusers_sdpa_flash", "diffusers_sdpa_efficient"),
    None,
    BASE_STEPS,
)
SGLANG = Family(
    "sglang",
    ("sglang_sdpa_flash", "sglang_sdpa_efficient"),
    ("sglang_sdpa_flash_teacher", "sglang_sdpa_efficient_teacher"),
    BASE_STEPS,
)
OMNIREF = Family(
    "fastvideo_omniref",
    ("fastvideo_omniref_triton_layerwise",),
    ("fastvideo_omniref_triton_teacher",),
    tuple(range(8)),
)

# Workloads judged against the diffusers reference.
DIFFUSERS_WORKLOADS = (
    "t2va_16x9_5s",
    "t2va_9x16_5s",
    "t2va_16x9_4s",
    "t2va_16x9_15s",
    "fl2va_first_8s",
    "fl2va_last_8s",
    "fl2va_first_last_8s",
    "ref2va_image_5s",
    "ref2va_image_audio_5s",
    "ref2va_video_audio_5s",
    "ref2va_two_videos_5s",
)

# Workload directory -> its reference family and whether UniServe's runs
# come from ``omniref_parity.py`` (FastH3 OmniRef) or ``uniserve_parity.py``.
WORKLOADS = {
    **dict.fromkeys(DIFFUSERS_WORKLOADS, (DIFFUSERS, False)),
    "ref2va_image_keyframe_5s": (SGLANG, False),
    "omniref/ref2va_video_audio_5s": (OMNIREF, True),
    "omniref/ref2va_video_audio_5s_480x832": (OMNIREF, True),
}


def rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    value, reference = value.double(), reference.double()
    return float((value - reference).norm() / reference.norm())


def critical_count(cells: int) -> int:
    """Smallest K with P(Binomial(cells, 1/2) >= K) <= ALPHA."""
    tail = 0.0
    for k in range(cells, -1, -1):
        tail += math.comb(cells, k) / 2**cells
        if tail > ALPHA:
            return k + 1
    return 0


def _final(run: Path, name: str) -> torch.Tensor:
    """The last recorded sample of a modality's generated rows."""
    return load_file(str(run / "trajectory.safetensors"))[f"{name}_samples"][-1]


def _predictions(run: Path, name: str) -> torch.Tensor:
    return load_file(str(run / "trajectory.safetensors"))[f"{name}_predictions"]


def judge(artifacts: Path, workload: str) -> dict:
    family, omniref = WORKLOADS[workload]
    name = workload.removeprefix("omniref/")
    reference = artifacts / "reference"
    acceptance = artifacts / "acceptance"

    per_step = {modality: {"cells": [], "exceed": 0} for modality in MODALITIES}
    free = {modality: {"valid": [], "uniserve": []} for modality in MODALITIES}
    decoded = []
    for seed in SEEDS:
        canonical = reference / family.canonical / name / f"seed{seed}"
        if omniref:
            ours = acceptance / "parity" / "omniref" / name / f"seed{seed}"
            teacher_metrics = json.loads(
                (ours / "teacher" / "metrics.json").read_text()
            )
            free_run = ours / "free"
        else:
            ours = acceptance / name / f"seed{seed}"
            teacher_metrics = json.loads(
                (ours / "uniserve_teacher" / "metrics.json").read_text()
            )
            free_run = ours / "uniserve_free"

        # Per-step level: the alternatives' deviations on the canonical
        # inputs, from kernel_floor.py or the alternatives' teacher runs.
        if family.teacher is None:
            kernels = json.loads(
                (
                    acceptance / name / f"seed{seed}" / "kernel_steps.json"
                ).read_text()
            )
            index = {
                step: position for position, step in enumerate(kernels["steps"])
            }
            valid = {
                modality: [
                    [
                        kernels[modality][kernel][index[step]]
                        for kernel in ("flash", "efficient")
                    ]
                    for step in family.steps
                ]
                for modality in MODALITIES
            }
        else:
            valid = {
                modality: [[] for _ in family.steps] for modality in MODALITIES
            }
            for impl in family.teacher:
                run = reference / impl / name / f"seed{seed}"
                for modality in MODALITIES:
                    canonical_predictions = _predictions(canonical, modality)
                    predictions = _predictions(run, modality)
                    for position, step in enumerate(family.steps):
                        valid[modality][position].append(
                            rel_l2(
                                predictions[step], canonical_predictions[step]
                            )
                        )
        for modality in MODALITIES:
            ours_steps = teacher_metrics[f"{modality}_prediction_rel_l2"]
            for position, step in enumerate(family.steps):
                bound = max(valid[modality][position])
                exceeded = ours_steps[step] > bound
                per_step[modality]["cells"].append(
                    {
                        "seed": seed,
                        "step": step,
                        "uniserve": ours_steps[step],
                        "valid": valid[modality][position],
                        "exceeds": exceeded,
                    }
                )
                per_step[modality]["exceed"] += int(exceeded)

        # Free-running level: final generated samples against the canonical.
        for modality in MODALITIES:
            final = _final(canonical, modality)
            for impl in family.alternatives:
                run = reference / impl / name / f"seed{seed}"
                free[modality]["valid"].append(
                    {
                        "seed": seed,
                        "impl": impl,
                        "value": rel_l2(_final(run, modality), final),
                    }
                )
            free[modality]["uniserve"].append(
                {
                    "seed": seed,
                    "value": rel_l2(_final(free_run, modality), final),
                }
            )
        if not omniref:
            metrics = compare(canonical, free_run)
            decoded.append(
                {
                    "seed": seed,
                    **{
                        key: metrics[key]
                        for key in (
                            "video_psnr_db",
                            "video_ssim",
                            "audio_spectral_cosine",
                            "refined_text_rel_l2",
                        )
                        if key in metrics
                    },
                }
            )

    verdict = {
        "workload": workload,
        "reference": family.canonical,
        "seeds": list(SEEDS),
    }
    passed = True
    for modality in MODALITIES:
        cells = len(per_step[modality]["cells"])
        limit = critical_count(cells)
        step_pass = per_step[modality]["exceed"] < limit
        bound = max(item["value"] for item in free[modality]["valid"])
        above = sum(
            item["value"] > bound for item in free[modality]["uniserve"]
        )
        free_pass = above < len(free[modality]["uniserve"])
        passed &= step_pass and free_pass
        verdict[modality] = {
            "per_step": {
                "cells": cells,
                "exceeding": per_step[modality]["exceed"],
                "fail_at": limit,
                "pass": step_pass,
                "detail": per_step[modality]["cells"],
            },
            "free_running": {
                "valid_max": bound,
                "uniserve": free[modality]["uniserve"],
                "valid": free[modality]["valid"],
                "seeds_above": above,
                "pass": free_pass,
            },
        }
    verdict["decoded"] = decoded
    verdict["pass"] = passed
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--workload", action="append", choices=sorted(WORKLOADS)
    )
    args = parser.parse_args()

    verdicts = {}
    for workload in args.workload or WORKLOADS:
        try:
            verdict = judge(args.artifacts, workload)
        except FileNotFoundError as missing:
            print(f"{workload}: incomplete ({missing.filename})")
            continue
        verdicts[workload] = verdict
        directory = args.artifacts / "acceptance" / workload
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verdict.json").write_text(
            json.dumps(verdict, indent=2) + "\n"
        )
        row = [workload]
        for modality in MODALITIES:
            step = verdict[modality]["per_step"]
            run = verdict[modality]["free_running"]
            uniserve = max(item["value"] for item in run["uniserve"])
            row.append(
                f"{modality}: K {step['exceeding']}/{step['cells']} "
                f"(fail at {step['fail_at']}), "
                f"free max {uniserve:.4f} vs valid max {run['valid_max']:.4f} "
                f"({run['seeds_above']}/4 above)"
            )
        row.append("PASS" if verdict["pass"] else "FAIL")
        print(" | ".join(row))
    if not args.workload:
        (args.artifacts / "acceptance" / "verdicts.json").write_text(
            json.dumps(verdicts, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
