"""Measure the FastH3 OmniRef trajectory tolerance envelope.

The OmniRef trajectory thresholds come from independent evidence: the
deviation between two equally valid block-sparse kernels of the FastVideo
reference on identical inputs. VSA-H3's 128-token tiles have two
implementations in FastVideo's kernel package: the sm_100a CUDA kernel of
the canonical ``fastvideo_omniref`` runs, and ``block_sparse_attn_128``'s
Triton route, which runs the same block map on the 64-token Triton kernel
(``fastvideo_reference.py --vsa-kernel triton``). The Triton runs replay the
canonical run's noise and Qwen encoder output and stream the DiT layer by
layer to fit one process in 80 GB; their refined text, condition latents and
initial samples are bitwise those of the canonical run, so the kernel is the
only difference.

Two Triton runs exist per workload. A teacher-forced run
(``fastvideo_omniref_triton_teacher``, ``--teacher-trajectory``) starts every
step from the canonical run's recorded sample, so each step's prediction and
one-step update deviate by the kernel alone at that step; this is the
per-step parity threshold. A free-running run
(``fastvideo_omniref_triton_layerwise``) lets the trajectories diverge and
gives the free-running final-latent, decoded-media and per-step
observations.

Writes, for every workload directory present, the per-pair metrics of
``trajectory_metrics.py`` and the worst value of every metric over the
seeds and the elementwise maximum of every per-step relative L2 curve to
``<output>/omniref_envelope.json``.

Usage: ``omniref_envelope.py --output artifacts/minimax_h3``.
"""

import argparse
import json
from pathlib import Path

from tolerance_envelope import _finite, worst
from trajectory_metrics import compare, summarize

CANONICAL = "fastvideo_omniref"
VARIANTS = {
    "teacher_forced": "fastvideo_omniref_triton_teacher",
    "free_running": "fastvideo_omniref_triton_layerwise",
}
# Teacher forcing overwrites every step's result with the canonical sample,
# so only its per-step curves describe the kernel.
PER_STEP_ONLY = {"teacher_forced"}


def _envelope(reference: Path, variant: str, per_step_only: bool) -> dict:
    workloads = {}
    for workload in sorted((reference / variant).iterdir()):
        pairs = {}
        for run in sorted(workload.glob("seed*")):
            canonical = reference / CANONICAL / workload.name / run.name
            if (run / "metadata.json").exists():
                pairs[run.name] = compare(canonical, run)
        if not pairs:
            continue
        results = list(pairs.values())
        curves = {
            key: [max(values) for values in zip(*(r[key] for r in results))]
            for key, value in results[0].items()
            if isinstance(value, list)
        }
        entry = {"pairs": pairs, "envelope_per_step": curves}
        if not per_step_only:
            summaries = {name: summarize(r) for name, r in pairs.items()}
            entry["summaries"] = summaries
            entry["envelope"] = worst(list(summaries.values()))
        workloads[workload.name] = entry
    return workloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = args.output / "reference"
    document = {
        "kernels": {
            "reference": "fastvideo-kernel block_sparse_attn_sm100a, "
            "128-token blocks",
            "candidate": "fastvideo-kernel block_sparse_attn_128, Triton "
            "route (64-token kernel on the expanded block map)",
        },
        **{
            mode: _envelope(reference, variant, mode in PER_STEP_ONLY)
            for mode, variant in VARIANTS.items()
            if (reference / variant).is_dir()
        },
    }
    if not any(document.get(mode) for mode in VARIANTS):
        raise SystemExit("no sm100a/Triton OmniRef pair found")
    text = json.dumps(_finite(document), indent=2, allow_nan=False)
    (args.output / "omniref_envelope.json").write_text(text + "\n")
    for mode in VARIANTS:
        for name, workload in document.get(mode, {}).items():
            print(mode, name)
            for key, values in workload["envelope_per_step"].items():
                print(f"  {key}: {[f'{value:.4g}' for value in values]}")


if __name__ == "__main__":
    main()
