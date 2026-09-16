#!/usr/bin/env python3
"""Evaluate retained Qwen MLP NVFP4 on independent held-out prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from uniserve_eval.h3_modelopt_calibration import (  # noqa: E402
    _configure_environment,
    _generator,
    setup_deployment_quant,
    text_conditioning_conformance,
)


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        name: statistics.mean(float(row[name]) for row in rows)
        for name in ("nrmse", "cosine", "maximum_absolute")
    }


def _language(record: dict) -> str:
    tags = set(record["tags"])
    for name in ("mixed_language", "chinese", "english", "japanese"):
        if name in tags:
            return name
    return "nonverbal_or_other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    records = [
        json.loads(line) for line in args.prompts.read_text().splitlines()
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer")
    token_ids = [
        tokenizer.encode(record["prompt_compiled"], add_special_tokens=False)
        for record in records
    ]
    ordered_lengths = sorted(len(value) for value in token_ids)
    boundaries = [
        ordered_lengths[len(records) * index // 5] for index in range(1, 5)
    ]
    scales = json.loads(args.scales.read_text())
    text_scales = {
        name: value
        for name, value in scales.items()
        if name.startswith("text_encoder.")
    }

    _configure_environment()
    generator = _generator(
        args.model,
        output_type="latent",
        num_gpus=args.num_gpus,
        text_encoder_offload=False,
    )
    prompts = []
    try:
        generator.executor.collective_rpc(
            setup_deployment_quant,
            kwargs={
                "activation_scales": scales,
                "components": ("text_encoder",),
            },
        )
        for record, ids in zip(records, token_ids, strict=True):
            ranks = generator.executor.collective_rpc(
                text_conditioning_conformance,
                kwargs={"input_ids": ids, "activation_scales": text_scales},
            )
            rank_zero = next(item for item in ranks if item["rank"] == 0)
            prompts.append(
                {
                    "id": record["id"],
                    "language": _language(record),
                    "prompt_length_quintile": sum(
                        len(ids) >= boundary for boundary in boundaries
                    ),
                    "tokens": len(ids),
                    "final_hidden_state": rank_zero["final_hidden_state"],
                    "token_metrics": rank_zero["token_metrics"],
                    "ranks": ranks,
                }
            )
            print(
                "[text-conformance] "
                f"{len(prompts)}/{len(records)} {record['id']}",
                flush=True,
            )
    finally:
        generator.shutdown()

    grouped = {
        "overall": _mean_metrics([row["final_hidden_state"] for row in prompts])
    }
    for field in ("language", "prompt_length_quintile"):
        values = defaultdict(list)
        for row in prompts:
            values[str(row[field])].append(row["final_hidden_state"])
        grouped[field] = {
            name: {"prompts": len(rows), **_mean_metrics(rows)}
            for name, rows in sorted(values.items())
        }

    projection_rows = defaultdict(list)
    saturation = defaultdict(lambda: {"clipped": 0, "values": 0})
    for prompt in prompts:
        for rank in prompt["ranks"]:
            for name, metrics in rank["representative_projections"].items():
                projection_rows[name].append(metrics)
            for name, counts in rank["activation_saturation"].items():
                saturation[name]["clipped"] += counts["clipped"]
                saturation[name]["values"] += counts["values"]
    saturation_report = {
        name: {
            **counts,
            "rate": counts["clipped"] / counts["values"]
            if counts["values"]
            else 0.0,
        }
        for name, counts in sorted(saturation.items())
    }
    result = {
        "protocol": {
            "source": str(args.prompts),
            "tokenizer": str(args.model / "tokenizer"),
            "add_special_tokens": False,
            "reference": (
                "retained-layer raw BF16 hidden state without final norm"
            ),
            "candidate": "ModelOpt calibrated MLP-only NVFP4 fake quant",
            "acceptance": {"cosine_minimum": 0.98, "nrmse_maximum": 0.20},
        },
        "prompt_length_quintile_boundaries": boundaries,
        "prompts": prompts,
        "summary": grouped,
        "representative_projections": {
            name: _mean_metrics(rows)
            for name, rows in sorted(projection_rows.items())
        },
        "activation_saturation": saturation_report,
        "maximum_saturation_rate": max(
            (row["rate"] for row in saturation_report.values()), default=0.0
        ),
    }
    overall = grouped["overall"]
    result["passed_tensor_gate"] = (
        overall["cosine"] >= 0.98
        and overall["nrmse"] <= 0.20
        and result["maximum_saturation_rate"] <= 0.001
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed_tensor_gate"]:
        raise RuntimeError("held-out text conditioning tensor gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
