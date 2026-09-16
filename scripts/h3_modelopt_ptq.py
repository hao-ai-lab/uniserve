#!/usr/bin/env python3
"""Prepare and execute the fixed MiniMax H3 ModelOpt PTQ protocol.

The ``prepare`` command is intentionally independent of CUDA/model loading. It
freezes the exact 1,000-record calibration cohort, its balanced Phase-B text
assignment, coverage statistics, provenance, and the predeclared acceptance
criteria before any calibration result exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct ``python scripts/...`` execution otherwise exposes only scripts/.
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA = ROOT / "artifacts/h3_t2va_prompts_10k.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/h3-modelopt-ptq"
MODEL = Path(
    "/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
)
MODELOPT_COMMIT = "6a4b3f147e14a6fec690fedbced8df402344085d"
SAMPLE_SEED = "h3-modelopt-ptq-calibration-v1"
INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> tuple[list[dict], list[int]]:
    records, recovered = [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                repaired = INVALID_ESCAPE.sub(r"\\\\", line)
                record = json.loads(repaired)
                recovered.append(line_number)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            record["_source_line"] = line_number
            records.append(record)
    if len({record.get("id") for record in records}) != len(records):
        raise ValueError("calibration record ids must be unique")
    return records, recovered


def _largest_remainder(values: list[str], count: int) -> dict[str, int]:
    frequencies = Counter(values)
    total = len(values)
    raw = {key: value * count / total for key, value in frequencies.items()}
    target = {key: int(value) for key, value in raw.items()}
    order = sorted(
        raw,
        key=lambda key: (
            -(raw[key] - target[key]),
            hashlib.sha256(f"{SAMPLE_SEED}:{key}".encode()).hexdigest(),
        ),
    )
    for key in order[: count - sum(target.values())]:
        target[key] += 1
    return target


def _prompt_length_bins(records: list[dict]) -> tuple[list[int], list[str]]:
    lengths = sorted(len(record["prompt_compiled"]) for record in records)
    boundaries = [lengths[len(lengths) * index // 5] for index in range(1, 5)]
    labels = []
    for record in records:
        value = len(record["prompt_compiled"])
        labels.append(str(sum(value >= boundary for boundary in boundaries)))
    return boundaries, labels


def _fields(records: list[dict]) -> tuple[dict[str, list[str]], list[int]]:
    dimensions = sorted(records[0]["sampling"]["dimensions"])
    fields = {
        f"dimension.{name}": [
            str(record["sampling"]["dimensions"][name]) for record in records
        ]
        for name in dimensions
    }
    runtime = {
        "runtime.resolution": [
            f"{record['runtime_config']['width']}x{record['runtime_config']['height']}"
            for record in records
        ],
        "runtime.frames": [
            str(record["runtime_config"]["num_frames"]) for record in records
        ],
        "runtime.fps": [
            str(record["runtime_config"]["fps"]) for record in records
        ],
        "runtime.audio": [
            str(bool(record["runtime_config"]["generate_audio"])).lower()
            for record in records
        ],
    }
    boundaries, bins = _prompt_length_bins(records)
    return {**fields, **runtime, "prompt.character_quintile": bins}, boundaries


def _balanced_indices(
    fields: dict[str, list[str]], count: int, *, seed: str
) -> tuple[list[int], dict[str, dict[str, int]]]:
    size = len(next(iter(fields.values())))
    order = sorted(
        range(size),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )
    rows, columns, lower = [], [], []
    targets: dict[str, dict[str, int]] = {}
    row = 0
    for field, values in fields.items():
        targets[field] = _largest_remainder(values, count)
        for category, target in sorted(targets[field].items()):
            selected = [
                column
                for column, original in enumerate(order)
                if values[original] == category
            ]
            rows.extend([row] * len(selected))
            columns.extend(selected)
            lower.append(target)
            row += 1
    matrix = csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(row, size)
    )
    result = milp(
        np.zeros(size),
        integrality=np.ones(size),
        bounds=Bounds(np.zeros(size), np.ones(size)),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(lower)),
        options={"presolve": True, "time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(
            f"balanced cohort optimization failed: {result.message}"
        )
    selected = sorted(
        order[column] for column in np.flatnonzero(result.x > 0.5)
    )
    if len(selected) != count:
        raise RuntimeError(
            f"expected {count} balanced records, got {len(selected)}"
        )
    return selected, targets


def _distribution(fields: dict[str, list[str]], indices: list[int]) -> dict:
    return {
        field: dict(sorted(Counter(values[index] for index in indices).items()))
        for field, values in fields.items()
    }


def _token_counts(
    model: Path, records: list[dict], indices: list[int]
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer")
    return [
        len(
            tokenizer.encode(
                records[index]["prompt_compiled"], add_special_tokens=True
            )
        )
        for index in indices
    ]


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verify_modelopt_revision() -> None:
    actual = _git_revision(ROOT / "refs/Model-Optimizer")
    if actual != MODELOPT_COMMIT:
        raise RuntimeError(
            f"ModelOpt must be pinned to {MODELOPT_COMMIT}, found {actual!r}"
        )


def _environment(data: Path, model: Path, model_revision: str | None) -> dict:
    import torch

    try:
        driver = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        driver = []
    try:
        import flashinfer

        flashinfer_version = flashinfer.__version__
    except (ImportError, AttributeError):
        flashinfer_version = None
    packages = {}
    for name in (
        "fastvideo",
        "fastvideo-kernel",
        "nvidia-cutlass-dsl",
        "nvidia-modelopt",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    provenance_path = model / "provenance.json"
    model_provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.is_file()
        else None
    )
    inference_contract = json.loads(
        (model / "fastvideo_inference.json").read_text()
    )
    return {
        "created_before_calibration": True,
        "model_path": str(model),
        "model_revision": model_revision,
        "uniserve_revision": _git_revision(ROOT),
        "fastvideo_revision": _git_revision(ROOT / "refs/FastVideo"),
        "modelopt_revision": _git_revision(ROOT / "refs/Model-Optimizer"),
        "required_modelopt_revision": MODELOPT_COMMIT,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer_version,
        "packages": packages,
        "model_provenance": model_provenance,
        "inference_contract": inference_contract,
        "gpus": driver,
        "calibration_source": str(data),
        "calibration_source_sha256": _sha256(data),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def prepare(
    data: Path,
    output: Path,
    *,
    model: Path = MODEL,
    model_revision: str | None = None,
    acceptance_protocol: Path = ROOT / "specs/h3-modelopt-ptq-protocol.md",
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    records, recovered = _load_records(data)
    if len(records) != 10_000:
        raise ValueError(
            f"fixed source must contain 10,000 records, found {len(records)}"
        )
    fields, character_boundaries = _fields(records)
    selected, targets = _balanced_indices(fields, 1_000, seed=SAMPLE_SEED)
    selected_records = [records[index] for index in selected]

    phase_b_fields = {
        "dimension.content_language",
        "dimension.duration_bucket",
        "dimension.motion_complexity",
        "runtime.resolution",
        "runtime.audio",
        "prompt.character_quintile",
    }
    selected_fields = {
        field: [values[index] for index in selected]
        for field, values in fields.items()
        if field in phase_b_fields
    }
    bf16_local, phase_b_targets = _balanced_indices(
        selected_fields, 500, seed=SAMPLE_SEED + ":phase-b-text-bf16"
    )
    bf16 = set(bf16_local)
    token_counts = _token_counts(model, records, selected)
    sample_rows = []
    for local_index, (source_index, record, tokens) in enumerate(
        zip(selected, selected_records, token_counts, strict=True)
    ):
        clean = {
            key: value for key, value in record.items() if key != "_source_line"
        }
        clean["calibration"] = {
            "sample_index": local_index,
            "source_line": record["_source_line"],
            "source_record_index": source_index,
            "prompt_tokens": tokens,
            "phase_b_text": "bf16" if local_index in bf16 else "nvfp4",
        }
        sample_rows.append(clean)

    protocol = output / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    sample_path = protocol / "calibration-sample-1000.jsonl"
    sample_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows
        ),
        encoding="utf-8",
    )
    coverage = {
        "algorithm": (
            "binary MILP exact marginal balancing; SHA-256 seeded variable "
            "order"
        ),
        "seed": SAMPLE_SEED,
        "source_records": len(records),
        "selected_records": len(selected),
        "json_escape_recovery_lines": recovered,
        "character_quintile_boundaries": character_boundaries,
        "source_distribution": _distribution(fields, list(range(len(records)))),
        "sample_distribution": _distribution(fields, selected),
        "sample_targets": targets,
        "phase_b_text_bf16_distribution": _distribution(
            selected_fields, sorted(bf16)
        ),
        "phase_b_text_nvfp4_distribution": _distribution(
            selected_fields, sorted(set(range(1_000)) - bf16)
        ),
        "phase_b_half_targets": phase_b_targets,
        "prompt_tokens": {
            "minimum": min(token_counts),
            "maximum": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "p50": sorted(token_counts)[500],
            "p90": sorted(token_counts)[900],
            "p99": sorted(token_counts)[990],
        },
    }
    _write_json(protocol / "calibration-coverage.json", coverage)
    provenance = _environment(data, model, model_revision)
    provenance.update(
        {
            "sample_manifest": str(sample_path),
            "sample_manifest_sha256": _sha256(sample_path),
            "requested_records": 1_000,
            "parsed_records": len(records),
            "selected_records": len(selected),
            "evaluation_prompts_sha256": _sha256(
                protocol / "evaluation-prompts.jsonl"
            ),
            "acceptance_protocol_sha256": _sha256(acceptance_protocol),
            "acceptance_protocol": str(acceptance_protocol),
        }
    )
    _write_json(protocol / "provenance.json", provenance)
    print(f"wrote {sample_path} ({_sha256(sample_path)})")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--model", type=Path, default=MODEL)
    prepare_parser.add_argument("--model-revision")
    prepare_parser.add_argument(
        "--acceptance-protocol",
        type=Path,
        default=ROOT / "specs/h3-modelopt-ptq-protocol.md",
    )
    for command in ("smoke", "phase-a", "phase-b"):
        run_parser = subparsers.add_parser(command)
        run_parser.add_argument("--model", type=Path, default=MODEL)
        run_parser.add_argument(
            "--sample",
            type=Path,
            default=DEFAULT_OUTPUT / "protocol/calibration-sample-1000.jsonl",
        )
        run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        run_parser.add_argument("--num-gpus", type=int, default=4)
    export_parser = subparsers.add_parser("export-component")
    export_parser.add_argument(
        "component", choices=("denoiser", "text_encoder", "video_vae")
    )
    export_parser.add_argument("--model", type=Path, default=MODEL)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    candidates_parser = subparsers.add_parser("build-candidates")
    candidates_parser.add_argument("--model", type=Path, default=MODEL)
    candidates_parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    candidates_parser.add_argument(
        "--text-encoder",
        choices=("bf16", "nvfp4", "both"),
        default="both",
    )
    candidates_parser.add_argument(
        "--video-vae",
        choices=("bf16", "nvfp4"),
        default="nvfp4",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.data,
            args.output,
            model=args.model,
            model_revision=args.model_revision,
            acceptance_protocol=args.acceptance_protocol,
        )
    elif args.command == "export-component":
        _verify_modelopt_revision()
        from uniserve_eval.h3_modelopt_export import export_component

        report = export_component(
            args.model,
            args.output / "checkpoints/components" / args.component,
            args.component,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "build-candidates":
        _verify_modelopt_revision()
        from uniserve_eval.h3_modelopt_calibration import (
            load_h3_inference_contract,
        )
        from uniserve_eval.h3_modelopt_export import (
            build_candidate,
            build_manifest,
        )

        protocol = args.output / "protocol"
        sample = protocol / "calibration-sample-1000.jsonl"
        phase_a = args.output / "phase-a/activation-scales.json"
        phase_b = (
            args.output / "phase-b/activation-scales.json"
            if args.video_vae == "nvfp4"
            else None
        )
        contract = load_h3_inference_contract(args.model)
        common = {
            "phase_a_scales": phase_a,
            "phase_b_scales": phase_b,
            "calibration_sha256": _sha256(sample),
            "denoising_forwards_per_record": int(
                contract["transformer_forwards"]
            ),
        }
        candidates = args.output / "checkpoints/candidates"
        variants = {
            "bf16": (("candidate-a-text-bf16", False),),
            "nvfp4": (("candidate-b-text-nvfp4", True),),
            "both": (
                ("candidate-a-text-bf16", False),
                ("candidate-b-text-nvfp4", True),
            ),
        }[args.text_encoder]
        for name, text_nvfp4 in variants:
            if args.video_vae == "bf16":
                name += "-vae-bf16"
            manifest = build_manifest(text_nvfp4=text_nvfp4, **common)
            build_candidate(
                model=args.model,
                components=args.output / "checkpoints/components",
                destination=candidates / name,
                manifest=manifest,
            )
        print(str(candidates))
    else:
        _verify_modelopt_revision()
        from uniserve_eval.h3_modelopt_calibration import (
            run_activation_calibration,
        )

        phase = "smoke" if args.command == "smoke" else args.command
        directory = (
            args.output / "smoke" if phase == "smoke" else args.output / phase
        )
        result = run_activation_calibration(
            model=args.model,
            sample=args.sample,
            output=directory,
            phase=phase,
            num_gpus=args.num_gpus,
            limit=1 if phase == "smoke" else None,
            resume=phase != "smoke",
            deployment_scales=(args.output / "phase-a/activation-scales.json")
            if phase == "phase-b"
            else None,
        )
        if phase == "smoke":
            from uniserve_eval.h3_modelopt_export import smoke_export

            result["export"] = smoke_export(args.model, directory / "export")
            _write_json(directory / "result.json", result)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
