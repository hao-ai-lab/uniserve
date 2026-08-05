"""Summary assembly + human-readable report for a single benchmark run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .artifacts import ArtifactWriter
from .image_outputs import (
    ImageOutputContract,
    ImageOutputMode,
    image_output_mismatch,
    inspect_image_bytes,
)
from .metrics import RequestRecord, summarize_image, summarize_stream
from .spec import IMAGE_TASKS, BenchmarkSpec, TaskName


def spec_to_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["task"] = spec.task.value
    rate = data.get("request_rate")
    if isinstance(rate, float) and math.isinf(rate):
        data["request_rate"] = "inf"
    return cast(
        dict[str, Any],
        json.loads(
            json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        ),
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_contract(
    spec: BenchmarkSpec,
    selected_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = {
        "count": len(selected_rows),
        "sha256": canonical_digest(list(selected_rows)),
    }
    payload = {
        "schema_version": 1,
        "spec": spec_to_dict(spec),
        "selected_rows": selected,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


_PARITY_IDENTITY_FIELDS = frozenset(
    {"dataset_path", "model", "name", "runtime_profile_id", "request_schema"}
)


def benchmark_parity_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Normalize a harness contract to backend-independent comparison semantics."""
    spec = contract.get("spec")
    selected_rows = contract.get("selected_rows")
    if not isinstance(spec, dict) or not isinstance(selected_rows, dict):
        raise ValueError("benchmark contract has no normalized spec or selected rows")
    normalized_spec = {
        key: value for key, value in spec.items() if key not in _PARITY_IDENTITY_FIELDS
    }
    # Some backends expose the number of scheduler points while others expose
    # the number of actual denoiser updates.  The point-local harness contract
    # still binds the raw ``steps`` value sent on each wire, but cross-backend
    # parity is defined by the explicitly declared semantic work.
    if normalized_spec.get("denoise_updates") is not None:
        normalized_spec.pop("steps", None)
    payload = {
        "schema_version": 1,
        "spec": normalized_spec,
        "selected_rows": selected_rows,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def benchmark_contract_is_valid(
    contract: Any,
    spec: BenchmarkSpec,
    request_count: int,
) -> bool:
    if not isinstance(contract, dict):
        return False
    selected = contract.get("selected_rows")
    if not isinstance(selected, dict):
        return False
    payload = {
        "schema_version": contract.get("schema_version"),
        "spec": contract.get("spec"),
        "selected_rows": selected,
    }
    return bool(
        contract.get("schema_version") == 1
        and contract.get("spec") == spec_to_dict(spec)
        and selected.get("count") == request_count
        and isinstance(selected.get("sha256"), str)
        and len(selected["sha256"]) == 64
        and contract.get("fingerprint") == canonical_digest(payload)
    )


def artifact_summary_matches(
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        return False
    checks = artifact.get("checks")
    if not (
        artifact.get("schema_version") == 4
        and artifact.get("valid") is True
        and artifact.get("valid_marker") in {"artifact-valid-v4", "canonical-valid-v4"}
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
        and artifact.get("contract") == expected_contract
    ):
        return False
    if profile_contract_fingerprint is None:
        return True
    profile_contract = artifact.get("profile_contract")
    return bool(
        isinstance(profile_contract, dict)
        and profile_contract.get("schema_version") == 2
        and profile_contract.get("fingerprint") == profile_contract_fingerprint
    )


def canonical_summary_matches(
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
) -> bool:
    if not artifact_summary_matches(
        summary,
        expected_contract,
        profile_contract_fingerprint=profile_contract_fingerprint,
    ):
        return False
    assert isinstance(summary, dict)
    return summary["artifact"].get("valid_marker") == "canonical-valid-v4"


def record_collection_contract(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    payload = list(records)
    return {
        "schema_version": 1,
        "count": len(payload),
        "sha256": canonical_digest(payload),
    }


def image_sample_collection_contract(
    request_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Bind every generated-image reference to its unique sample file."""
    references: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for request in request_records:
        request_id = request.get("request_id")
        outputs = request.get("image_outputs", [])
        if not isinstance(request_id, str) or not isinstance(outputs, list):
            raise ValueError("request record has invalid image output metadata")
        for index, output in enumerate(outputs):
            metadata = _normalized_image_metadata(output)
            filename = metadata["sample_filename"]
            existing = files.get(filename)
            if existing is not None and existing != metadata:
                raise ValueError("one image sample filename has conflicting metadata")
            files[filename] = metadata
            references.append(
                {
                    "request_id": request_id,
                    "image_index": index,
                    "sample_filename": filename,
                }
            )
    ordered_files = [files[name] for name in sorted(files)]
    payload = {"references": references, "files": ordered_files}
    return {
        "schema_version": 1,
        "reference_count": len(references),
        "file_count": len(ordered_files),
        "sha256": canonical_digest(payload),
        "files": ordered_files,
    }


def summary_payload_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Bind every result field while excluding the artifact envelope itself."""
    payload = {key: value for key, value in summary.items() if key != "artifact"}
    return {
        "schema_version": 1,
        "sha256": canonical_digest(payload),
    }


def _artifact_bundle_matches(
    output_dir: str | Path,
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
    require_canonical: bool,
) -> bool:
    """Validate every durable member of a committed benchmark point."""
    summary_matches = canonical_summary_matches if require_canonical else artifact_summary_matches
    if not summary_matches(
        summary,
        expected_contract,
        profile_contract_fingerprint=profile_contract_fingerprint,
    ):
        return False
    assert isinstance(summary, dict)
    artifact = summary["artifact"]
    output_path = Path(output_dir)
    try:
        manifest = json.loads((output_path / "artifact_manifest.json").read_text(encoding="utf-8"))
        run = json.loads((output_path / "run.json").read_text(encoding="utf-8"))
        requests = _read_jsonl(output_path / "requests.jsonl")
        gpu_samples = _read_jsonl(output_path / "gpu_samples.jsonl")
        image_samples = image_sample_collection_contract(requests)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    if manifest != artifact:
        return False
    if not (
        run.get("harness_status") == "completed"
        and run.get("artifact_valid") is True
        and run.get("items") == summary.get("request_count")
        and run.get("spec") == summary.get("spec")
        and run.get("base_url") == summary.get("base_url")
    ):
        return False
    return bool(
        artifact.get("request_records") == record_collection_contract(requests)
        and artifact.get("gpu_samples") == record_collection_contract(gpu_samples)
        and artifact.get("image_samples") == image_samples
        and _image_sample_files_match(output_path / "samples", image_samples)
        and artifact.get("summary_payload") == summary_payload_contract(summary)
    )


def artifact_bundle_matches(
    output_dir: str | Path,
    summary: Any,
    expected_contract: dict[str, Any],
) -> bool:
    """Validate a harness-owned bundle before a profile or matrix canonicalizes it."""

    return _artifact_bundle_matches(
        output_dir,
        summary,
        expected_contract,
        require_canonical=False,
    )


def canonical_artifact_bundle_matches(
    output_dir: str | Path,
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
) -> bool:
    """Validate a bundle carrying its profile or matrix execution contract."""

    return _artifact_bundle_matches(
        output_dir,
        summary,
        expected_contract,
        profile_contract_fingerprint=profile_contract_fingerprint,
        require_canonical=True,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object record")
            records.append(value)
    return records


def _image_sample_files_match(samples_dir: Path, contract: dict[str, Any]) -> bool:
    files = contract.get("files")
    if not isinstance(files, list) or not samples_dir.is_dir() or samples_dir.is_symlink():
        return False
    try:
        entries = list(samples_dir.iterdir())
    except OSError:
        return False
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        return False
    expected_names = {
        metadata.get("sample_filename") for metadata in files if isinstance(metadata, dict)
    }
    if len(expected_names) != len(files) or {entry.name for entry in entries} != expected_names:
        return False
    try:
        for metadata in files:
            normalized = _normalized_image_metadata(metadata)
            data = (samples_dir / normalized["sample_filename"]).read_bytes()
            decoded = inspect_image_bytes(data, declared_mime=normalized["mime"])
            if decoded.metadata_dict() != normalized:
                return False
    except (OSError, ValueError):
        return False
    return True


def _normalized_image_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("image output metadata is not an object")
    metadata = {
        "sha256": value.get("sha256"),
        "byte_size": value.get("byte_size"),
        "mime": value.get("mime"),
        "width": value.get("width"),
        "height": value.get("height"),
        "sample_filename": value.get("sample_filename"),
    }
    digest = metadata["sha256"]
    byte_size = metadata["byte_size"]
    width = metadata["width"]
    height = metadata["height"]
    filename = metadata["sample_filename"]
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and isinstance(byte_size, int)
        and not isinstance(byte_size, bool)
        and byte_size > 0
        and isinstance(metadata["mime"], str)
        and metadata["mime"].startswith("image/")
        and isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
        and isinstance(filename, str)
        and filename.startswith(f"{digest}.")
        and "/" not in filename
        and "\\" not in filename
    ):
        raise ValueError("image output metadata is invalid")
    return metadata


def attach_execution_contract(
    summary: dict[str, Any],
    name: str,
    contract: dict[str, Any],
    *,
    valid: bool = True,
) -> None:
    """Attach execution evidence and recompute the canonical validity marker."""
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("benchmark summary has no artifact contract")
    checks = artifact.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("benchmark artifact has no validity checks")
    was_canonical = artifact.get("valid_marker") == "canonical-valid-v4"
    artifact[name] = contract
    checks[name] = valid
    artifact["valid"] = all(value is True for value in checks.values())
    canonical = was_canonical or name in {"matrix_contract", "profile_contract"}
    artifact["valid_marker"] = (
        ("canonical-valid-v4" if canonical else "artifact-valid-v4") if artifact["valid"] else None
    )


def write_summary_artifacts(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Persist the manifest first and summary last as the point's commit marker."""
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("benchmark summary has no artifact contract")
    if isinstance(artifact.get("checks"), dict):
        if "image_samples" not in artifact:
            requests_path = Path(output_dir) / "requests.jsonl"
            requests = _read_jsonl(requests_path) if requests_path.is_file() else []
            attach_execution_contract(
                summary,
                "image_samples",
                image_sample_collection_contract(requests),
            )
        attach_execution_contract(summary, "summary_payload", summary_payload_contract(summary))
    else:
        artifact["summary_payload"] = summary_payload_contract(summary)
    writer = ArtifactWriter(output_dir)
    writer.write_json("artifact_manifest.json", artifact)
    writer.write_json("summary.json", summary)


def load_mode(spec: BenchmarkSpec) -> str:
    if spec.request_rate != float("inf"):
        return "open_loop_poisson"
    if spec.max_concurrency == 1:
        return "single_stream"
    if spec.max_concurrency:
        return "saturated_concurrency"
    return "saturation"


def artifact_contract(
    spec: BenchmarkSpec,
    *,
    request_count: int,
    ok_count: int,
    failed_count: int,
    total_images: int,
    contract: dict[str, Any] | None,
    generation_conformance: dict[str, Any],
    interleave_latency_conformance: dict[str, Any] | None,
    warnings: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "declared_request_count": request_count == spec.num_prompts,
        "minimum_successful_requests": ok_count >= spec.acceptance_min_success,
        "maximum_failed_requests": failed_count <= spec.acceptance_max_failed,
        "single_measured_run": spec.measured_runs == 1,
        "minimum_images": total_images
        >= math.ceil(ok_count * spec.acceptance_min_images_per_success),
        "contract_fingerprint": benchmark_contract_is_valid(contract, spec, request_count),
        "generation_conformance": generation_conformance.get("valid") is True,
    }
    if interleave_latency_conformance is not None:
        checks["interleave_latency_conformance"] = (
            interleave_latency_conformance.get("valid") is True
        )
    valid = all(checks.values())
    artifact: dict[str, Any] = {
        "schema_version": 4,
        "valid": valid,
        "valid_marker": "artifact-valid-v4" if valid else None,
        "checks": checks,
        "acceptance": {
            "minimum_successful_requests": spec.acceptance_min_success,
            "maximum_failed_requests": spec.acceptance_max_failed,
            "minimum_images_per_success": spec.acceptance_min_images_per_success,
        },
        "measured_runs": spec.measured_runs,
        "server_topology": spec.server_topology,
        "contract": contract,
        "generation_conformance": generation_conformance,
        "warnings": warnings,
    }
    if interleave_latency_conformance is not None:
        artifact["interleave_latency_conformance"] = interleave_latency_conformance
    return artifact


def build_summary(
    spec: BenchmarkSpec,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
    server_info: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [r for r in records if r.success]
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1

    if spec.is_stream_task:
        family = "stream"
        metrics = summarize_stream(records, dur_s, tokenizer=tokenizer)
    else:
        family = "image"
        metrics = summarize_image(records, dur_s)
    warnings = _warning_summary(records, metrics)
    endpoint = _observed_endpoint(records) or spec.endpoint

    summary: dict[str, Any] = {
        "harness_status": "completed",
        "task": spec.task.value,
        "dataset": spec.dataset,
        "endpoint": endpoint,
        "wire": spec.wire,
        "model": spec.model,
        "base_url": base_url,
        "server_info": server_info,
        "spec": spec_to_dict(spec),
        "load": {
            "mode": load_mode(spec),
            "request_rate": "inf" if spec.request_rate == float("inf") else spec.request_rate,
            "max_concurrency": spec.max_concurrency,
            "warmup_requests": spec.warmup_requests,
            "num_prompts": spec.num_prompts,
            "seed": spec.seed,
        },
        "elapsed_s": dur_s,
        "request_count": len(records),
        "ok_count": len(ok),
        "failed_count": len(records) - len(ok),
        "classifiers": classifiers,
        "warnings": warnings,
        "metric_family": family,
        "metrics": metrics,
    }
    generation_conformance = _generation_conformance(spec, records)
    summary["artifact"] = artifact_contract(
        spec,
        request_count=summary["request_count"],
        ok_count=summary["ok_count"],
        failed_count=summary["failed_count"],
        total_images=_completed_images(family, metrics),
        contract=contract,
        generation_conformance=generation_conformance,
        interleave_latency_conformance=_interleave_latency_conformance(spec, metrics),
        warnings=warnings,
    )
    if spec.runtime_profile_id != "unspecified":
        server_info_valid = bool(
            isinstance(server_info, dict)
            and isinstance(server_info.get("source_endpoint"), str)
            and server_info.get("source_endpoint")
            and isinstance(server_info.get("payload"), dict)
            and server_info.get("payload")
        )
        summary["artifact"]["checks"]["server_info"] = server_info_valid
        summary["artifact"]["valid"] = all(
            value is True for value in summary["artifact"]["checks"].values()
        )
        summary["artifact"]["valid_marker"] = (
            "artifact-valid-v4" if summary["artifact"]["valid"] else None
        )
    summary["artifact"]["image_samples"] = image_sample_collection_contract(
        [record.record_dict() for record in records]
    )
    return summary


def _warning_summary(records: Sequence[RequestRecord], metrics: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    warned_request_ids: set[str] = set()
    for record in records:
        if record.warnings:
            warned_request_ids.add(record.request_id)
        for warning in record.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    interleave = metrics.get("modality_interleave")
    timing = interleave.get("transition_timing") if isinstance(interleave, dict) else None
    server_attribution = timing.get("server_attribution") if isinstance(timing, dict) else None
    request_statuses = (
        server_attribution.get("request_statuses") if isinstance(server_attribution, dict) else None
    )
    if isinstance(request_statuses, dict):
        for request_id, status in request_statuses.items():
            if not isinstance(request_id, str) or status not in {"partial", "unavailable"}:
                continue
            warning = f"server_public_commit_{status}"
            warned_request_ids.add(request_id)
            counts[warning] = counts.get(warning, 0) + 1
    return {
        "request_count": len(warned_request_ids),
        "total_count": sum(counts.values()),
        "counts": counts,
    }


def _completed_images(family: str, metrics: dict[str, Any]) -> int:
    if family == "image":
        return int(metrics.get("completed_images", 0))
    images = metrics.get("images")
    return int(images.get("total_images", 0)) if isinstance(images, dict) else 0


def _generation_conformance(
    spec: BenchmarkSpec,
    records: list[RequestRecord],
) -> dict[str, Any]:
    if spec.task == TaskName.INTERLEAVE:
        return _interleave_generation_conformance(spec, records)
    if spec.task in IMAGE_TASKS:
        return _image_generation_conformance(spec, records)
    return _text_generation_conformance(spec, records)


def _interleave_latency_conformance(
    spec: BenchmarkSpec,
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    if spec.task != TaskName.INTERLEAVE:
        return None
    interleave = metrics.get("modality_interleave")
    timing = interleave.get("transition_timing") if isinstance(interleave, dict) else None
    if not isinstance(timing, dict):
        return {
            "valid": False,
            "checks": {"latency_measurements_present": False},
        }
    latency_definition_digest = timing.get("latency_definition_digest")
    expected_transitions = timing.get("expected_transition_count")
    measured_transitions = timing.get("measured_transition_count")
    ttft = metrics.get("ttft_ms")
    tpot = metrics.get("tpot_ms")
    images = metrics.get("images")
    image_latency = images.get("image_latency_ms") if isinstance(images, dict) else None
    total_images = images.get("total_images") if isinstance(images, dict) else None
    transition_latency = timing.get("transition_latency_ms")
    text_to_image = timing.get("text_to_image_transition_latency_ms")
    image_to_text = timing.get("image_to_text_transition_latency_ms")
    text_to_image_count = text_to_image.get("count") if isinstance(text_to_image, dict) else None
    image_to_text_count = image_to_text.get("count") if isinstance(image_to_text, dict) else None
    checks = {
        "latency_measurements_present": True,
        "latency_definition_digest": (
            isinstance(latency_definition_digest, str)
            and len(latency_definition_digest) == 64
            and all(character in "0123456789abcdef" for character in latency_definition_digest)
        ),
        "ttft_samples": _complete_latency_distribution(ttft, expected_count=spec.num_prompts),
        "tpot_samples": _complete_latency_distribution(tpot, expected_count=spec.num_prompts),
        "image_latency_samples": (
            isinstance(total_images, int)
            and not isinstance(total_images, bool)
            and total_images > 0
            and _complete_latency_distribution(image_latency, expected_count=total_images)
        ),
        "declared_request_count": timing.get("request_count") == spec.num_prompts,
        "complete_request_count": timing.get("complete_request_count") == spec.num_prompts,
        "timestamp_coverage": timing.get("timestamp_coverage") == 1.0,
        "unambiguous_events": timing.get("ambiguous_event_count") == 0,
        "monotonic_events": timing.get("non_monotonic_event_count") == 0,
        "positive_transition_work": (
            isinstance(expected_transitions, int)
            and not isinstance(expected_transitions, bool)
            and expected_transitions > 0
        ),
        "complete_transition_samples": measured_transitions == expected_transitions,
        "transition_sample_coverage": timing.get("transition_sample_coverage") == 1.0,
        "transition_latency_samples": (
            isinstance(expected_transitions, int)
            and not isinstance(expected_transitions, bool)
            and _complete_latency_distribution(
                transition_latency,
                expected_count=expected_transitions,
            )
        ),
        "directional_transition_samples": (
            isinstance(text_to_image_count, int)
            and not isinstance(text_to_image_count, bool)
            and isinstance(image_to_text_count, int)
            and not isinstance(image_to_text_count, bool)
            and isinstance(expected_transitions, int)
            and not isinstance(expected_transitions, bool)
            and text_to_image_count + image_to_text_count == expected_transitions
            and _complete_latency_distribution(text_to_image, expected_count=text_to_image_count)
            and _complete_latency_distribution(image_to_text, expected_count=image_to_text_count)
        ),
        "transition_summary": timing.get("valid") is True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "latency_definition_digest": latency_definition_digest,
        "request_signatures": timing.get("request_signatures"),
        "expected_transition_count": expected_transitions,
        "measured_transition_count": measured_transitions,
    }


def _complete_latency_distribution(value: Any, *, expected_count: int) -> bool:
    if not isinstance(value, dict) or value.get("count") != expected_count:
        return False
    if expected_count == 0:
        return True
    statistics = ("mean", "std", "min", "p50", "p90", "p95", "p99", "max")
    return all(
        isinstance(value.get(statistic), (int, float))
        and not isinstance(value.get(statistic), bool)
        and math.isfinite(float(value[statistic]))
        and float(value[statistic]) >= 0.0
        for statistic in statistics
    )


def _image_generation_conformance(
    spec: BenchmarkSpec, records: list[RequestRecord]
) -> dict[str, Any]:
    successful = [record for record in records if record.success]
    mismatches = [
        record.request_id for record in records if not _image_record_conforms(record, spec)
    ]
    return {
        "schema_version": 1,
        "policy": (
            "optional_decoded_images_within_declared_cap"
            if spec.task == TaskName.INTERLEAVE
            else "decoded_image_exact_declared_work"
        ),
        "successful_requests": len(successful),
        "checked_requests": len(records),
        "mismatch_count": len(mismatches),
        "mismatched_request_ids": mismatches,
        "valid": bool(records) and not mismatches,
    }


def _interleave_generation_conformance(
    spec: BenchmarkSpec, records: list[RequestRecord]
) -> dict[str, Any]:
    image = _image_generation_conformance(spec, records)
    modality_mismatches = [
        record.request_id for record in records if not _interleave_record_conforms(record)
    ]
    mismatches = list(dict.fromkeys(image["mismatched_request_ids"] + modality_mismatches))
    return {
        "schema_version": 1,
        "policy": "visible_text_with_optional_decoded_image_transitions",
        "successful_requests": sum(record.success for record in records),
        "checked_requests": len(records),
        "mismatch_count": len(mismatches),
        "mismatched_request_ids": mismatches,
        "components": {
            "image": image,
            "modality_transition": {
                "mismatch_count": len(modality_mismatches),
                "mismatched_request_ids": modality_mismatches,
            },
        },
        "valid": bool(records) and not mismatches,
    }


def _text_generation_conformance(
    spec: BenchmarkSpec, records: list[RequestRecord]
) -> dict[str, Any]:
    successful = [record for record in records if record.success]
    exact_length_required = spec.task in {TaskName.TEXT, TaskName.I2T} and spec.ignore_eos
    checked = [record for record in successful if record.requested_output_len > 0]
    mismatches = [
        record.request_id
        for record in checked
        if record.output_len != record.requested_output_len
        or (exact_length_required and record.finish_reason != "length")
        or record.output_len_source != "server_usage"
        or record.prompt_len_source != "server_usage"
    ]
    valid = bool(successful)
    if exact_length_required:
        valid = valid and len(checked) == len(successful) and not mismatches
    return {
        "schema_version": 1,
        "policy": "server_usage_exact_length" if exact_length_required else "successful_response",
        "successful_requests": len(successful),
        "checked_requests": len(checked),
        "mismatch_count": len(mismatches),
        "mismatched_request_ids": mismatches,
        "valid": valid,
    }


def _interleave_record_conforms(record: RequestRecord) -> bool:
    if not record.success or not record.generated_text or "text" not in record.output_modalities:
        return False
    if record.images == 0:
        return record.image_output_mode == "optional"
    return bool(
        record.image_output_mode == "optional"
        and "image" in record.output_modalities
        and len(record.output_modalities) >= 2
    )


def _image_record_conforms(record: RequestRecord, spec: BenchmarkSpec) -> bool:
    expected_mode: ImageOutputMode = "optional" if spec.task == TaskName.INTERLEAVE else "required"
    contract = ImageOutputContract(
        mode=expected_mode,
        count=(
            record.requested_image_count
            if record.requested_image_count is not None
            else spec.max_images
        ),
        count_is_cap=(
            record.requested_image_count_is_cap
            if record.requested_image_count is not None
            else spec.task.value in {"default", "interleave"}
        ),
        width=(
            record.requested_image_width if record.requested_image_width is not None else spec.width
        ),
        height=(
            record.requested_image_height
            if record.requested_image_height is not None
            else spec.height
        ),
    )
    return bool(
        record.success
        and record.image_output_mode == expected_mode
        and record.images == len(record.decoded_images)
        and image_output_mismatch(record.decoded_images, contract) is None
    )


def _observed_endpoint(records: list[RequestRecord]) -> str | None:
    endpoints = {record.endpoint for record in records if record.endpoint}
    return next(iter(endpoints)) if len(endpoints) == 1 else None


def render_markdown(summary: dict[str, Any]) -> str:
    load = summary["load"]
    lines = [
        f"# {summary['task']} benchmark ({summary['dataset']})",
        "",
        f"- model: `{summary['model']}`  endpoint: `{summary['endpoint']}`",
        f"- load: {load['mode']}  rate={load['request_rate']}  "
        f"max_concurrency={load['max_concurrency']}  num_prompts={load['num_prompts']}",
        f"- requests: {summary['ok_count']}/{summary['request_count']} ok  "
        f"elapsed={summary['elapsed_s']:.2f}s",
        "",
    ]
    metrics = summary["metrics"]
    if summary["metric_family"] == "stream":
        lines += _stream_markdown(metrics)
    else:
        lines += _image_markdown(metrics)
    return "\n".join(lines) + "\n"


def _row(label: str, *values: Any) -> str:
    cells = " | ".join(_fmt(value) for value in values)
    return f"| {label} | {cells} |"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def _stream_markdown(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row(
            "TTFT (ms)",
            metrics["p50_ttft_ms"],
            metrics["p90_ttft_ms"],
            metrics["p95_ttft_ms"],
            metrics["p99_ttft_ms"],
            metrics["mean_ttft_ms"],
        ),
        _row(
            "TPOT (ms)",
            metrics["p50_tpot_ms"],
            metrics["p90_tpot_ms"],
            metrics["p95_tpot_ms"],
            metrics["p99_tpot_ms"],
            metrics["mean_tpot_ms"],
        ),
        _row(
            "ITL (ms)",
            metrics["p50_itl_ms"],
            metrics["p90_itl_ms"],
            metrics["p95_itl_ms"],
            metrics["p99_itl_ms"],
            metrics["mean_itl_ms"],
        ),
        _row(
            "E2E (ms)",
            metrics["p50_e2e_latency_ms"],
            metrics["p90_e2e_latency_ms"],
            metrics["p95_e2e_latency_ms"],
            metrics["p99_e2e_latency_ms"],
            metrics["mean_e2e_latency_ms"],
        ),
        "",
        f"- output throughput: {_fmt(metrics['output_throughput'])} tok/s",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- total throughput: {_fmt(metrics['total_throughput'])} tok/s",
        f"- concurrency: {_fmt(metrics['concurrency'])}  "
        f"peak tok/s: {_fmt(metrics['max_output_tokens_per_s'])}  "
        f"peak concurrent: {metrics['max_concurrent_requests']}",
    ]
    if metrics.get("token_timing_available") is False:
        lines.extend(
            [
                "",
                "- TTFT, TPOT, and ITL: unavailable because the response did not stream token events",
            ]
        )
    if "images" in metrics:
        img = metrics["images"]["image_latency_ms"]
        lines += [
            "",
            f"- in-response images: {metrics['images']['total_images']} total, "
            f"{_fmt(metrics['images']['images_per_second'])} img/s, "
            f"image E2E p50/p99 = {_fmt(img['p50'])}/{_fmt(img['p99'])} ms",
        ]
    interleave = metrics.get("modality_interleave")
    timing = interleave.get("transition_timing") if isinstance(interleave, dict) else None
    if isinstance(timing, dict):
        overall = timing["transition_latency_ms"]
        text_to_image = timing["text_to_image_transition_latency_ms"]
        image_to_text = timing["image_to_text_transition_latency_ms"]
        server_attribution = timing.get("server_attribution")
        server_attribution_status = (
            server_attribution.get("status")
            if isinstance(server_attribution, dict)
            else "unavailable"
        )
        lines += [
            "",
            f"- transition timing: {'valid' if timing.get('valid') else 'invalid'}, "
            f"coverage={_fmt(timing.get('transition_sample_coverage'))}",
            f"- server transition attribution: {server_attribution_status}",
            f"- transition latency mean/p95 = {_fmt(overall['mean'])}/{_fmt(overall['p95'])} ms",
            f"- text-to-image mean/p95 = "
            f"{_fmt(text_to_image['mean'])}/{_fmt(text_to_image['p95'])} ms",
            f"- image-to-text mean/p95 = "
            f"{_fmt(image_to_text['mean'])}/{_fmt(image_to_text['p95'])} ms",
        ]
    return lines


def _image_markdown(metrics: dict[str, Any]) -> list[str]:
    lat = metrics["image_latency_ms"]
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row("image latency (ms)", lat["p50"], lat["p90"], lat["p95"], lat["p99"], lat["mean"]),
        "",
        f"- images/s: {_fmt(metrics['images_per_second'])}  "
        f"images/min: {_fmt(metrics['images_per_minute'])}",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- completed: {metrics['completed_requests']} requests, {metrics['completed_images']} images",
    ]
    if "time_to_first_image_ms" in metrics:
        ttfi = metrics["time_to_first_image_ms"]
        lines.append(f"- time-to-first-image p50/p99 = {_fmt(ttfi['p50'])}/{_fmt(ttfi['p99'])} ms")
    if "steps_per_second" in metrics:
        sps = metrics["steps_per_second"]
        lines.append(f"- steps/s p50 = {_fmt(sps['p50'])}")
    return lines
