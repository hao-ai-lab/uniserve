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
    ImageOutputRequirements,
    image_output_mismatch,
    inspect_image_bytes,
)
from .metrics import RequestRecord, summarize_image, summarize_mixed, summarize_stream
from .spec import BenchmarkSpec, TaskName


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
    {"dataset_path", "model", "name", "runtime_profile_id", "plan_evidence_policy"}
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
        artifact.get("schema_version") == 2
        and artifact.get("valid") is True
        and artifact.get("valid_marker") in {"artifact-valid-v2", "canonical-valid-v2"}
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
    return summary["artifact"].get("valid_marker") == "canonical-valid-v2"


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
    was_canonical = artifact.get("valid_marker") == "canonical-valid-v2"
    artifact[name] = contract
    checks[name] = valid
    artifact["valid"] = all(value is True for value in checks.values())
    canonical = was_canonical or name in {"matrix_contract", "profile_contract"}
    artifact["valid_marker"] = (
        ("canonical-valid-v2" if canonical else "artifact-valid-v2") if artifact["valid"] else None
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


def plan_summary(spec: BenchmarkSpec) -> dict[str, Any]:
    """Sanitized semantic plan declared for one formal measurement point."""
    return {
        "runtime_profile_id": spec.runtime_profile_id,
        "protocol_adapter": spec.wire,
        "measurement_interface": spec.measurement_interface,
        "context": {
            "dataset": spec.dataset,
            "preprocessing": spec.preprocessing,
            "prompt_source": spec.dataset_path or spec.dataset,
            "workload_mix": spec.workload_mix,
            "warmup_mix": spec.warmup_mix,
            "t2i_dataset_revision": spec.t2i_dataset_revision,
            "i2t_dataset_revision": spec.i2t_dataset_revision,
        },
        "generation": {
            "output_constraint": spec.output_constraint,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "min_p": spec.min_p,
            "repetition_penalty": spec.repetition_penalty,
            "frequency_penalty": spec.frequency_penalty,
            "presence_penalty": spec.presence_penalty,
            "seed": spec.sampling_seed,
            "chat_template_kwargs": spec.chat_template_kwargs,
            "ignore_eos": spec.ignore_eos,
            "structured_output": spec.structured_output_policy,
        },
        "image": {
            "width": spec.width,
            "height": spec.height,
            "steps": spec.steps,
            "denoise_updates": spec.denoise_updates,
            "max_images": spec.max_images,
            "seed": spec.seed,
            "guidance_scale": spec.guidance_scale,
            "image_guidance_scale": spec.image_guidance_scale,
            "cfg_norm": spec.cfg_norm,
            "cfg_interval": list(spec.cfg_interval) if spec.cfg_interval is not None else None,
            "timestep_shift": spec.timestep_shift,
            "think": spec.image_think,
            "t_eps": spec.image_t_eps,
        },
        "cache": {
            "read": spec.cache_read_policy,
            "write": spec.cache_write_policy,
        },
        "adapter": spec.adapter_selection,
    }


def artifact_contract(
    spec: BenchmarkSpec,
    *,
    request_count: int,
    ok_count: int,
    failed_count: int,
    total_images: int,
    plan_evidence: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    generation_conformance: dict[str, Any],
) -> dict[str, Any]:
    expected_plan_source = spec.plan_evidence_policy
    if plan_evidence is None:
        plan_evidence = {
            "source": "declared_contract",
            "plan": plan_summary(spec),
        }
    plan = plan_evidence.get("plan")
    plan_evidence_valid = plan_evidence.get("source") == expected_plan_source
    if plan_evidence_valid and expected_plan_source == "declared_contract":
        plan_evidence_valid = isinstance(plan, dict)
    elif plan_evidence_valid and expected_plan_source == "runtime_inspection":
        if spec.task == TaskName.MIXED:
            plan_evidence_valid = _mixed_plan_evidence_matches_contract(
                plan_evidence,
                spec,
                source="runtime_inspection",
            )
        else:
            request = plan_evidence.get("request")
            plan_evidence_valid = (
                isinstance(plan, dict)
                and isinstance(request, dict)
                and _reference_request_matches_contract(request, spec)
                and _runtime_plan_matches_declared_contract(plan, spec, request=request)
            )
    elif plan_evidence_valid and expected_plan_source == "reference_protocol":
        if spec.task == TaskName.MIXED:
            plan_evidence_valid = _mixed_plan_evidence_matches_contract(
                plan_evidence,
                spec,
                source="reference_protocol",
            )
        else:
            request = plan_evidence.get("request")
            plan_evidence_valid = (
                isinstance(request, dict)
                and plan == plan_summary(spec)
                and _reference_request_matches_contract(request, spec)
            )
    checks = {
        "declared_request_count": request_count == spec.num_prompts,
        "minimum_successful_requests": ok_count >= spec.acceptance_min_success,
        "maximum_failed_requests": failed_count <= spec.acceptance_max_failed,
        "single_measured_run": spec.measured_runs == 1,
        "minimum_images": total_images
        >= math.ceil(ok_count * spec.acceptance_min_images_per_success),
        "plan_evidence": plan_evidence_valid,
        "contract_fingerprint": benchmark_contract_is_valid(contract, spec, request_count),
        "generation_conformance": generation_conformance.get("valid") is True,
    }
    valid = all(checks.values())
    return {
        "schema_version": 2,
        "valid": valid,
        "valid_marker": "artifact-valid-v2" if valid else None,
        "checks": checks,
        "acceptance": {
            "minimum_successful_requests": spec.acceptance_min_success,
            "maximum_failed_requests": spec.acceptance_max_failed,
            "minimum_images_per_success": spec.acceptance_min_images_per_success,
        },
        "measured_runs": spec.measured_runs,
        "server_topology": spec.server_topology,
        "declared_contract": plan_summary(spec),
        "plan_summary": plan_evidence.get("plan", plan_evidence.get("request")),
        "plan_evidence": plan_evidence,
        "contract": contract,
        "generation_conformance": generation_conformance,
    }


def _mixed_plan_evidence_matches_contract(
    evidence: dict[str, Any],
    spec: BenchmarkSpec,
    *,
    source: str,
) -> bool:
    from .tasks.mixed import mixed_subtask_specs

    workloads = evidence.get("workloads")
    aggregate = evidence.get("plan")
    subtask_specs = mixed_subtask_specs(spec)
    expected_tasks = set(subtask_specs)
    if not isinstance(workloads, dict) or set(workloads) != expected_tasks:
        return False
    expected_aggregate: dict[str, Any] = {"workloads": {}}
    for task_name, subtask_spec in subtask_specs.items():
        entry = workloads.get(task_name)
        if not isinstance(entry, dict):
            return False
        plan = entry.get("plan")
        request = entry.get("request")
        if not isinstance(plan, dict) or not isinstance(request, dict):
            return False
        if not _reference_request_matches_contract(request, subtask_spec):
            return False
        if source == "runtime_inspection":
            if not _runtime_plan_matches_declared_contract(
                plan,
                subtask_spec,
                request=request,
            ):
                return False
        elif source == "reference_protocol":
            if plan != plan_summary(subtask_spec):
                return False
        else:
            return False
        expected_aggregate["workloads"][task_name] = plan
    return aggregate == expected_aggregate


def _runtime_plan_matches_declared_contract(
    plan: dict[str, Any], spec: BenchmarkSpec, *, request: dict[str, Any] | None = None
) -> bool:
    expected_profile = spec.runtime_profile_id
    if expected_profile == "unspecified":
        return False
    profile_id = plan.get("profile_id")
    dialect_id = plan.get("dialect_id")
    identity_matches = (
        isinstance(profile_id, str)
        and isinstance(dialect_id, str)
        and (
            profile_id == expected_profile
            or profile_id.startswith(f"{expected_profile}:")
            or dialect_id == expected_profile
        )
    )
    generation = plan.get("generation")
    cache = plan.get("cache")
    if not identity_matches or not isinstance(generation, dict) or not isinstance(cache, dict):
        return False
    if generation.get("constraint") != spec.output_constraint:
        return False
    if spec.max_tokens is not None and generation.get("max_tokens") != spec.max_tokens:
        return False
    if request is not None:
        request_generation = request.get("generation")
        if not isinstance(request_generation, dict):
            return False
        requested_max = request_generation.get("max_tokens")
        if requested_max is not None and generation.get("max_tokens") != requested_max:
            return False
    if not _same_float(generation.get("temperature"), spec.temperature):
        return False
    if not _same_float(generation.get("top_p"), spec.top_p):
        return False
    optional_generation_fields: dict[str, float | int | None] = {
        "top_k": spec.top_k,
        "min_p": spec.min_p,
        "repetition_penalty": spec.repetition_penalty,
        "frequency_penalty": spec.frequency_penalty,
        "presence_penalty": spec.presence_penalty,
        "seed": spec.sampling_seed,
    }
    for key, optional_expected in optional_generation_fields.items():
        if optional_expected is None:
            continue
        actual = generation.get(key)
        if isinstance(optional_expected, float):
            if not _same_float(actual, optional_expected):
                return False
        elif actual != optional_expected:
            return False
    if generation.get("ignore_eos") is not spec.ignore_eos:
        return False
    if cache.get("read_enabled") is not (spec.cache_read_policy == "enabled"):
        return False
    if cache.get("write_enabled") is not (spec.cache_write_policy == "enabled"):
        return False
    if _adapter_name(plan.get("adapter")) != spec.adapter_selection:
        return False
    if spec.output_constraint not in {"default", "gen_only"}:
        return True
    image = generation.get("image")
    if not isinstance(image, dict):
        return False
    exact_image_fields: dict[str, int | str | None] = {
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "max_images": spec.max_images,
        "seed": spec.seed,
        "cfg_renorm_type": spec.cfg_norm,
    }
    for key, expected_exact in exact_image_fields.items():
        if expected_exact is not None and image.get(key) != expected_exact:
            return False
    float_image_fields: dict[str, float | None] = {
        "cfg_text_scale": spec.guidance_scale,
        "cfg_img_scale": spec.image_guidance_scale,
        "timestep_shift": spec.timestep_shift,
    }
    for key, expected_float in float_image_fields.items():
        if expected_float is not None and not _same_float(image.get(key), expected_float):
            return False
    if spec.cfg_interval is not None:
        actual_interval = image.get("cfg_interval")
        if not isinstance(actual_interval, (list, tuple)) or len(actual_interval) != 2:
            return False
        if not all(
            _same_float(actual, expected)
            for actual, expected in zip(actual_interval, spec.cfg_interval, strict=True)
        ):
            return False
    return True


def _reference_request_matches_contract(request: dict[str, Any], spec: BenchmarkSpec) -> bool:
    if request.get("endpoint") != spec.endpoint or request.get("kind") != spec.wire:
        return False
    if request.get("model") != spec.model or request.get("adapter") != spec.adapter_selection:
        return False
    modalities = request.get("modalities")
    if spec.output_constraint == "gen_only":
        if request.get("kind") != "images_generations" and modalities != ["image"]:
            return False
    elif spec.output_constraint == "default":
        if modalities != ["text", "image"]:
            return False
    elif spec.output_constraint == "und_only" and modalities is not None and modalities != ["text"]:
        return False
    generation = request.get("generation")
    if not isinstance(generation, dict):
        return False
    if spec.max_tokens is not None and generation.get("max_tokens") != spec.max_tokens:
        return False
    for key, expected in (("temperature", spec.temperature), ("top_p", spec.top_p)):
        actual = generation.get(key)
        if actual is not None and not _same_float(actual, expected):
            return False
    optional_generation_fields = {
        "top_k": spec.top_k,
        "min_p": spec.min_p,
        "repetition_penalty": spec.repetition_penalty,
        "frequency_penalty": spec.frequency_penalty,
        "presence_penalty": spec.presence_penalty,
        "seed": spec.sampling_seed,
    }
    for key, expected_value in optional_generation_fields.items():
        if expected_value is None:
            continue
        actual = generation.get(key)
        if isinstance(expected_value, float):
            if not _same_float(actual, expected_value):
                return False
        elif actual != expected_value:
            return False
    if generation.get("chat_template_kwargs") != spec.chat_template_kwargs:
        return False
    actual_ignore_eos = generation.get("ignore_eos")
    if actual_ignore_eos is not None and actual_ignore_eos is not spec.ignore_eos:
        return False
    if spec.structured_output_policy == "none" and (
        generation.get("structured_outputs") is not None
        or generation.get("response_format") is not None
    ):
        return False
    if spec.task.value in {"i2i", "i2t"} and request.get("input_image_count", 0) < 1:
        return False
    if spec.output_constraint not in {"default", "gen_only"}:
        return True
    image = request.get("image")
    if not isinstance(image, dict) or image.get("steps_consistent") is not True:
        return False
    exact_fields = {
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "max_images": spec.max_images,
        "seed": spec.seed,
        "cfg_norm": spec.cfg_norm,
    }
    for key, expected_exact in exact_fields.items():
        if expected_exact is not None and image.get(key) != expected_exact:
            return False
    float_fields = {
        "guidance_scale": spec.guidance_scale,
        "image_guidance_scale": spec.image_guidance_scale,
        "timestep_shift": spec.timestep_shift,
        "t_eps": spec.image_t_eps,
    }
    for key, expected_float in float_fields.items():
        if expected_float is not None and not _same_float(image.get(key), expected_float):
            return False
    if spec.image_think is not None and image.get("think") is not spec.image_think:
        return False
    if spec.cfg_interval is not None:
        actual_interval = image.get("cfg_interval")
        if not isinstance(actual_interval, (list, tuple)) or len(actual_interval) != 2:
            return False
        if not all(
            _same_float(actual, expected)
            for actual, expected in zip(actual_interval, spec.cfg_interval, strict=True)
        ):
            return False
    return True


def _same_float(actual: Any, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    )


def _adapter_name(adapter: Any) -> str | None:
    if isinstance(adapter, str):
        return adapter.lower()
    if isinstance(adapter, dict) and len(adapter) == 1:
        return next(iter(adapter)).lower()
    return None


def build_summary(
    spec: BenchmarkSpec,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
    server_info: dict[str, Any] | None = None,
    plan_evidence: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [r for r in records if r.success]
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1

    if spec.task == TaskName.MIXED:
        family = "mixed"
        metrics = summarize_mixed(records, dur_s)
    elif spec.is_stream_task:
        family = "stream"
        metrics = summarize_stream(records, dur_s, tokenizer=tokenizer)
    else:
        family = "image"
        metrics = summarize_image(records, dur_s)
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
        "metric_family": family,
        "metrics": metrics,
    }
    summary["artifact"] = artifact_contract(
        spec,
        request_count=summary["request_count"],
        ok_count=summary["ok_count"],
        failed_count=summary["failed_count"],
        total_images=_completed_images(family, metrics),
        plan_evidence=plan_evidence,
        contract=contract,
        generation_conformance=_generation_conformance(spec, records),
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
            "artifact-valid-v2" if summary["artifact"]["valid"] else None
        )
    summary["artifact"]["image_samples"] = image_sample_collection_contract(
        [record.record_dict() for record in records]
    )
    return summary


def _completed_images(family: str, metrics: dict[str, Any]) -> int:
    if family == "image":
        return int(metrics.get("completed_images", 0))
    if family == "mixed":
        image = metrics.get("t2i")
        return int(image.get("completed_images", 0)) if isinstance(image, dict) else 0
    images = metrics.get("images")
    return int(images.get("total_images", 0)) if isinstance(images, dict) else 0


def _generation_conformance(
    spec: BenchmarkSpec,
    records: list[RequestRecord],
) -> dict[str, Any]:
    if spec.task == TaskName.MIXED:
        image_records = [record for record in records if record.task == TaskName.T2I.value]
        text_records = [record for record in records if record.task == TaskName.I2T.value]
        image = _image_generation_conformance(spec, image_records)
        text = _text_generation_conformance(spec, text_records)
        counts_match = (
            len(image_records) == spec.workload_mix[TaskName.T2I.value]
            and len(text_records) == spec.workload_mix[TaskName.I2T.value]
        )
        return {
            "schema_version": 1,
            "policy": "per_task_declared_work",
            "successful_requests": sum(record.success for record in records),
            "checked_requests": len(records),
            "mismatch_count": image["mismatch_count"] + text["mismatch_count"],
            "mismatched_request_ids": image["mismatched_request_ids"]
            + text["mismatched_request_ids"],
            "components": {"t2i": image, "i2t": text},
            "task_counts_match": counts_match,
            "valid": image["valid"] is True and text["valid"] is True and counts_match,
        }
    if spec.task == TaskName.INTERLEAVE:
        return _interleave_generation_conformance(spec, records)
    image_output_required = spec.task.value in {"t2i", "i2i", "default"}
    if image_output_required:
        return _image_generation_conformance(spec, records)
    return _text_generation_conformance(spec, records)


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
            "decoded_image_within_declared_cap"
            if spec.task.value in {"default", "interleave"}
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
        record.request_id
        for record in records
        if not (
            record.success
            and record.generated_text
            and "text" in record.output_modalities
            and "image" in record.output_modalities
            and len(record.output_modalities) >= 2
        )
    ]
    mismatches = list(dict.fromkeys(image["mismatched_request_ids"] + modality_mismatches))
    return {
        "schema_version": 1,
        "policy": "decoded_image_and_visible_modality_transition",
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
    exact_length_required = spec.task.value in {"text", "i2t"} and spec.ignore_eos
    if spec.task == TaskName.MIXED:
        exact_length_required = spec.ignore_eos
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


def _image_record_conforms(record: RequestRecord, spec: BenchmarkSpec) -> bool:
    requirements = ImageOutputRequirements(
        expected=True,
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
        and record.images == len(record.decoded_images)
        and image_output_mismatch(record.decoded_images, requirements) is None
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
    elif summary["metric_family"] == "mixed":
        lines += _mixed_markdown(metrics)
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
            f"- default-task images: {metrics['images']['total_images']} total, "
            f"{_fmt(metrics['images']['images_per_second'])} img/s, "
            f"image E2E p50/p99 = {_fmt(img['p50'])}/{_fmt(img['p99'])} ms",
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


def _mixed_markdown(metrics: dict[str, Any]) -> list[str]:
    image = metrics["t2i"]
    text = metrics["i2t"]
    overlap = metrics["client_cross_task_overlap"]
    return [
        f"- mixed request throughput: {_fmt(metrics['mixed_request_throughput'])} req/s",
        f"- T2I: {image['completed_images']} images, {_fmt(image['images_per_minute'])} images/min",
        f"- I2T: {text['total_output_tokens']} output tokens, {_fmt(text['output_throughput'])} tok/s",
        f"- client cross-task overlap: {_fmt(overlap['duration_s'])}s "
        f"({_fmt(100.0 * overlap['timed_region_fraction'])}% of timed region)",
    ]
