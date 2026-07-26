"""Offline conformance and diagnostic evidence for generated-image artifacts.

The evaluator consumes two committed benchmark point directories. It revalidates
their canonical artifact contracts and exact content-addressed image samples,
pairs outputs by request identity and image index, and emits only redacted
quality evidence. It is a same-model regression gate, not an absolute measure
of prompt/image quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from uniserve_eval.profiles import (
    DEFAULT_CONFIG,
    benchmark_matrix_profile_binding_matches,
    load_config,
)

from .image_outputs import inspect_image_bytes
from .report import (
    benchmark_parity_contract,
    canonical_artifact_bundle_matches,
    canonical_digest,
)

LPIPS_MAXIMUM = 0.15
PSNR_DB_MINIMUM = 20.0
UINT8_MAE_MAXIMUM = 12.0
COSINE_SIMILARITY_MINIMUM = 0.98
RELATIVE_L2_MAXIMUM = 0.20
LPIPS_SIZE = (256, 256)
_IMAGE_TASKS = frozenset({"i2i", "t2i"})
_POINT_SUPPORT_FILES = frozenset(
    {"command.txt", "preflight.txt", "postflight.txt", "run.json", "run.log"}
)
_SERVER_SUPPORT_FILES = frozenset(
    {
        "server_command.txt",
        "server.log",
        "server.exit",
        "pre_server_snapshot.txt",
        "post_server_snapshot.txt",
    }
)


class QualityRuntimeError(RuntimeError):
    """The fixed image-quality implementation could not be provisioned."""


@dataclass(frozen=True)
class _ImageReference:
    request_id: str
    image_index: int
    path: Path
    metadata: dict[str, Any]
    declared_width: int
    declared_height: int


@dataclass
class _ArtifactInspection:
    canonical_valid: bool
    image_workload: bool
    semantic_contract: dict[str, Any] | None
    comparison_role: str | None
    server_profile: str | None
    server_profile_binding_matches: bool
    server_execution_identity: dict[str, Any] | None
    required_source_revision_valid: bool
    requests_valid: bool
    successful_requests: bool
    image_counts_valid: bool
    declared_dimensions_valid: bool
    request_ids: set[str]
    images: dict[tuple[str, int], _ImageReference]
    binding: dict[str, Any]


@dataclass(frozen=True)
class _QualityRuntime:
    model: Any
    transform: Callable[[Image.Image], Any]
    torch: Any
    numpy: Any
    structural_similarity: Callable[..., float]
    provenance: dict[str, Any]

    def compare(self, reference: Any, candidate: Any) -> dict[str, float]:
        reference_pil = Image.fromarray(reference, mode="RGB")
        candidate_pil = Image.fromarray(candidate, mode="RGB")
        reference_tensor = self.transform(reference_pil).unsqueeze(0)
        candidate_tensor = self.transform(candidate_pil).unsqueeze(0)
        previous_determinism = self.torch.are_deterministic_algorithms_enabled()
        previous_warn_only = self.torch.is_deterministic_algorithms_warn_only_enabled()
        self.torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            with self.torch.inference_mode():
                lpips_value = float(self.model(reference_tensor, candidate_tensor).item())
        finally:
            self.torch.use_deterministic_algorithms(
                previous_determinism,
                warn_only=previous_warn_only,
            )

        reference_float = reference.astype(self.numpy.float64, copy=False)
        candidate_float = candidate.astype(self.numpy.float64, copy=False)
        difference = reference_float - candidate_float
        mse = float(self.numpy.mean(self.numpy.square(difference)))
        mae = float(self.numpy.mean(self.numpy.abs(difference)))
        difference_l2 = float(self.numpy.linalg.norm(difference.reshape(-1)))
        reference_l2 = float(self.numpy.linalg.norm(reference_float.reshape(-1)))
        candidate_l2 = float(self.numpy.linalg.norm(candidate_float.reshape(-1)))
        if reference_l2 == 0.0:
            relative_l2 = 0.0 if difference_l2 == 0.0 else math.inf
        else:
            relative_l2 = difference_l2 / reference_l2
        if reference_l2 == 0.0 and candidate_l2 == 0.0:
            cosine_similarity = 1.0
        elif reference_l2 == 0.0 or candidate_l2 == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = float(
                self.numpy.dot(reference_float.reshape(-1), candidate_float.reshape(-1))
                / (reference_l2 * candidate_l2)
            )
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        psnr_db = math.inf if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
        ssim = float(
            self.structural_similarity(
                reference,
                candidate,
                channel_axis=2,
                data_range=255,
                gaussian_weights=False,
                use_sample_covariance=True,
                win_size=7,
            )
        )
        return {
            "lpips": lpips_value,
            "ssim": ssim,
            "psnr_db": psnr_db,
            "uint8_mae": mae,
            "cosine_similarity": cosine_similarity,
            "relative_l2": relative_l2,
        }


def evaluate_image_quality(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    _runtime_factory: Callable[[], _QualityRuntime] | None = None,
    _profile_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return redacted paired-image quality evidence for two canonical points."""
    failures: list[dict[str, Any]] = []
    profile_config = load_config(DEFAULT_CONFIG) if _profile_config is None else _profile_config
    reference = _inspect_artifact(
        Path(reference_directory),
        side="reference",
        failures=failures,
        profile_config=profile_config,
    )
    candidate = _inspect_artifact(
        Path(candidate_directory),
        side="candidate",
        failures=failures,
        profile_config=profile_config,
    )

    distinct_artifacts = Path(reference_directory).resolve() != Path(candidate_directory).resolve()
    if not distinct_artifacts:
        _add_failure(failures, "reference_and_candidate_are_same_artifact")
    canonical_artifacts = reference.canonical_valid and candidate.canonical_valid
    image_workloads = reference.image_workload and candidate.image_workload
    semantic_parity = bool(
        reference.semantic_contract is not None
        and candidate.semantic_contract is not None
        and reference.semantic_contract == candidate.semantic_contract
    )
    if not semantic_parity:
        _add_failure(failures, "semantic_contract_mismatch")
    comparison_roles = (
        reference.comparison_role == "reference" and candidate.comparison_role == "candidate"
    )
    if not comparison_roles:
        _add_failure(failures, "comparison_role_mismatch")
    active_server_profile_bindings = bool(
        reference.server_profile_binding_matches and candidate.server_profile_binding_matches
    )
    if not active_server_profile_bindings:
        _add_failure(failures, "server_profile_binding_mismatch")
    distinct_server_implementation_identities = bool(
        reference.server_execution_identity is not None
        and candidate.server_execution_identity is not None
        and reference.server_execution_identity != candidate.server_execution_identity
    )
    if not distinct_server_implementation_identities:
        _add_failure(failures, "candidate_and_reference_server_identities_are_not_distinct")
    pinned_reference_revision = reference.required_source_revision_valid
    if not pinned_reference_revision:
        _add_failure(failures, "reference_revision_not_pinned")
    successful_requests = reference.successful_requests and candidate.successful_requests
    request_identity = bool(
        reference.requests_valid
        and candidate.requests_valid
        and reference.request_ids
        and reference.request_ids == candidate.request_ids
    )
    if not request_identity:
        _add_failure(
            failures,
            "request_identity_mismatch",
            reference_count=len(reference.request_ids),
            candidate_count=len(candidate.request_ids),
        )

    image_counts = bool(
        reference.image_counts_valid
        and candidate.image_counts_valid
        and reference.images
        and set(reference.images) == set(candidate.images)
    )
    if not image_counts:
        _add_failure(
            failures,
            "image_identity_mismatch",
            reference_count=len(reference.images),
            candidate_count=len(candidate.images),
        )

    declared_dimensions = (
        reference.declared_dimensions_valid and candidate.declared_dimensions_valid
    )
    if request_identity and image_counts:
        for key in sorted(reference.images):
            reference_image = reference.images[key]
            candidate_image = candidate.images[key]
            if (
                reference_image.declared_width != candidate_image.declared_width
                or reference_image.declared_height != candidate_image.declared_height
            ):
                declared_dimensions = False
                _add_failure(
                    failures,
                    "paired_declared_dimensions_mismatch",
                    request_id_sha256=_redacted_request_id(key[0]),
                    image_index=key[1],
                )
    if not declared_dimensions:
        _add_failure(failures, "declared_dimensions_invalid")

    runtime: _QualityRuntime | None = None
    quality_dependencies = False
    runtime_factory = _runtime_factory or _load_quality_runtime
    try:
        runtime = runtime_factory()
        quality_dependencies = True
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        _add_failure(failures, "quality_runtime_unavailable")

    structural_gates = (
        distinct_artifacts
        and canonical_artifacts
        and image_workloads
        and semantic_parity
        and comparison_roles
        and active_server_profile_bindings
        and distinct_server_implementation_identities
        and pinned_reference_revision
        and successful_requests
        and request_identity
        and image_counts
        and declared_dimensions
        and quality_dependencies
    )
    pair_results: list[dict[str, Any]] = []
    pair_keys = sorted(reference.images) if request_identity and image_counts else []
    if structural_gates and runtime is not None:
        for pair_index, key in enumerate(pair_keys, start=1):
            reference_image = reference.images[key]
            candidate_image = candidate.images[key]
            try:
                reference_pixels = _load_bound_rgb(reference_image, runtime.numpy)
                candidate_pixels = _load_bound_rgb(candidate_image, runtime.numpy)
                if reference_pixels.shape != candidate_pixels.shape:
                    raise ValueError("paired image arrays have different shapes")
                if min(reference_pixels.shape[:2]) < 7:
                    raise ValueError("declared image is too small for the fixed SSIM window")
                metrics = runtime.compare(reference_pixels, candidate_pixels)
                if not _metric_values_valid(metrics):
                    raise ValueError("image metric result is invalid")
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                _add_failure(
                    failures,
                    "metric_computation_failed",
                    pair_index=pair_index,
                    request_id_sha256=_redacted_request_id(key[0]),
                    image_index=key[1],
                )
                continue
            threshold_checks = _threshold_checks(metrics)
            pair_results.append(
                {
                    "pair_index": pair_index,
                    "request_id_sha256": _redacted_request_id(key[0]),
                    "image_index": key[1],
                    "width": reference_image.declared_width,
                    "height": reference_image.declared_height,
                    "metrics": {name: _finite_or_none(value) for name, value in metrics.items()},
                    "nonfinite_metrics": {
                        name: _nonfinite_label(value)
                        for name, value in metrics.items()
                        if not math.isfinite(value)
                    },
                    "threshold_checks": threshold_checks,
                }
            )

    metrics_computed = bool(pair_keys) and len(pair_results) == len(pair_keys)
    if structural_gates and not metrics_computed:
        _add_failure(
            failures,
            "incomplete_metric_results",
            expected=len(pair_keys),
            observed=len(pair_results),
        )

    lpips_gate = bool(
        metrics_computed
        and all(result["threshold_checks"]["lpips_maximum"] for result in pair_results)
    )
    diagnostic_names = (
        "psnr_db_minimum",
        "uint8_mae_maximum",
        "cosine_similarity_minimum",
        "relative_l2_maximum",
    )
    diagnostic_threshold_checks = {
        name: bool(
            metrics_computed and all(result["threshold_checks"][name] for result in pair_results)
        )
        for name in diagnostic_names
    }

    evidence_gates = {
        "distinct_artifacts": distinct_artifacts,
        "canonical_artifacts": canonical_artifacts,
        "image_workloads": image_workloads,
        "semantic_parity": semantic_parity,
        "comparison_roles": comparison_roles,
        "active_server_profile_bindings": active_server_profile_bindings,
        "distinct_server_implementation_identities": distinct_server_implementation_identities,
        "pinned_reference_revision": pinned_reference_revision,
        "successful_requests": successful_requests,
        "request_identity": request_identity,
        "image_counts": image_counts,
        "declared_dimensions": declared_dimensions,
        "quality_dependencies": quality_dependencies,
        "metrics_computed": metrics_computed,
    }
    evidence_valid = all(evidence_gates.values())
    regression_canary_passed = evidence_valid and lpips_gate
    gates = {**evidence_gates, "lpips_maximum": lpips_gate}
    return {
        "schema_version": 1,
        "kind": "paired_image_smoke",
        "passed": regression_canary_passed,
        "evidence_valid": evidence_valid,
        "regression_canary_passed": regression_canary_passed,
        "artifact_bindings": {
            "reference": reference.binding,
            "candidate": candidate.binding,
        },
        "comparison_contract": {
            "decode": "exact bound bytes decoded by Pillow and converted to RGB uint8",
            "source_dimensions": "must equal each request's declared width and height",
            "lpips": {
                "network": "alex",
                "input_resize": [LPIPS_SIZE[0], LPIPS_SIZE[1]],
                "resize_interpolation": "bilinear",
                "resize_antialias": True,
                "normalization": "ToTensor then Normalize(mean=0.5,std=0.5)",
                "aggregation": "every paired output must pass",
                "acceptance_statistic": "maximum paired LPIPS",
            },
            "pixel_metrics": "full declared-resolution RGB uint8 arrays",
            "ssim": {
                "channel_axis": 2,
                "data_range": 255,
                "gaussian_weights": False,
                "use_sample_covariance": True,
                "win_size": 7,
                "gated": False,
            },
        },
        "thresholds": {
            "lpips": {
                "operator": "<=",
                "value": LPIPS_MAXIMUM,
                "diagnostic_only": True,
                "gated": True,
            },
            "psnr_db": {
                "operator": ">=",
                "value": PSNR_DB_MINIMUM,
                "diagnostic_only": True,
                "gated": False,
            },
            "uint8_mae": {
                "operator": "<=",
                "value": UINT8_MAE_MAXIMUM,
                "diagnostic_only": True,
                "gated": False,
            },
            "cosine_similarity": {
                "operator": ">=",
                "value": COSINE_SIMILARITY_MINIMUM,
                "diagnostic_only": True,
                "gated": False,
            },
            "relative_l2": {
                "operator": "<=",
                "value": RELATIVE_L2_MAXIMUM,
                "diagnostic_only": True,
                "gated": False,
            },
            "ssim": {
                "operator": None,
                "value": None,
                "diagnostic_only": True,
                "gated": False,
            },
        },
        "runtime": runtime.provenance if runtime is not None else None,
        "gates": gates,
        "diagnostic_threshold_checks": diagnostic_threshold_checks,
        "expected_pair_count": len(pair_keys),
        "pair_count": len(pair_results),
        "pairs": pair_results,
        "aggregate": _aggregate_pair_metrics(pair_results),
        "failures": failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the redacted machine evidence as a concise Markdown report."""
    lines = [
        "# Image smoke check",
        "",
        f"Evidence status: **{'valid' if report.get('evidence_valid') else 'invalid'}**.",
        "",
        f"Paired regression canary: **{'pass' if report.get('regression_canary_passed') else 'fail'}**.",
        "",
        "## Metrics",
        "",
        f"- Regression canary: maximum paired LPIPS (AlexNet, 256×256) ≤ {LPIPS_MAXIMUM}",
        f"- Diagnostic threshold: PSNR ≥ {PSNR_DB_MINIMUM} dB",
        f"- Diagnostic threshold: RGB uint8 MAE ≤ {UINT8_MAE_MAXIMUM}",
        f"- Diagnostic threshold: RGB uint8 cosine similarity ≥ {COSINE_SIMILARITY_MINIMUM}",
        f"- Diagnostic threshold: RGB uint8 relative L2 ≤ {RELATIVE_L2_MAXIMUM}",
        "- SSIM: reported as an ungated diagnostic",
        "",
        "## Evidence gates",
        "",
    ]
    for name, passed in report.get("gates", {}).items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")

    lines.extend(
        [
            "",
            "## Paired outputs",
            "",
            "| Pair | Request digest | Image | Dimensions | LPIPS | PSNR (dB) | MAE | Cosine | Relative L2 | SSIM | LPIPS canary |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for pair in report.get("pairs", []):
        metrics = pair.get("metrics", {})
        nonfinite = pair.get("nonfinite_metrics", {})
        checks = pair.get("threshold_checks", {})
        lines.append(
            "| "
            f"{pair.get('pair_index')} | "
            f"`{str(pair.get('request_id_sha256', ''))[:12]}` | "
            f"{pair.get('image_index')} | "
            f"{pair.get('width')}×{pair.get('height')} | "
            f"{_format_metric(metrics.get('lpips'), nonfinite.get('lpips'))} | "
            f"{_format_metric(metrics.get('psnr_db'), nonfinite.get('psnr_db'))} | "
            f"{_format_metric(metrics.get('uint8_mae'), nonfinite.get('uint8_mae'))} | "
            f"{_format_metric(metrics.get('cosine_similarity'), nonfinite.get('cosine_similarity'))} | "
            f"{_format_metric(metrics.get('relative_l2'), nonfinite.get('relative_l2'))} | "
            f"{_format_metric(metrics.get('ssim'), nonfinite.get('ssim'))} | "
            f"{'pass' if checks.get('lpips_maximum') else 'fail'} |"
        )

    failures = report.get("failures", [])
    if failures:
        lines.extend(["", "## Failed evidence", ""])
        for failure in failures:
            context = [
                f"{key}={json.dumps(value, sort_keys=True, ensure_ascii=True)}"
                for key, value in failure.items()
                if key != "code"
            ]
            suffix = f" ({', '.join(context)})" if context else ""
            lines.append(f"- `{failure.get('code', 'unknown')}`{suffix}")
    return "\n".join(lines) + "\n"


def _inspect_artifact(
    directory: Path,
    *,
    side: str,
    failures: list[dict[str, Any]],
    profile_config: dict[str, Any],
) -> _ArtifactInspection:
    summary = _safe_json(directory / "summary.json")
    if summary is None:
        _add_failure(failures, "summary_unreadable", side=side)
        return _empty_inspection()
    artifact = summary.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    contract = artifact.get("contract")
    contract_dict = contract if isinstance(contract, dict) else None
    contract_valid = contract_dict is not None and _fingerprint_valid(contract_dict)
    if not contract_valid:
        _add_failure(failures, "harness_contract_invalid", side=side)

    bundle_valid = False
    if contract_valid:
        try:
            assert contract_dict is not None
            bundle_valid = canonical_artifact_bundle_matches(directory, summary, contract_dict)
        except (KeyError, OSError, TypeError, ValueError):
            bundle_valid = False

    checks = artifact.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    matrix_contract = artifact.get("matrix_contract")
    matrix_valid = bool(
        checks.get("matrix_contract") is True and _matrix_contract_valid(matrix_contract, contract)
    )
    if not matrix_valid:
        _add_failure(failures, "matrix_contract_invalid", side=side)
    execution_bundle = artifact.get("execution_bundle_contract")
    execution_bundle_valid = bool(
        checks.get("execution_bundle_contract") is True
        and isinstance(matrix_contract, dict)
        and _execution_bundle_valid(directory, execution_bundle, matrix_contract)
    )
    if not execution_bundle_valid:
        _add_failure(failures, "execution_bundle_invalid", side=side)
    canonical_valid = bundle_valid and matrix_valid and execution_bundle_valid
    if not canonical_valid:
        _add_failure(failures, "canonical_artifact_invalid", side=side)

    semantic_contract = (
        _matrix_semantic_contract(matrix_contract, contract_dict)
        if contract_dict is not None and contract_valid and matrix_valid
        else None
    )
    if semantic_contract is None:
        _add_failure(failures, "semantic_contract_invalid", side=side)
    matrix = matrix_contract if isinstance(matrix_contract, dict) else {}
    comparison_role = matrix.get("comparison_role")
    if comparison_role not in {"candidate", "reference"}:
        comparison_role = None
        _add_failure(failures, "comparison_role_invalid", side=side)
    server_profile = matrix.get("server_profile")
    if not isinstance(server_profile, str) or not server_profile:
        server_profile = None
        _add_failure(failures, "server_profile_invalid", side=side)
    server_profile_binding_matches = benchmark_matrix_profile_binding_matches(
        matrix,
        profile_config,
    )
    if not server_profile_binding_matches:
        _add_failure(failures, "server_profile_binding_invalid", side=side)
    server_execution_identity = _server_execution_identity(matrix.get("server_execution"))
    required_revision = matrix.get("required_server_source_revision")
    required_source_role = matrix.get("required_server_source_role")
    required_source_revision_valid = bool(
        isinstance(required_revision, str)
        and required_revision
        and isinstance(required_source_role, str)
        and required_source_role
        and _required_source_revision_matches(
            matrix,
            required_revision,
            required_source_role,
        )
    )
    if required_revision is not None and not required_source_revision_valid:
        _add_failure(failures, "required_source_revision_mismatch", side=side)
    spec = summary.get("spec")
    task = spec.get("task") if isinstance(spec, dict) else None
    image_workload = task in _IMAGE_TASKS
    if not image_workload:
        _add_failure(failures, "non_image_workload", side=side)

    records = _safe_jsonl(directory / "requests.jsonl")
    if records is None or not records:
        _add_failure(failures, "requests_unreadable", side=side)
        records = []
    requests_valid = bool(records)
    successful_requests = bool(records)
    image_counts_valid = bool(records)
    declared_dimensions_valid = bool(records)
    selected_rows = contract.get("selected_rows") if isinstance(contract, dict) else None
    selected_count = selected_rows.get("count") if isinstance(selected_rows, dict) else None
    if not (
        isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and selected_count > 0
        and selected_count == len(records)
        and summary.get("request_count") == len(records)
    ):
        requests_valid = False
        _add_failure(
            failures,
            "selected_request_count_mismatch",
            side=side,
            selected_count=(
                selected_count
                if isinstance(selected_count, int) and not isinstance(selected_count, bool)
                else None
            ),
            observed_count=len(records),
        )
    request_ids: set[str] = set()
    images: dict[tuple[str, int], _ImageReference] = {}
    for record_index, record in enumerate(records, start=1):
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            requests_valid = False
            _add_failure(failures, "request_id_invalid", side=side, record=record_index)
            continue
        redacted_id = _redacted_request_id(request_id)
        if request_id in request_ids:
            requests_valid = False
            _add_failure(
                failures,
                "duplicate_request_id",
                side=side,
                request_id_sha256=redacted_id,
            )
            continue
        request_ids.add(request_id)
        if record.get("success") is not True:
            successful_requests = False
            _add_failure(
                failures,
                "request_not_successful",
                side=side,
                request_id_sha256=redacted_id,
            )
        if record.get("generated_images_expected") is not True:
            image_counts_valid = False
            _add_failure(
                failures,
                "generated_image_not_declared",
                side=side,
                request_id_sha256=redacted_id,
            )
        declared_count = _positive_int(record.get("requested_image_count"))
        declared_width = _positive_int(record.get("requested_image_width"))
        declared_height = _positive_int(record.get("requested_image_height"))
        outputs = record.get("image_outputs")
        if declared_count is None or not isinstance(outputs, list):
            image_counts_valid = False
            _add_failure(
                failures,
                "image_count_not_declared",
                side=side,
                request_id_sha256=redacted_id,
            )
            outputs = []
        if declared_count is not None and (
            len(outputs) != declared_count or record.get("images") != declared_count
        ):
            image_counts_valid = False
            _add_failure(
                failures,
                "declared_image_count_mismatch",
                side=side,
                request_id_sha256=redacted_id,
            )
        if declared_width is None or declared_height is None:
            declared_dimensions_valid = False
            _add_failure(
                failures,
                "image_dimensions_not_declared",
                side=side,
                request_id_sha256=redacted_id,
            )
            continue
        for image_index, output in enumerate(outputs):
            if not isinstance(output, dict):
                declared_dimensions_valid = False
                continue
            if output.get("width") != declared_width or output.get("height") != declared_height:
                declared_dimensions_valid = False
                _add_failure(
                    failures,
                    "output_dimensions_differ_from_declaration",
                    side=side,
                    request_id_sha256=redacted_id,
                    image_index=image_index,
                )
            filename = output.get("sample_filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                declared_dimensions_valid = False
                continue
            images[(request_id, image_index)] = _ImageReference(
                request_id=request_id,
                image_index=image_index,
                path=directory / "samples" / filename,
                metadata=output,
                declared_width=declared_width,
                declared_height=declared_height,
            )

    return _ArtifactInspection(
        canonical_valid=canonical_valid,
        image_workload=image_workload,
        semantic_contract=semantic_contract,
        comparison_role=comparison_role,
        server_profile=server_profile,
        server_profile_binding_matches=server_profile_binding_matches,
        server_execution_identity=server_execution_identity,
        required_source_revision_valid=required_source_revision_valid,
        requests_valid=requests_valid,
        successful_requests=successful_requests,
        image_counts_valid=image_counts_valid,
        declared_dimensions_valid=declared_dimensions_valid,
        request_ids=request_ids,
        images=images,
        binding=_artifact_binding(artifact),
    )


def _matrix_contract_valid(value: Any, harness_contract: Any) -> bool:
    required = {
        "schema_version",
        "benchmark",
        "benchmark_profile",
        "benchmark_definition",
        "server_profile",
        "server_profile_contract",
        "server_profile_binding",
        "comparison_role",
        "required_server_source_revision",
        "required_server_source_role",
        "model_revision_contract",
        "execution_policy",
        "server_execution",
        "harness_execution",
        "hardware",
        "harness_contract_fingerprint",
        "parity_group",
        "parity_contract",
        "fingerprint",
    }
    if not (
        isinstance(value, dict)
        and set(value) == required
        and value.get("schema_version") == 2
        and _fingerprint_valid(value)
        and isinstance(value.get("benchmark"), str)
        and bool(value.get("benchmark"))
        and isinstance(value.get("benchmark_profile"), str)
        and bool(value.get("benchmark_profile"))
        and isinstance(value.get("server_profile"), str)
        and bool(value.get("server_profile"))
        and isinstance(value.get("parity_group"), str)
        and bool(value.get("parity_group"))
        and value.get("comparison_role") in {"candidate", "reference"}
        and isinstance(harness_contract, dict)
        and value.get("harness_contract_fingerprint") == harness_contract.get("fingerprint")
    ):
        return False
    for name in (
        "benchmark_definition",
        "server_profile_contract",
        "server_profile_binding",
        "server_execution",
        "harness_execution",
        "hardware",
    ):
        nested = value.get(name)
        if not isinstance(nested, dict) or not _fingerprint_valid(nested):
            return False
    parity = value.get("parity_contract")
    if not (
        isinstance(parity, dict)
        and set(parity) == {"schema_version", "harness", "model", "fingerprint"}
        and parity.get("schema_version") == 2
        and _fingerprint_valid(parity)
    ):
        return False
    harness = parity.get("harness")
    model = parity.get("model")
    return bool(
        isinstance(harness, dict)
        and _fingerprint_valid(harness)
        and isinstance(model, dict)
        and model
    )


def _server_execution_identity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if key not in {"fingerprint", "environment"}}


def _required_source_revision_matches(
    matrix_contract: dict[str, Any],
    expected: Any,
    expected_role: Any,
) -> bool:
    if not (
        isinstance(expected, str)
        and len(expected) == 40
        and isinstance(expected_role, str)
        and expected_role
    ):
        return False
    execution = matrix_contract.get("server_execution")
    if not isinstance(execution, dict):
        return False
    for revision in execution.get("source_revisions", []):
        if not isinstance(revision, dict):
            continue
        state = revision.get("state")
        roles = revision.get("roles")
        if (
            isinstance(roles, list)
            and expected_role in roles
            and isinstance(state, dict)
            and state.get("head") == expected
        ):
            return True
    return False


def _matrix_semantic_contract(
    matrix_contract: Any, contract: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        harness_parity = benchmark_parity_contract(contract)
    except (TypeError, ValueError):
        return None
    if not _matrix_contract_valid(matrix_contract, contract):
        return None
    assert isinstance(matrix_contract, dict)
    matrix_parity = matrix_contract.get("parity_contract")
    assert isinstance(matrix_parity, dict)
    if matrix_parity.get("harness") != harness_parity:
        return None
    return matrix_parity


def _execution_bundle_valid(
    directory: Path,
    value: Any,
    matrix_contract: dict[str, Any],
) -> bool:
    required = {
        "schema_version",
        "benchmark",
        "server_group",
        "point_files",
        "server_files",
        "fingerprint",
    }
    if not (
        isinstance(value, dict)
        and set(value) == required
        and value.get("schema_version") == 2
        and _fingerprint_valid(value)
        and value.get("benchmark") == matrix_contract.get("benchmark")
        and isinstance(value.get("server_group"), str)
        and bool(value.get("server_group"))
    ):
        return False
    point_files = value.get("point_files")
    server_files = value.get("server_files")
    if not (
        isinstance(point_files, dict)
        and set(point_files) == _POINT_SUPPORT_FILES
        and isinstance(server_files, dict)
        and set(server_files) == _SERVER_SUPPORT_FILES
        and _file_contracts_match(directory, point_files)
    ):
        return False
    root = _execution_bundle_root(directory, value)
    return root is not None and _file_contracts_match(
        root / "servers" / value["server_group"] / value["benchmark"],
        server_files,
    )


def _execution_bundle_root(directory: Path, contract: dict[str, Any]) -> Path | None:
    benchmark = contract.get("benchmark")
    server_group = contract.get("server_group")
    if not (
        isinstance(benchmark, str) and benchmark and isinstance(server_group, str) and server_group
    ):
        return None
    resolved = directory.resolve()
    roots = {
        ancestor
        for ancestor in (resolved, *resolved.parents)
        if (ancestor / "servers" / server_group / benchmark).is_dir()
    }
    return next(iter(roots)) if len(roots) == 1 else None


def _file_contracts_match(directory: Path, contracts: dict[str, Any]) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    for name, expected in contracts.items():
        if not (
            isinstance(name, str)
            and name
            and Path(name).name == name
            and "/" not in name
            and "\\" not in name
            and isinstance(expected, dict)
            and set(expected) == {"size_bytes", "sha256"}
        ):
            return False
        expected_size = expected.get("size_bytes")
        expected_sha256 = expected.get("sha256")
        if not (
            isinstance(expected_size, int)
            and not isinstance(expected_size, bool)
            and expected_size >= 0
            and isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
        ):
            return False
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return False
        try:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError:
            return False
        if size != expected_size or digest.hexdigest() != expected_sha256:
            return False
    return True


def _artifact_binding(artifact: dict[str, Any]) -> dict[str, Any]:
    contract = artifact.get("contract")
    matrix_contract = artifact.get("matrix_contract")
    execution_bundle = artifact.get("execution_bundle_contract")
    request_records = artifact.get("request_records")
    image_samples = artifact.get("image_samples")
    summary_payload = artifact.get("summary_payload")
    return {
        "contract_fingerprint": (
            contract.get("fingerprint") if isinstance(contract, dict) else None
        ),
        "matrix_contract_fingerprint": (
            matrix_contract.get("fingerprint") if isinstance(matrix_contract, dict) else None
        ),
        "execution_bundle_fingerprint": (
            execution_bundle.get("fingerprint") if isinstance(execution_bundle, dict) else None
        ),
        "request_records_sha256": (
            request_records.get("sha256") if isinstance(request_records, dict) else None
        ),
        "image_samples_sha256": (
            image_samples.get("sha256") if isinstance(image_samples, dict) else None
        ),
        "summary_payload_sha256": (
            summary_payload.get("sha256") if isinstance(summary_payload, dict) else None
        ),
    }


def _load_bound_rgb(reference: _ImageReference, numpy: Any) -> Any:
    if reference.path.is_symlink() or not reference.path.is_file():
        raise ValueError("bound image is not one regular file")
    data = reference.path.read_bytes()
    decoded = inspect_image_bytes(data, declared_mime=reference.metadata.get("mime"))
    expected_metadata = {
        key: reference.metadata.get(key)
        for key in ("sha256", "byte_size", "mime", "width", "height", "sample_filename")
    }
    if decoded.metadata_dict() != expected_metadata:
        raise ValueError("bound image metadata changed during the smoke check")
    if (decoded.width, decoded.height) != (
        reference.declared_width,
        reference.declared_height,
    ):
        raise ValueError("bound image dimensions differ from the request declaration")
    with Image.open(BytesIO(data)) as image:
        image.load()
        rgb = image.convert("RGB")
        pixels = numpy.asarray(rgb, dtype=numpy.uint8).copy()
    expected_shape = (reference.declared_height, reference.declared_width, 3)
    if pixels.shape != expected_shape:
        raise ValueError("decoded RGB array has the wrong declared shape")
    return pixels


def _load_quality_runtime() -> _QualityRuntime:
    try:
        import lpips
        import numpy
        import torch
        from skimage.metrics import structural_similarity
        from torchvision import transforms
    except ImportError as error:
        raise QualityRuntimeError("install the benchmark dependency set") from error
    try:
        model = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().cpu().float()
    except (OSError, RuntimeError, ValueError) as error:
        raise QualityRuntimeError("LPIPS AlexNet weights are unavailable") from error
    transform = transforms.Compose(
        [
            transforms.Resize(
                LPIPS_SIZE,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    provenance = {
        "device": "cpu",
        "device_policy": "fixed_cpu_no_accelerator_fallback",
        "deterministic_algorithms": True,
        "inference_mode": True,
        "input_dtype": str(torch.float32),
        "lpips_network": "alex",
        "lpips_version": _distribution_version("lpips"),
        "lpips_model_state_sha256": _model_state_digest(model, torch),
        "model_mode": "eval",
        "model_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
        "numpy_version": numpy.__version__,
        "pillow_version": _distribution_version("pillow"),
        "scikit_image_version": _distribution_version("scikit-image"),
        "torch_version": torch.__version__,
        "torchvision_version": _distribution_version("torchvision"),
    }
    return _QualityRuntime(
        model=model,
        transform=transform,
        torch=torch,
        numpy=numpy,
        structural_similarity=structural_similarity,
        provenance=provenance,
    )


def _model_state_digest(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        _update_framed_digest(digest, name.encode("utf-8"))
        _update_framed_digest(digest, str(contiguous.dtype).encode("ascii"))
        _update_framed_digest(
            digest,
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"),
        )
        _update_framed_digest(digest, contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _update_framed_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _threshold_checks(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "lpips_maximum": math.isfinite(metrics["lpips"]) and metrics["lpips"] <= LPIPS_MAXIMUM,
        "psnr_db_minimum": metrics["psnr_db"] >= PSNR_DB_MINIMUM,
        "uint8_mae_maximum": math.isfinite(metrics["uint8_mae"])
        and metrics["uint8_mae"] <= UINT8_MAE_MAXIMUM,
        "cosine_similarity_minimum": math.isfinite(metrics["cosine_similarity"])
        and metrics["cosine_similarity"] >= COSINE_SIMILARITY_MINIMUM,
        "relative_l2_maximum": metrics["relative_l2"] <= RELATIVE_L2_MAXIMUM,
    }


def _metric_values_valid(metrics: Any) -> bool:
    names = {
        "lpips",
        "ssim",
        "psnr_db",
        "uint8_mae",
        "cosine_similarity",
        "relative_l2",
    }
    if not isinstance(metrics, dict) or set(metrics) != names:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in metrics.values()
    ):
        return False
    finite_names = {"lpips", "ssim", "uint8_mae", "cosine_similarity"}
    if any(not math.isfinite(float(metrics[name])) for name in finite_names):
        return False
    psnr = float(metrics["psnr_db"])
    relative_l2 = float(metrics["relative_l2"])
    return bool(
        not math.isnan(psnr)
        and psnr != -math.inf
        and not math.isnan(relative_l2)
        and relative_l2 != -math.inf
        and relative_l2 >= 0.0
    )


def _aggregate_pair_metrics(pair_results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not pair_results:
        return None
    metric_names = (
        "lpips",
        "ssim",
        "psnr_db",
        "uint8_mae",
        "cosine_similarity",
        "relative_l2",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for pair in pair_results:
        metrics = pair["metrics"]
        nonfinite = pair["nonfinite_metrics"]
        for name in metric_names:
            value = metrics[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name].append(float(value))
            elif nonfinite.get(name) == "positive_infinity":
                values[name].append(math.inf)
            elif nonfinite.get(name) == "negative_infinity":
                values[name].append(-math.inf)
    return {name: _metric_distribution(metric_values) for name, metric_values in values.items()}


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "positive_infinity_count": sum(value == math.inf for value in values),
        "negative_infinity_count": sum(value == -math.inf for value in values),
        "finite_mean": statistics.fmean(finite) if finite else None,
        "finite_minimum": min(finite) if finite else None,
        "finite_p95": _linear_percentile(finite, 0.95) if finite else None,
        "finite_maximum": max(finite) if finite else None,
    }


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _safe_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if path.is_symlink() or not path.is_file():
        return None
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return records


def _fingerprint_valid(contract: dict[str, Any]) -> bool:
    fingerprint = contract.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        return False
    payload = {key: value for key, value in contract.items() if key != "fingerprint"}
    try:
        return fingerprint == canonical_digest(payload)
    except (TypeError, ValueError):
        return False


def _empty_inspection() -> _ArtifactInspection:
    return _ArtifactInspection(
        canonical_valid=False,
        image_workload=False,
        semantic_contract=None,
        comparison_role=None,
        server_profile=None,
        server_profile_binding_matches=False,
        server_execution_identity=None,
        required_source_revision_valid=False,
        requests_valid=False,
        successful_requests=False,
        image_counts_valid=False,
        declared_dimensions_valid=False,
        request_ids=set(),
        images={},
        binding={
            "contract_fingerprint": None,
            "matrix_contract_fingerprint": None,
            "execution_bundle_fingerprint": None,
            "request_records_sha256": None,
            "image_samples_sha256": None,
            "summary_payload_sha256": None,
        },
    )


def _redacted_request_id(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _nonfinite_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0.0 else "negative_infinity"


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as error:
        raise QualityRuntimeError(f"required distribution {name} is unavailable") from error


def _add_failure(failures: list[dict[str, Any]], code: str, **context: Any) -> None:
    failure = {"code": code, **context}
    if failure not in failures:
        failures.append(failure)


def _format_metric(value: Any, nonfinite: Any) -> str:
    if nonfinite == "positive_infinity":
        return "∞"
    if nonfinite == "negative_infinity":
        return "-∞"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.6g}"
