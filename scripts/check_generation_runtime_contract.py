#!/usr/bin/env python3
"""Validate the canonical generation-runtime contracts and frozen profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from uniserve_eval.profiles import benchmark_matrix_definition_contract, load_config

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    Path("specs/decode-runtime.md"),
    Path("specs/serving-surface.md"),
    Path("specs/decode-runtime-construction.md"),
    Path("specs/generation-runtime-qualification.md"),
    Path("docs/benchmark-protocol.md"),
)
REFERENCE_PATH = Path("qualification/generation-runtime/references.json")
REFERENCE_SCHEMA_PATH = Path("schemas/generation-runtime-references.schema.json")
CANDIDATE_SCHEMA_PATH = Path("schemas/generation-runtime-candidate.schema.json")
CANDIDATE_PLAN_SCHEMA_PATH = Path("schemas/generation-runtime-candidate-plan.schema.json")
PROFILE_PATH = Path("uniserve_eval/profiles.json")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")


class ContractError(ValueError):
    """A canonical contract input is invalid."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_schema(schema_path: Path, instance_path: Path | None = None) -> None:
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
        if instance_path is not None:
            Draft202012Validator(schema).validate(_load_json(instance_path))
    except Exception as error:
        subject = instance_path or schema_path
        raise ContractError(f"schema validation failed for {subject}: {error}") from error


def _validate_markdown_links(path: Path) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for match in MARKDOWN_LINK_RE.finditer(text):
        raw_target = match.group("target").strip("<>")
        if raw_target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_text = raw_target.split("#", 1)[0]
        if not target_text:
            continue
        target = (ROOT / path).parent / target_text
        if not target.exists():
            line = text.count("\n", 0, match.start()) + 1
            raise ContractError(f"{path}:{line} references missing path {raw_target!r}")


def _git_object_exists(object_id: str) -> bool:
    process = subprocess.run(
        ["git", "cat-file", "-e", f"{object_id}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.returncode == 0


def _git_text(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ContractError(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout


def validate_p0(parent: str, candidate: str) -> dict[str, Any]:
    parent_commit = _git_text("rev-parse", "--verify", f"{parent}^{{commit}}").strip()
    candidate_commit = _git_text("rev-parse", "--verify", f"{candidate}^{{commit}}").strip()
    changed_paths = tuple(
        path
        for path in _git_text("diff", "--name-only", parent_commit, candidate_commit).splitlines()
        if path
    )
    executable_prefixes = ("crates/", "uniserve_kernel/", "uniserve_worker/")
    executable_files = {"scripts/run_benchmarks.py"}
    executable_changes = [
        path
        for path in changed_paths
        if path.startswith(executable_prefixes)
        or (path.startswith("uniserve_eval/") and path != "uniserve_eval/profiles.json")
        or path in executable_files
    ]
    if executable_changes:
        raise ContractError(
            "P0 candidate changes configured executable paths: " + ", ".join(executable_changes)
        )

    try:
        parent_config = json.loads(_git_text("show", f"{parent_commit}:{PROFILE_PATH}"))
        candidate_config = json.loads(_git_text("show", f"{candidate_commit}:{PROFILE_PATH}"))
    except json.JSONDecodeError as error:
        raise ContractError(f"P0 profile snapshot is invalid JSON: {error}") from error
    if candidate_config != parent_config:
        normalized_candidate = deepcopy(candidate_config)
        benchmark = normalized_candidate["benchmarks"]["main"]
        expected_load_case = benchmark["load_cases"].pop("interleave_qualification", None)
        expected_group = benchmark["groups"].pop("sensenova-uniserve-stochastic", None)
        expected_point = benchmark["points"].pop(
            "sensenova_ueval_stochastic_interleave_uniserve", None
        )
        if not (
            expected_load_case
            == [{"id": "c4", "request_rate": "inf", "max_concurrency": 4}]
            and expected_group
            == {
                "server": "benchmark/server/sensenova-uniserve",
                "points": ["sensenova_ueval_stochastic_interleave_uniserve"],
            }
            and isinstance(expected_point, dict)
            and normalized_candidate == parent_config
        ):
            raise ContractError("P0 candidate changes an existing benchmark or serving profile")
    _validate_stochastic_profile(candidate_config, _load_json(REFERENCE_PATH))
    return {
        "parent": parent_commit,
        "candidate": candidate_commit,
        "changed_paths": list(changed_paths),
        "configured_executable_paths_changed": False,
        "existing_profiles_changed": False,
    }


def _validate_performance_reference(
    reference: dict[str, Any],
    construction_anchor: dict[str, Any],
) -> None:
    source_commit = reference["source_commit"]
    measurement_commit = reference["measurement_commit"]
    if not _git_object_exists(measurement_commit):
        raise ContractError(f"performance measurement commit is unavailable: {measurement_commit}")
    if _git_text("rev-parse", f"{source_commit}^{{tree}}").strip() != reference["source_tree"]:
        raise ContractError("performance reference source tree does not match its commit")
    if (
        _git_text("rev-parse", f"{measurement_commit}^{{tree}}").strip()
        != reference["measurement_tree"]
    ):
        raise ContractError("performance reference measurement tree does not match its commit")
    parents = _git_text("rev-list", "--parents", "-n", "1", measurement_commit).split()
    if parents != [measurement_commit, source_commit]:
        raise ContractError("performance measurement must have the named source as its sole parent")
    overlay = reference["control_overlay"]
    changed = _git_text("diff", "--name-only", source_commit, measurement_commit).splitlines()
    if changed != overlay["paths"]:
        raise ContractError("performance measurement control overlay paths do not match its diff")
    for path in overlay["paths"]:
        measured_blob = _git_text("rev-parse", f"{measurement_commit}:{path}").strip()
        contract_blob = _git_text("rev-parse", f"{overlay['contract_commit']}:{path}").strip()
        if measured_blob != contract_blob:
            raise ContractError(f"performance measurement control differs from its contract: {path}")
    if reference["qualified_gates"] != ["P1"]:
        raise ContractError("performance reference qualification must name its exact completed gates")
    if reference["pending_points"] != ["sensenova_default_travel"]:
        raise ContractError("performance reference pending point set is not canonical")

    point_contracts = {
        "qwen3_sharegpt_r16": 200,
        "sensenova_t2i_c32": 32,
        "sensenova_i2t_c32": 32,
    }
    if set(reference["artifacts"]) != set(point_contracts):
        raise ContractError("performance reference P1 artifact set is incomplete")
    for name, request_count in point_contracts.items():
        root = ROOT / reference["artifacts"][name]["root"]
        summary = _load_json(root / "summary.json")
        artifact = summary.get("artifact") if isinstance(summary, dict) else None
        matrix = artifact.get("matrix_contract") if isinstance(artifact, dict) else None
        policy = matrix.get("execution_policy") if isinstance(matrix, dict) else None
        build = policy.get("build_manifest") if isinstance(policy, dict) else None
        source = build.get("source_state") if isinstance(build, dict) else None
        anchor_root = ROOT / construction_anchor["artifacts"][name]["root"]
        anchor_summary = _load_json(anchor_root / "summary.json")
        anchor_artifact = anchor_summary.get("artifact")
        anchor_matrix = (
            anchor_artifact.get("matrix_contract") if isinstance(anchor_artifact, dict) else None
        )
        expected = (
            anchor_matrix.get("benchmark_definition") if isinstance(anchor_matrix, dict) else None
        )
        definition = matrix.get("benchmark_definition") if isinstance(matrix, dict) else None
        if not (
            artifact.get("valid") is True
            and artifact.get("valid_marker") == "canonical-valid-v2"
            and summary.get("request_count") == request_count
            and summary.get("ok_count") == request_count
            and summary.get("failed_count") == 0
            and isinstance(source, dict)
            and source.get("head") == measurement_commit
            and source.get("dirty") is False
            and definition == expected
        ):
            raise ContractError(f"performance reference artifact is not canonical: {name}")


def _validate_reference_artifacts(
    references: dict[str, Any],
) -> None:
    for set_name in ("construction_anchor", "performance_reference"):
        reference = references[set_name]
        source_commit = reference["source_commit"]
        if not _git_object_exists(source_commit):
            raise ContractError(f"{set_name} source commit is unavailable: {source_commit}")
        for artifact_name, artifact in reference["artifacts"].items():
            root = ROOT / artifact["root"]
            if not root.is_dir():
                raise ContractError(f"{set_name}.{artifact_name} root is unavailable: {root}")
            if "manifest_sha256" in artifact:
                manifest = root / "artifact_manifest.json"
                if not manifest.is_file() or _sha256(manifest) != artifact["manifest_sha256"]:
                    raise ContractError(f"{set_name}.{artifact_name} manifest digest mismatch")
            if "summary_sha256" in artifact:
                summary = root / "summary.json"
                if not summary.is_file() or _sha256(summary) != artifact["summary_sha256"]:
                    raise ContractError(f"{set_name}.{artifact_name} summary digest mismatch")
    _validate_performance_reference(
        references["performance_reference"],
        references["construction_anchor"],
    )
    for anchor_name, anchor in references["profile_anchors"].items():
        if not _git_object_exists(anchor["source_commit"]):
            raise ContractError(f"{anchor_name} source commit is unavailable")


def _expanded_points(config: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    benchmark = config["benchmarks"]["main"]
    groups = benchmark["groups"]
    points = benchmark["points"]
    load_cases = benchmark["load_cases"]
    expanded: list[tuple[str, str, str, str]] = []
    names: set[str] = set()
    for group_name, group in groups.items():
        for point_name in group["points"]:
            point = points[point_name]
            load_case_set = point["load_case_set"]
            for load_case in load_cases[load_case_set]:
                load_case_id = load_case["id"]
                name = point["name"].format(load_id=load_case_id)
                if name in names:
                    raise ContractError(f"expanded benchmark name is duplicated: {name}")
                names.add(name)
                benchmark_matrix_definition_contract(
                    config,
                    "main",
                    group_name=group_name,
                    point_name=point_name,
                    load_case_set=load_case_set,
                    load_case_id=load_case_id,
                )
                expanded.append((group_name, point_name, load_case_set, load_case_id))
    return expanded


def _validate_stochastic_profile(config: dict[str, Any], references: dict[str, Any]) -> str:
    benchmark = config["benchmarks"]["main"]
    group_name = "sensenova-uniserve-stochastic"
    point_name = "sensenova_ueval_stochastic_interleave_uniserve"
    load_case_set = "interleave_qualification"
    load_case_id = "c4"
    group = benchmark["groups"].get(group_name)
    if group != {
        "server": "benchmark/server/sensenova-uniserve",
        "points": [point_name],
    }:
        raise ContractError("stochastic interleave group does not have its frozen declaration")
    cases = benchmark["load_cases"].get(load_case_set)
    if cases != [{"id": "c4", "request_rate": "inf", "max_concurrency": 4}]:
        raise ContractError("stochastic interleave load case does not have its frozen declaration")
    point = benchmark["points"].get(point_name)
    if not isinstance(point, dict):
        raise ContractError("stochastic interleave point is unavailable")
    harness = point.get("harness")
    expected = {
        "task": "interleave",
        "model": "SenseNova-U1",
        "dataset": "ueval",
        "dataset_revision": "fdeac6654b113d20e7e89d496167a4a5bb55bc66",
        "num_prompts": 32,
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "sampling_seed": 42,
        "disable_ignore_eos": True,
        "steps": 50,
        "denoise_updates": 50,
        "width": 2048,
        "height": 1152,
        "wire": "openai_chat",
        "guidance_scale": 4.0,
        "image_guidance_scale": 1.0,
        "cfg_norm": "none",
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 3.0,
        "image_think": False,
        "image_t_eps": 0.02,
        "runtime_profile_id": "sensenova-u1",
        "plan_evidence_policy": "runtime_inspection",
        "output_constraint": "default",
        "preprocessing": "ueval_prompt_text_v1",
        "acceptance_min_success": 32,
        "acceptance_min_images_per_success": 1.0,
    }
    if harness != expected:
        raise ContractError("stochastic interleave request controls differ from the frozen profile")
    contract = benchmark_matrix_definition_contract(
        config,
        "main",
        group_name=group_name,
        point_name=point_name,
        load_case_set=load_case_set,
        load_case_id=load_case_id,
    )
    anchor = references["profile_anchors"]["sensenova_stochastic_interleave_c4"]
    if contract["fingerprint"] != anchor["benchmark_definition_fingerprint"]:
        raise ContractError("stochastic interleave benchmark definition differs from its anchor")
    return contract["fingerprint"]


def validate_contract() -> dict[str, Any]:
    _validate_schema(CANDIDATE_SCHEMA_PATH)
    _validate_schema(CANDIDATE_PLAN_SCHEMA_PATH)
    _validate_schema(REFERENCE_SCHEMA_PATH, REFERENCE_PATH)
    for document in CANONICAL_DOCUMENTS:
        _validate_markdown_links(document)
    config = load_config(ROOT / PROFILE_PATH)
    references = _load_json(REFERENCE_PATH)
    _validate_reference_artifacts(references)
    expanded = _expanded_points(config)
    if len(expanded) != 46:
        raise ContractError(f"main benchmark must expand to 46 serial points, found {len(expanded)}")
    stochastic_fingerprint = _validate_stochastic_profile(config, references)
    file_digests = {
        str(path): _sha256(ROOT / path)
        for path in (*CANONICAL_DOCUMENTS, PROFILE_PATH, REFERENCE_PATH)
    }
    payload = {
        "schema_version": 1,
        "valid": True,
        "expanded_point_count": len(expanded),
        "stochastic_profile_fingerprint": stochastic_fingerprint,
        "file_digests": file_digests,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "fingerprint": hashlib.sha256(canonical).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--p0-parent")
    parser.add_argument("--p0-candidate")
    args = parser.parse_args()
    try:
        report = validate_contract()
        if (args.p0_parent is None) != (args.p0_candidate is None):
            raise ContractError("P0 validation requires both --p0-parent and --p0-candidate")
        if args.p0_parent is not None and args.p0_candidate is not None:
            report["p0"] = validate_p0(args.p0_parent, args.p0_candidate)
    except ContractError as error:
        print(f"generation runtime contract invalid: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        output = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
