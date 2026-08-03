#!/usr/bin/env python3
"""Materialize, validate, execute, and accept generation-runtime candidates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = Path("schemas/generation-runtime-candidate-plan.schema.json")
CANDIDATE_SCHEMA = Path("schemas/generation-runtime-candidate.schema.json")
PERFORMANCE_GATES = frozenset({"P1", "P2", "P3", "P4"})
REQUIRED_PERFORMANCE_COMPARATORS = frozenset(
    {"construction_anchor", "performance_reference", "previous_candidate"}
)


class ControllerError(RuntimeError):
    """Candidate construction or acceptance violated the runtime contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControllerError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ControllerError(f"JSON document must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_json(repo: Path, schema_path: Path, value: dict[str, Any]) -> None:
    schema = _read_json(repo / schema_path)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except Exception as error:
        raise ControllerError(f"schema validation failed against {schema_path}: {error}") from error


class GitRepo:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def run(
        self,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            ["git", *arguments],
            cwd=self.path,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise ControllerError(f"git {' '.join(arguments)} failed: {detail}")
        return process

    def text(self, *arguments: str) -> str:
        return self.run(*arguments).stdout.strip()

    def commit(self, revision: str) -> str:
        return self.text("rev-parse", "--verify", f"{revision}^{{commit}}")

    def tree(self, revision: str) -> str:
        return self.text("rev-parse", "--verify", f"{revision}^{{tree}}")

    def ref(self, name: str) -> str:
        return self.text("rev-parse", "--verify", name)

    def common_dir(self) -> Path:
        path = Path(self.text("rev-parse", "--git-common-dir"))
        return path.resolve() if path.is_absolute() else (self.path / path).resolve()


def _ordered_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks = list(manifest["checks"])
    ids = [check["id"] for check in checks]
    orders = [check["order"] for check in checks]
    if len(set(ids)) != len(ids):
        raise ControllerError("check identifiers must be unique")
    if len(set(orders)) != len(orders):
        raise ControllerError("check serial orders must be unique")
    ordered = sorted(checks, key=lambda check: int(check["order"]))
    completed: set[str] = set()
    for check in ordered:
        prerequisites = set(check.get("prerequisites", []))
        unknown = prerequisites - set(ids)
        if unknown:
            raise ControllerError(
                f"check {check['id']} has unknown prerequisites: {', '.join(sorted(unknown))}"
            )
        unavailable = prerequisites - completed
        if unavailable:
            raise ControllerError(
                f"check {check['id']} is ordered before prerequisites: "
                + ", ".join(sorted(unavailable))
            )
        completed.add(check["id"])
    return ordered


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_measurement_point(repo: GitRepo, point: dict[str, Any]) -> None:
    profile = point.get("profile")
    if profile is None:
        return
    profiles = _read_json(repo.path / "uniserve_eval/profiles.json")
    if point["kind"] == "verification":
        if profile not in profiles.get("workloads", {}):
            raise ControllerError(
                f"measurement point {point['id']} names unknown workload profile {profile}"
            )
        return
    benchmarks = profiles.get("benchmarks", {})
    benchmark = benchmarks.get(profile)
    if not isinstance(benchmark, dict):
        raise ControllerError(
            f"measurement point {point['id']} names unknown benchmark profile {profile}"
        )
    group_name = point.get("group")
    point_name = point.get("point")
    load_case_set = point.get("load_case_set")
    load_case_id = point.get("load_case_id")
    groups = benchmark.get("groups", {})
    points = benchmark.get("points", {})
    load_cases = benchmark.get("load_cases", {})
    group = groups.get(group_name) if isinstance(groups, dict) else None
    declaration = points.get(point_name) if isinstance(points, dict) else None
    cases = load_cases.get(load_case_set) if isinstance(load_cases, dict) else None
    matching_cases = (
        [case for case in cases if isinstance(case, dict) and case.get("id") == load_case_id]
        if isinstance(cases, list)
        else []
    )
    if not (
        isinstance(group, dict)
        and point_name in group.get("points", [])
        and isinstance(declaration, dict)
        and declaration.get("load_case_set") == load_case_set
        and len(matching_cases) == 1
    ):
        raise ControllerError(
            f"measurement point {point['id']} does not resolve to one active profile point"
        )


def _validate_manifest_semantics(repo: GitRepo, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_json(repo.path, CANDIDATE_SCHEMA, manifest)
    ordered = _ordered_checks(manifest)
    candidate = manifest["candidate"]
    parent = repo.commit(candidate["parent"])
    commit = repo.commit(candidate["commit"])
    parents = repo.text("rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, parent]:
        raise ControllerError("candidate must have exactly the declared accepted parent")
    declared_paths = sorted(manifest["affected_scope"]["paths"])
    actual_paths = sorted(
        path
        for path in repo.text("diff", "--name-only", parent, commit).splitlines()
        if path
    )
    if actual_paths != declared_paths:
        raise ControllerError(
            "candidate affected scope differs from its source diff: "
            f"declared={declared_paths}, actual={actual_paths}"
        )
    artifact_root = (repo.path / manifest["artifact_root"]).resolve()
    for check in ordered:
        artifact_path = check.get("artifact_path")
        if artifact_path is None:
            continue
        resolved = (repo.path / artifact_path).resolve()
        if not _path_is_within(resolved, artifact_root):
            raise ControllerError(f"check {check['id']} artifact is outside the candidate root")
    gates = {check["gate"] for check in ordered}
    if gates & PERFORMANCE_GATES:
        missing = REQUIRED_PERFORMANCE_COMPARATORS - set(manifest["comparators"])
        if missing:
            raise ControllerError(
                "performance candidate lacks required comparators: "
                + ", ".join(sorted(missing))
            )
    measurement_ids: set[tuple[str, str]] = set()
    for check in ordered:
        for point in (check.get("measurement") or {}).get("points", []):
            _validate_measurement_point(repo, point)
            identity = (check["gate"], point["id"])
            if identity in measurement_ids:
                raise ControllerError(
                    f"measurement point {point['id']} is duplicated within gate {check['gate']}"
                )
            measurement_ids.add(identity)
            if point.get("metrics"):
                if not point.get("artifact_path"):
                    raise ControllerError(
                        f"measurement point {point['id']} has metrics without a candidate artifact path"
                    )
                comparator_paths = set(point.get("comparator_paths", {}))
                required = (
                    REQUIRED_PERFORMANCE_COMPARATORS
                    if check["gate"] in PERFORMANCE_GATES
                    else frozenset()
                )
                if not required.issubset(comparator_paths):
                    missing = sorted(required - comparator_paths)
                    raise ControllerError(
                        f"measurement point {point['id']} lacks comparator paths: "
                        + ", ".join(missing)
                    )
                unknown = comparator_paths - set(manifest["comparators"])
                if unknown:
                    raise ControllerError(
                        f"measurement point {point['id']} names undeclared comparators: "
                        + ", ".join(sorted(unknown))
                    )
    return ordered


def _render_prepare_value(value: Any, bindings: dict[str, str]) -> Any:
    if isinstance(value, str):
        for name, replacement in bindings.items():
            value = value.replace("{" + name + "}", replacement)
        return value
    if isinstance(value, list):
        return [_render_prepare_value(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: _render_prepare_value(item, bindings) for key, item in value.items()}
    return value


def prepare_candidate(
    repo_path: Path,
    plan_path: Path,
    output_path: Path,
    message: str,
) -> dict[str, Any]:
    repo = GitRepo(repo_path)
    plan = _read_json(plan_path)
    _validate_json(repo.path, PLAN_SCHEMA, plan)
    development_ref = plan["development_ref"]
    parent = repo.ref(development_ref)
    symbolic_ref = repo.text("symbolic-ref", "-q", "HEAD")
    if symbolic_ref != development_ref:
        raise ControllerError(
            f"active worktree is attached to {symbolic_ref!r}, expected {development_ref!r}"
        )
    if repo.run("diff", "--cached", "--check", check=False).returncode != 0:
        raise ControllerError("staged candidate diff fails whitespace validation")
    if repo.text("diff", "--name-only", "--diff-filter=U"):
        raise ControllerError("candidate index contains unresolved merges")
    staged_paths = sorted(
        path for path in repo.text("diff", "--cached", "--name-only", parent).splitlines() if path
    )
    declared_paths = sorted(plan["affected_scope"]["paths"])
    if staged_paths != declared_paths:
        raise ControllerError(
            "staged source differs from the candidate plan: "
            f"declared={declared_paths}, staged={staged_paths}"
        )
    unstaged_declared = sorted(
        set(repo.text("diff", "--name-only").splitlines()) & set(declared_paths)
    )
    if unstaged_declared:
        raise ControllerError(
            "affected paths contain unstaged changes: " + ", ".join(unstaged_declared)
        )
    tree = repo.text("write-tree")
    if tree == repo.tree(parent):
        raise ControllerError("candidate tree is identical to its accepted parent")
    commit = repo.text("commit-tree", tree, "-p", parent, "-m", message)
    bindings = {
        "boundary": plan["boundary"],
        "candidate": commit,
        "parent": parent,
    }
    artifact_root = _render_prepare_value(plan["artifact_root"], bindings)
    manifest = {
        "schema_version": 1,
        "candidate": {
            "parent": parent,
            "commit": commit,
            "development_ref": development_ref,
        },
        "boundary": plan["boundary"],
        "affected_scope": _render_prepare_value(plan["affected_scope"], bindings),
        "checks": _render_prepare_value(plan["checks"], bindings),
        "comparators": _render_prepare_value(plan["comparators"], bindings),
        "artifact_root": artifact_root,
        "acceptance": plan["acceptance"],
    }
    _validate_manifest_semantics(repo, manifest)
    expected_output_root = (repo.path / artifact_root).resolve()
    rendered_output = Path(_render_prepare_value(str(output_path), bindings))
    resolved_output = (
        rendered_output.resolve()
        if rendered_output.is_absolute()
        else (repo.path / rendered_output).resolve()
    )
    if not _path_is_within(resolved_output, expected_output_root):
        raise ControllerError("candidate manifest must be written inside its artifact root")
    if resolved_output.exists():
        raise ControllerError(f"candidate manifest already exists: {resolved_output}")
    _write_json(resolved_output, manifest)
    return manifest


def _runtime_bindings(repo: GitRepo, worktree: Path, manifest: dict[str, Any]) -> dict[str, str]:
    candidate = manifest["candidate"]
    return {
        "workspace": str(repo.path),
        "worktree": str(worktree),
        "artifact_root": str((repo.path / manifest["artifact_root"]).resolve()),
        "candidate": candidate["commit"],
        "parent": candidate["parent"],
        "boundary": manifest["boundary"],
    }


def _render_runtime_token(value: str, bindings: dict[str, str]) -> str:
    rendered = value
    for name, replacement in bindings.items():
        rendered = rendered.replace("{" + name + "}", replacement)
    if "{" in rendered or "}" in rendered:
        raise ControllerError(f"command token contains an unresolved placeholder: {value!r}")
    return rendered


def _provision_worktree(repo: GitRepo, worktree: Path) -> None:
    venv = repo.path / ".venv"
    if venv.exists():
        (worktree / ".venv").symlink_to(venv, target_is_directory=True)
    source_refs = repo.path / "refs"
    if source_refs.is_dir():
        target_refs = worktree / "refs"
        target_refs.mkdir(parents=True, exist_ok=True)
        for entry in source_refs.iterdir():
            (target_refs / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    source_artifacts = repo.path / "artifacts" / "qualification" / "decode_runtime"
    if source_artifacts.is_dir():
        target_parent = worktree / "artifacts" / "qualification"
        target_parent.mkdir(parents=True, exist_ok=True)
        (target_parent / "decode_runtime").symlink_to(source_artifacts, target_is_directory=True)
    source_references = (
        repo.path / "artifacts" / "qualification" / "generation_runtime" / "reference"
    )
    if source_references.is_dir():
        target_generation = worktree / "artifacts" / "qualification" / "generation_runtime"
        target_generation.mkdir(parents=True, exist_ok=True)
        (target_generation / "reference").symlink_to(
            source_references,
            target_is_directory=True,
        )


def _worktree_is_clean(worktree: Path, candidate: str, tree: str) -> bool:
    repo = GitRepo(worktree)
    return bool(
        repo.commit("HEAD") == candidate
        and repo.tree("HEAD") == tree
        and not repo.text("status", "--porcelain=v1", "--untracked-files=no")
    )


def _validation_report(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    status: str,
    stopped_after: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "boundary": manifest["boundary"],
        "parent": manifest["candidate"]["parent"],
        "candidate": manifest["candidate"]["commit"],
        "status": status,
        "declared_check_count": len(manifest["checks"]),
        "completed_check_count": len(results),
        "stopped_after": stopped_after,
        "checks": results,
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def dry_run(repo_path: Path, manifest_path: Path) -> dict[str, Any]:
    repo = GitRepo(repo_path)
    manifest = _read_json(manifest_path)
    ordered = _validate_manifest_semantics(repo, manifest)
    checks = []
    for check in ordered:
        checks.append(
            {
                "id": check["id"],
                "gate": check["gate"],
                "order": check["order"],
                "command": check["command"],
                "prerequisites": check.get("prerequisites", []),
                "measurement_points": (check.get("measurement") or {}).get("points", []),
            }
        )
    payload = {
        "schema_version": 1,
        "boundary": manifest["boundary"],
        "parent": manifest["candidate"]["parent"],
        "candidate": manifest["candidate"]["commit"],
        "artifact_root": manifest["artifact_root"],
        "checks": checks,
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def run_candidate(
    repo_path: Path,
    manifest_path: Path,
    *,
    accept: bool,
) -> dict[str, Any]:
    repo = GitRepo(repo_path)
    manifest = _read_json(manifest_path)
    ordered = _validate_manifest_semantics(repo, manifest)
    candidate = manifest["candidate"]["commit"]
    parent = manifest["candidate"]["parent"]
    development_ref = manifest["candidate"]["development_ref"]
    if repo.ref(development_ref) != parent:
        raise ControllerError("development ref no longer names the declared accepted parent")
    artifact_root = (repo.path / manifest["artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    canonical_manifest = artifact_root / "candidate.json"
    if canonical_manifest.exists():
        if _read_json(canonical_manifest) != manifest:
            raise ControllerError("candidate artifact root contains a different manifest")
    else:
        _write_json(canonical_manifest, manifest)
    validation_path = artifact_root / "validation.json"
    acceptance_path = artifact_root / "acceptance.json"
    if validation_path.exists() or acceptance_path.exists():
        raise ControllerError("candidate artifact root already contains a terminal result")

    tree = repo.tree(candidate)
    results: list[dict[str, Any]] = []
    stopped_after: str | None = None
    validation_status = "passed"
    worktree = Path(tempfile.mkdtemp(prefix="uniserve-candidate-"))
    added = False
    lock_handle = (repo.common_dir() / "generation-runtime-candidate.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        repo.run("worktree", "add", "--detach", str(worktree), candidate)
        added = True
        _provision_worktree(repo, worktree)
        if not _worktree_is_clean(worktree, candidate, tree):
            raise ControllerError("isolated candidate worktree is not clean and immutable")
        bindings = _runtime_bindings(repo, worktree, manifest)
        completed: set[str] = set()
        for check in ordered:
            prerequisites = set(check.get("prerequisites", []))
            if not prerequisites.issubset(completed):
                raise ControllerError(f"check {check['id']} prerequisites did not pass")
            command = [_render_runtime_token(token, bindings) for token in check["command"]]
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                f"{worktree}:{existing_pythonpath}" if existing_pythonpath else str(worktree)
            )
            environment.update(
                {
                    key: _render_runtime_token(value, bindings)
                    for key, value in check.get("environment", {}).items()
                }
            )
            environment.update(
                {
                    "UNISERVE_CANDIDATE_COMMIT": candidate,
                    "UNISERVE_CANDIDATE_PARENT": parent,
                    "UNISERVE_QUALIFICATION_ROOT": str(artifact_root),
                    "UNISERVE_WORKSPACE": str(repo.path),
                }
            )
            artifact_path = check.get("artifact_path")
            log_path = (
                (repo.path / artifact_path).resolve()
                if artifact_path is not None
                else artifact_root / "checks" / f"{check['id']}.log"
            )
            if not _path_is_within(log_path, artifact_root):
                raise ControllerError(f"check {check['id']} log is outside the artifact root")
            if log_path.exists():
                raise ControllerError(f"check {check['id']} log already exists")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started = _now()
            start_time = time.monotonic()
            timed_out = False
            launch_error: str | None = None
            with log_path.open("w", encoding="utf-8") as log:
                log.write(json.dumps({"command": command, "started_at": started}) + "\n")
                log.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=worktree,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                except OSError as error:
                    returncode = 127
                    launch_error = f"{type(error).__name__}: {error}"
                else:
                    try:
                        returncode = process.wait(timeout=int(check["timeout_s"]))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        process.terminate()
                        try:
                            returncode = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            returncode = process.wait()
                log.write(
                    json.dumps(
                        {
                            "finished_at": _now(),
                            "returncode": returncode,
                            "timed_out": timed_out,
                            **({"launch_error": launch_error} if launch_error else {}),
                        }
                    )
                    + "\n"
                )
            result = {
                "id": check["id"],
                "gate": check["gate"],
                "status": "passed" if returncode == 0 and not timed_out else "failed",
                "returncode": returncode,
                "timed_out": timed_out,
                **({"launch_error": launch_error} if launch_error else {}),
                "elapsed_s": time.monotonic() - start_time,
                "artifact_path": str(log_path.relative_to(repo.path)),
                "artifact_sha256": _file_sha256(log_path),
            }
            results.append(result)
            if not _worktree_is_clean(worktree, candidate, tree):
                result["status"] = "failed"
                result["source_mutated"] = True
            if result["status"] != "passed":
                validation_status = "failed"
                stopped_after = check["id"]
                break
            completed.add(check["id"])
    finally:
        if added:
            repo.run("worktree", "remove", "--force", str(worktree), check=False)
        else:
            shutil.rmtree(worktree, ignore_errors=True)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    validation = _validation_report(
        manifest,
        results,
        status=validation_status,
        stopped_after=stopped_after,
    )
    _write_json(validation_path, validation)
    if validation_status != "passed":
        return validation
    if not accept:
        return validation
    if manifest["acceptance"]["update_ref"] is not True:
        raise ControllerError("manifest does not authorize development-ref acceptance")
    if repo.ref(development_ref) != parent:
        raise ControllerError("development ref changed after candidate validation")
    if repo.text("write-tree") != tree:
        raise ControllerError("active index no longer contains the validated candidate tree")
    if repo.text("symbolic-ref", "-q", "HEAD") != development_ref:
        raise ControllerError("active worktree is not attached to the development ref")
    repo.run("update-ref", development_ref, candidate, parent)
    acceptance_payload = {
        "schema_version": 1,
        "boundary": manifest["boundary"],
        "status": "accepted",
        "parent": parent,
        "candidate": candidate,
        "candidate_tree": tree,
        "development_ref": development_ref,
        "fast_forward": True,
        "candidate_manifest_sha256": _file_sha256(canonical_manifest),
        "validation_sha256": _file_sha256(validation_path),
    }
    acceptance = {
        **acceptance_payload,
        "fingerprint": _canonical_sha256(acceptance_payload),
    }
    _write_json(acceptance_path, acceptance)
    return acceptance


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--message", required=True)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("manifest", type=Path)

    execute = subparsers.add_parser("run")
    execute.add_argument("manifest", type=Path)
    execute.add_argument("--accept", action="store_true")

    args = parser.parse_args(argv)
    repo_path = args.repo.resolve()
    try:
        if args.command == "prepare":
            value = prepare_candidate(
                repo_path,
                args.plan.resolve(),
                args.out,
                args.message,
            )
        elif args.command == "inspect":
            value = dry_run(repo_path, args.manifest.resolve())
        else:
            value = run_candidate(repo_path, args.manifest.resolve(), accept=args.accept)
    except ControllerError as error:
        print(f"candidate control failed: {error}", file=sys.stderr)
        return 1
    _print_json(value)
    if args.command == "run" and value.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
