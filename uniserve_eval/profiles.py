"""Profile loading and resolution for the serving evaluation driver.

A profiles file (default ``uniserve_eval/profiles.json``) declares three
sections:

  servers   — how to launch one serving backend (UniServe ``serve_args`` or an
              explicit ``command``); specs support ``extends`` with deep-merge
              so variants stay single-source.
  workloads — what to run against a compatible server (``verify`` correctness
              gates, ``perf`` harness points, or generic ``script`` runs).
  suites    — ordered workload names for a broader pass; either a plain list or
              ``{"workloads": [...], "compare": [[baseline, candidate, ...]]}``.

Everything path-like resolves against the repository root; artifacts land under
the config's ``artifact_root`` (``servers/<name>`` and ``workloads/<name>``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "profiles.json"
ENV_REF_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_profile_value(value: Any) -> Any:
    """Expand environment-variable references in profile values.

    Missing variables intentionally remain as ``${NAME}`` so dry-run/audit
    commands can show the required environment without baking in local paths.
    Launch paths call :func:`require_resolved_profile_value` before exec.
    """

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_profile_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_profile_value(item) for key, item in value.items()}
    return value


def unresolved_env_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        for match in ENV_REF_RE.finditer(value):
            refs.add(match.group("braced") or match.group("plain") or "")
    elif isinstance(value, list):
        for item in value:
            refs.update(unresolved_env_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(unresolved_env_refs(item))
    refs.discard("")
    return refs


def require_resolved_profile_value(value: Any, *, context: str) -> None:
    refs = sorted(unresolved_env_refs(value))
    if refs:
        names = ", ".join(refs)
        raise SystemExit(f"{context} has unresolved environment variable(s): {names}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def server_spec(config: dict[str, Any], name: str, seen: set[str] | None = None) -> dict[str, Any]:
    servers = config.get("servers", {})
    if name not in servers:
        known = ", ".join(sorted(servers))
        raise SystemExit(f"unknown server {name!r}; known servers: {known}")
    raw = dict(servers[name])
    parent = raw.get("extends")
    if not parent:
        return raw
    seen = seen or set()
    if name in seen:
        raise SystemExit(f"server inheritance cycle at {name!r}")
    base = server_spec(config, str(parent), seen | {name})
    return _deep_merge(base, raw)


def server_profile_definition_contract(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Bind a named server profile to its inherited, unexpanded definition."""
    definition = server_spec(config, name)
    launcher_inputs = (
        {
            "python": config.get("python", ".venv/bin/python"),
            "server_bin": config.get("server_bin", "target/debug/uniserve"),
        }
        if not definition.get("command")
        else {}
    )
    payload = {
        "schema_version": 1,
        "profile": name,
        "inherited_definition_sha256": _canonical_sha256(definition),
        "launcher_inputs_sha256": _canonical_sha256(launcher_inputs),
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def server_profile_definition_matches(
    value: Any,
    config: dict[str, Any],
) -> bool:
    """Return whether a retained profile contract matches the active config."""
    if not (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "profile",
            "inherited_definition_sha256",
            "launcher_inputs_sha256",
            "fingerprint",
        }
        and value.get("schema_version") == 1
        and isinstance(value.get("profile"), str)
        and value.get("profile", "").startswith("benchmark/server/")
    ):
        return False
    try:
        expected = server_profile_definition_contract(config, value["profile"])
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    return value == expected


def server_command_template(config: dict[str, Any], name: str) -> list[str]:
    """Return the unexpanded command tokens declared by one server profile.

    Environment references remain template variables. Relative launcher paths
    follow the same repository-root resolution used by the serving launcher.
    Formal benchmark profiles always declare a NUMA node, so the template also
    binds the wrapper which is required by the formal runner.
    """

    definition = server_spec(config, name)
    command: list[str] = []
    numa_node = definition.get("numa_node", 0)
    if numa_node is not None:
        command.extend(
            [
                "numactl",
                f"--cpunodebind={int(numa_node)}",
                f"--membind={int(numa_node)}",
            ]
        )
    explicit = definition.get("command")
    if explicit:
        command.extend(str(part) for part in explicit)
        return command

    def launcher_path(value: Any) -> str:
        text = str(value)
        if unresolved_env_refs(text) or Path(text).is_absolute():
            return text
        return str(ROOT / text)

    command.extend(
        [
            launcher_path(config.get("server_bin", "target/debug/uniserve")),
            "serve",
            "--model-path",
            str(definition["model"]),
            "--served-model-name",
            str(definition["served_model_name"]),
            "--host",
            str(definition.get("host", "127.0.0.1")),
            "--port",
            str(definition["port"]),
            "--worker-python",
            launcher_path(config.get("python", ".venv/bin/python")),
        ]
    )
    command.extend(str(part) for part in definition.get("serve_args", []))
    return command


def _command_template_bindings(
    template: list[str] | tuple[str, ...],
    command: Any,
) -> dict[str, str] | None:
    if not isinstance(command, list) or len(command) != len(template):
        return None
    bindings: dict[str, str] = {}
    for template_token, actual_token in zip(template, command, strict=True):
        if not isinstance(actual_token, str):
            return None
        matches = list(ENV_REF_RE.finditer(str(template_token)))
        if not matches:
            if actual_token != str(template_token):
                return None
            continue
        pattern_parts: list[str] = []
        captured_variables: list[str] = []
        offset = 0
        for match in matches:
            pattern_parts.append(re.escape(str(template_token)[offset : match.start()]))
            variable = match.group("braced") or match.group("plain") or ""
            if not variable:
                return None
            if variable in bindings:
                pattern_parts.append(re.escape(bindings[variable]))
            else:
                pattern_parts.append("(.+?)")
                captured_variables.append(variable)
            offset = match.end()
        pattern_parts.append(re.escape(str(template_token)[offset:]))
        matched = re.fullmatch("".join(pattern_parts), actual_token)
        if matched is None:
            return None
        for variable, captured in zip(captured_variables, matched.groups(), strict=True):
            if not isinstance(captured, str) or not captured:
                return None
            if variable in bindings and bindings[variable] != captured:
                return None
            bindings.setdefault(variable, captured)
    return bindings


def command_template_matches(
    template: list[str] | tuple[str, ...],
    command: Any,
) -> bool:
    """Return whether an executed command is an instance of a profile template."""

    return _command_template_bindings(template, command) is not None


def _required_model_revision(definition: dict[str, Any]) -> dict[str, str] | None:
    value = definition.get("required_model_revision")
    if value is None:
        return None
    if not (
        isinstance(value, dict)
        and set(value) == {"repository", "revision"}
        and isinstance(value.get("repository"), str)
        and bool(value.get("repository"))
        and isinstance(value.get("revision"), str)
        and re.fullmatch(r"[0-9a-f]{40}", value["revision"])
    ):
        raise ValueError("server profile has an invalid required model revision")
    return {"repository": value["repository"], "revision": value["revision"]}


def _required_model_content(definition: dict[str, Any]) -> dict[str, Any] | None:
    value = definition.get("required_model_content")
    if value is None:
        return None
    if not (
        isinstance(value, dict)
        and set(value) == {"kind", "file_count", "total_size_bytes", "tree_sha256"}
        and value.get("kind") == "directory"
        and isinstance(value.get("file_count"), int)
        and not isinstance(value.get("file_count"), bool)
        and value.get("file_count", 0) > 0
        and isinstance(value.get("total_size_bytes"), int)
        and not isinstance(value.get("total_size_bytes"), bool)
        and value.get("total_size_bytes", 0) > 0
        and isinstance(value.get("tree_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["tree_sha256"])
    ):
        raise ValueError("server profile has an invalid required model content contract")
    return dict(value)


def _model_content_contract_matches(value: Any, requirement: dict[str, Any] | None) -> bool:
    if requirement is None:
        return isinstance(value, dict) and bool(value)
    return bool(
        isinstance(value, dict)
        and all(value.get(key) == expected for key, expected in requirement.items())
    )


def _profile_performance_environment_matches(
    definition: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    """Bind every explicitly configured safe performance override to execution."""

    raw_environment = definition.get("env") or {}
    actual = execution.get("performance_environment")
    if (
        not isinstance(raw_environment, dict)
        or unresolved_env_refs(raw_environment)
        or not isinstance(actual, dict)
    ):
        return False
    from .harness.provenance import performance_environment

    expected = performance_environment(
        {str(key): str(value) for key, value in raw_environment.items()}
    )
    return all(actual.get(key) == value for key, value in expected.items())


def _active_environment_overrides(
    definition: dict[str, Any],
    cuda_selector: str,
) -> dict[str, str] | None:
    raw_environment = definition.get("env") or {}
    if not isinstance(raw_environment, dict) or unresolved_env_refs(raw_environment):
        return None
    overrides = {str(key): str(value) for key, value in raw_environment.items()}
    overrides["CUDA_VISIBLE_DEVICES"] = cuda_selector
    return overrides


def _model_revision_contract_matches(
    value: Any,
    requirement: dict[str, str] | None,
    model_contract: Any,
) -> bool:
    if requirement is None:
        return value is None
    if not (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "repository",
            "revision",
            "proof",
            "model_contract_sha256",
            "fingerprint",
        }
        and value.get("schema_version") == 1
        and value.get("repository") == requirement["repository"]
        and value.get("revision") == requirement["revision"]
        and isinstance(model_contract, dict)
        and value.get("model_contract_sha256") == _canonical_sha256(model_contract)
        and _fingerprint_valid(value)
    ):
        return False
    proof = value.get("proof")
    if not isinstance(proof, dict):
        return False
    if proof.get("kind") == "git_checkout":
        return bool(
            set(proof) == {"kind", "head", "remote_repository"}
            and proof.get("head") == requirement["revision"]
            and proof.get("remote_repository") == requirement["repository"]
        )
    return bool(
        proof.get("kind") == "huggingface_cache_snapshot"
        and set(proof)
        == {"kind", "repository_cache_name", "snapshot_revision", "symlink_file_count"}
        and proof.get("repository_cache_name")
        == "models--" + requirement["repository"].replace("/", "--")
        and proof.get("snapshot_revision") == requirement["revision"]
        and isinstance(proof.get("symlink_file_count"), int)
        and not isinstance(proof.get("symlink_file_count"), bool)
        and proof.get("symlink_file_count", 0) > 0
    )


def _source_revision_matches(execution: dict[str, Any], revision: str, role: str) -> bool:
    for source in execution.get("source_revisions", []):
        if not isinstance(source, dict) or role not in source.get("roles", []):
            continue
        state = source.get("state")
        if isinstance(state, dict) and state.get("head") == revision:
            return True
    return False


def server_execution_matches_profile(
    config: dict[str, Any],
    profile: str,
    execution: Any,
    *,
    model_contract: Any,
    model_revision_contract: Any,
) -> bool:
    """Validate that retained execution provenance derives from an active profile."""

    if not isinstance(execution, dict) or not _fingerprint_valid(execution):
        return False
    try:
        definition = server_spec(config, profile)
        template = server_command_template(config, profile)
        requirement = _required_model_revision(definition)
        content_requirement = _required_model_content(definition)
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    command = execution.get("command")
    bindings = _command_template_bindings(template, command)
    if bindings is None or not isinstance(command, list) or not isinstance(model_contract, dict):
        return False

    model_template = str(definition.get("model", ""))
    model_indexes = [index for index, token in enumerate(template) if token == model_template]
    if not model_indexes:
        return False
    model_index = model_indexes[0]
    previous = command[model_index - 1] if model_index > 0 else ""
    model_role = (
        f"option:{previous}"
        if isinstance(previous, str) and previous.startswith("--")
        else f"command_argument:{model_index}"
    )
    expected_input = {"role": model_role, **model_contract}
    if expected_input not in execution.get("command_inputs", []):
        return False
    if not _model_content_contract_matches(model_contract, content_requirement):
        return False
    if not _profile_performance_environment_matches(definition, execution):
        return False
    if not _model_revision_contract_matches(model_revision_contract, requirement, model_contract):
        return False

    required_revision = definition.get("required_source_revision")
    required_role = definition.get("required_source_role")
    if required_revision is None or required_role is None:
        return required_revision is None and required_role is None
    return bool(
        isinstance(required_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", required_revision)
        and isinstance(required_role, str)
        and required_role
        and _source_revision_matches(execution, required_revision, required_role)
    )


def benchmark_comparison_profiles(
    config: dict[str, Any],
    benchmark_profile: str,
    parity_group: str,
) -> dict[str, str] | None:
    """Return the candidate/reference profiles declared by comparison points."""

    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return None
    benchmark = benchmarks.get(benchmark_profile)
    if not isinstance(benchmark, dict):
        return None
    groups = benchmark.get("groups")
    points = benchmark.get("points")
    if not isinstance(groups, dict) or not isinstance(points, dict):
        return None
    roles: dict[str, str] = {}
    for group in groups.values():
        if not isinstance(group, dict) or not isinstance(group.get("server"), str):
            return None
        for point_name in group.get("points", []):
            point = points.get(point_name)
            if not isinstance(point, dict) or point.get("parity_group") != parity_group:
                continue
            role = point.get("comparison_role")
            if role not in {"candidate", "reference"}:
                return None
            previous = roles.get(role)
            if previous is not None and previous != group["server"]:
                return None
            roles[role] = group["server"]
    if set(roles) != {"candidate", "reference"} or roles["candidate"] == roles["reference"]:
        return None
    return roles


_BENCHMARK_PARITY_IDENTITY_FIELDS = frozenset(
    {"dataset_path", "model", "name", "runtime_profile_id", "plan_evidence_policy"}
)


def _active_benchmark_parts(
    config: dict[str, Any],
    benchmark_profile: str,
    group_name: str,
    point_name: str,
    load_case_set: str,
    load_case_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any], dict[str, Any]]:
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, dict) or not isinstance(
        benchmark := benchmarks.get(benchmark_profile), dict
    ):
        raise ValueError("unknown benchmark profile")
    groups = benchmark.get("groups")
    points = benchmark.get("points")
    load_cases = benchmark.get("load_cases")
    case_set = load_cases.get(load_case_set) if isinstance(load_cases, dict) else None
    matching_cases = (
        [case for case in case_set if isinstance(case, dict) and case.get("id") == load_case_id]
        if isinstance(case_set, list)
        else []
    )
    if not (
        isinstance(groups, dict)
        and isinstance(group := groups.get(group_name), dict)
        and isinstance(points, dict)
        and isinstance(point := points.get(point_name), dict)
        and point_name in group.get("points", [])
        and isinstance(case_set, list)
        and str(point.get("load_case_set", "")) == load_case_set
        and len(matching_cases) == 1
    ):
        raise ValueError("benchmark point is not declared by the active group and load-case set")
    case = matching_cases[0]
    if set(case) - {"id", "request_rate", "max_concurrency"}:
        raise ValueError("load case has unknown fields")
    request_rate = case.get("request_rate")
    try:
        numeric_rate = float(request_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("load case request_rate must be numeric or 'inf'") from error
    max_concurrency = case.get("max_concurrency")
    if numeric_rate <= 0.0 or (
        max_concurrency is not None
        and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        )
    ):
        raise ValueError("load case has invalid load semantics")
    ids = [item.get("id") for item in case_set if isinstance(item, dict)]
    if len(ids) != len(case_set) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("load-case identifiers must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("load-case identifiers must be unique within a set")
    return benchmark, group, point, case_set, case


def _declared_benchmark_semantics(
    benchmark: dict[str, Any],
    point: dict[str, Any],
    load_case: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    from .harness.report import spec_to_dict
    from .harness.spec import BenchmarkSpec

    defaults = dict(benchmark.get("defaults") or {})
    harness = dict(point.get("harness") or {})
    controlled_fields = {"request_rate", "max_concurrency"}
    if controlled_fields & (set(defaults) | set(harness)):
        raise ValueError("load-controlled fields must be declared by the load case")
    dataset_ref = harness.pop("dataset_ref", None)
    disable_ignore_eos = bool(harness.pop("disable_ignore_eos", False))
    fields = BenchmarkSpec.__dataclass_fields__
    values: dict[str, Any] = {key: value for key, value in defaults.items() if key in fields}
    values.update({key: value for key, value in harness.items() if key in fields})
    values["warmup_requests"] = harness.get("warmup_requests", defaults.get("warmup_requests", 1))
    values["seed"] = harness.get("seed", defaults.get("seed", 42))
    values["ignore_eos"] = not disable_ignore_eos
    values["request_rate"] = float(load_case["request_rate"])
    values["max_concurrency"] = load_case.get("max_concurrency")
    if dataset_ref is not None:
        values["dataset_path"] = "<materialized-dataset>"
    required = {"task", "model", "num_prompts"}
    if not required.issubset(values):
        raise ValueError("benchmark point has an incomplete harness declaration")
    unknown = set(harness) - set(fields)
    if unknown:
        raise ValueError(f"benchmark point has unknown harness fields: {sorted(unknown)}")

    expanded = expand_profile_value(values)
    spec = spec_to_dict(BenchmarkSpec(**expanded))
    normalized = {
        key: value for key, value in spec.items() if key not in _BENCHMARK_PARITY_IDENTITY_FIELDS
    }
    if normalized.get("denoise_updates") is not None:
        normalized.pop("steps", None)
    wildcard_fields = sorted(key for key, value in normalized.items() if unresolved_env_refs(value))
    for key in wildcard_fields:
        normalized.pop(key)
    return normalized, wildcard_fields


def benchmark_matrix_definition_contract(
    config: dict[str, Any],
    benchmark_profile: str,
    *,
    group_name: str,
    point_name: str,
    load_case_set: str,
    load_case_id: str,
) -> dict[str, Any]:
    """Bind one generated matrix point to its canonical profile declarations."""

    benchmark, group, point, case_set, load_case = _active_benchmark_parts(
        config,
        benchmark_profile,
        group_name,
        point_name,
        load_case_set,
        load_case_id,
    )
    semantics, wildcard_fields = _declared_benchmark_semantics(
        benchmark,
        point,
        load_case,
    )
    payload = {
        "schema_version": 1,
        "benchmark_profile": benchmark_profile,
        "group": group_name,
        "point": point_name,
        "load_case_set": load_case_set,
        "load_case_id": load_case_id,
        "group_definition_sha256": _canonical_sha256(group),
        "point_definition_sha256": _canonical_sha256(point),
        "load_case_definition_sha256": _canonical_sha256(load_case),
        "load_case_set_definition_sha256": _canonical_sha256(case_set),
        "hardware_requirements_sha256": _canonical_sha256(benchmark.get("hardware_requirements")),
        "datasets_definition_sha256": _canonical_sha256(benchmark.get("datasets", {})),
        "declared_semantics_sha256": _canonical_sha256(semantics),
        "environment_wildcard_fields": wildcard_fields,
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def _active_matrix_hardware_matches(
    matrix: dict[str, Any],
    benchmark: dict[str, Any],
) -> bool:
    requirements = benchmark.get("hardware_requirements")
    hardware = matrix.get("hardware")
    if not (
        isinstance(requirements, dict)
        and requirements.get("gpu_count_per_process") == 1
        and isinstance(requirements.get("gpu_model"), str)
        and isinstance(hardware, dict)
        and _fingerprint_valid(hardware)
        and isinstance(selected := hardware.get("selected_accelerator"), dict)
        and _fingerprint_valid(selected)
        and isinstance(selector := selected.get("selector"), str)
        and selector
        and "," not in selector
        and isinstance(gpu := selected.get("gpu"), dict)
    ):
        return False
    for execution_name in ("server_execution", "harness_execution"):
        execution = matrix.get(execution_name)
        if not (
            isinstance(execution, dict)
            and isinstance(environment := execution.get("environment"), dict)
            and _fingerprint_valid(environment)
            and isinstance(performance := execution.get("performance_environment"), dict)
            and performance.get("CUDA_VISIBLE_DEVICES") == selector
        ):
            return False
    return gpu.get("name") == requirements["gpu_model"]


def benchmark_matrix_definition_matches(
    matrix: Any,
    config: dict[str, Any],
) -> bool:
    """Validate matrix identity, semantics, and hardware against active config."""

    if not isinstance(matrix, dict) or not isinstance(
        value := matrix.get("benchmark_definition"), dict
    ):
        return False
    expected_keys = {
        "schema_version",
        "benchmark_profile",
        "group",
        "point",
        "load_case_set",
        "load_case_id",
        "group_definition_sha256",
        "point_definition_sha256",
        "load_case_definition_sha256",
        "load_case_set_definition_sha256",
        "hardware_requirements_sha256",
        "datasets_definition_sha256",
        "declared_semantics_sha256",
        "environment_wildcard_fields",
        "fingerprint",
    }
    if not (
        set(value) == expected_keys
        and value.get("schema_version") == 1
        and _fingerprint_valid(value)
        and all(
            isinstance(value.get(key), str) and bool(value.get(key))
            for key in ("benchmark_profile", "group", "point", "load_case_set", "load_case_id")
        )
    ):
        return False
    try:
        expected = benchmark_matrix_definition_contract(
            config,
            value["benchmark_profile"],
            group_name=value["group"],
            point_name=value["point"],
            load_case_set=value["load_case_set"],
            load_case_id=value["load_case_id"],
        )
        benchmark, group, point, _case_set, load_case = _active_benchmark_parts(
            config,
            value["benchmark_profile"],
            value["group"],
            value["point"],
            value["load_case_set"],
            value["load_case_id"],
        )
        semantics, wildcard_fields = _declared_benchmark_semantics(
            benchmark,
            point,
            load_case,
        )
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    if value != expected or not _active_matrix_hardware_matches(matrix, benchmark):
        return False
    context = {"load_id": value["load_case_id"]}
    if not (
        matrix.get("benchmark_profile") == value["benchmark_profile"]
        and matrix.get("benchmark") == str(point.get("name", "")).format(**context)
        and matrix.get("server_profile") == group.get("server")
        and matrix.get("parity_group") == point.get("parity_group")
        and matrix.get("comparison_role") == point.get("comparison_role")
    ):
        return False
    parity = matrix.get("parity_contract")
    harness = parity.get("harness") if isinstance(parity, dict) else None
    actual_spec = harness.get("spec") if isinstance(harness, dict) else None
    selected_rows = harness.get("selected_rows") if isinstance(harness, dict) else None
    if not (
        isinstance(actual_spec, dict)
        and set(actual_spec) - set(wildcard_fields) == set(semantics)
        and all(actual_spec.get(key) == item for key, item in semantics.items())
        and isinstance(selected_rows, dict)
        and selected_rows.get("count") == semantics.get("num_prompts")
    ):
        return False

    execution = matrix.get("server_execution")
    try:
        bindings = _command_template_bindings(
            server_command_template(config, str(group["server"])),
            execution.get("command") if isinstance(execution, dict) else None,
        )
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    raw_harness = dict(point.get("harness") or {})
    if bindings is None:
        return False
    for field in wildcard_fields:
        raw = raw_harness.get(field)
        refs = unresolved_env_refs(raw)
        if not isinstance(raw, str) or len(refs) != 1:
            return False
        variable = next(iter(refs))
        if raw not in {f"${variable}", f"${{{variable}}}"}:
            return False
        if bindings.get(variable) != actual_spec.get(field):
            return False

    dataset_ref = raw_harness.get("dataset_ref")
    if dataset_ref is not None:
        datasets = benchmark.get("datasets")
        dataset = datasets.get(dataset_ref) if isinstance(datasets, dict) else None
        if not (
            isinstance(dataset, dict)
            and dataset.get("revision") == actual_spec.get("dataset_revision")
        ):
            return False
    return True


def benchmark_matrix_profile_binding_matches(
    matrix: Any,
    config: dict[str, Any],
) -> bool:
    """Validate a matrix point against the active profile and role mapping."""
    if not isinstance(matrix, dict):
        return False
    benchmark_profile = matrix.get("benchmark_profile")
    parity_group = matrix.get("parity_group")
    comparison_role = matrix.get("comparison_role")
    profile = matrix.get("server_profile")
    profile_contract = matrix.get("server_profile_contract")
    binding = matrix.get("server_profile_binding")
    server_execution = matrix.get("server_execution")
    model_revision_contract = matrix.get("model_revision_contract")
    parity_contract = matrix.get("parity_contract")
    if not (
        benchmark_matrix_definition_matches(matrix, config)
        and isinstance(benchmark_profile, str)
        and benchmark_profile
        and isinstance(parity_group, str)
        and parity_group
        and comparison_role in {"candidate", "reference"}
        and isinstance(profile, str)
        and profile.startswith("benchmark/server/")
        and isinstance(profile_contract, dict)
        and profile_contract.get("profile") == profile
        and server_profile_definition_matches(profile_contract, config)
        and isinstance(binding, dict)
        and set(binding)
        == {
            "schema_version",
            "profile_definition_fingerprint",
            "server_execution_fingerprint",
            "baseline_environment_fingerprint",
            "resolved_environment_fingerprint",
            "environment_override_keys",
            "environment_overrides_sha256",
            "command_template_sha256",
            "model_contract_sha256",
            "model_revision_contract_sha256",
            "required_source_revision",
            "required_source_role",
            "fingerprint",
        }
        and binding.get("schema_version") == 3
        and _fingerprint_valid(binding)
        and isinstance(server_execution, dict)
        and _fingerprint_valid(server_execution)
        and isinstance(parity_contract, dict)
    ):
        return False
    active_profiles = benchmark_comparison_profiles(
        config,
        benchmark_profile,
        parity_group,
    )
    if not (
        isinstance(active_profiles, dict)
        and active_profiles.get(comparison_role) == profile
        and parity_contract.get("schema_version") == 2
        and set(parity_contract) == {"schema_version", "harness", "model", "fingerprint"}
        and _fingerprint_valid(parity_contract)
    ):
        return False
    try:
        candidate_definition = server_profile_definition_contract(
            config,
            active_profiles["candidate"],
        )
        reference_definition = server_profile_definition_contract(
            config,
            active_profiles["reference"],
        )
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    if (
        candidate_definition["inherited_definition_sha256"]
        == reference_definition["inherited_definition_sha256"]
        and candidate_definition["launcher_inputs_sha256"]
        == reference_definition["launcher_inputs_sha256"]
    ):
        return False
    model_contract = parity_contract.get("model")
    try:
        active_server = server_spec(config, profile)
        command_template_sha256 = _canonical_sha256(server_command_template(config, profile))
        required_model_revision = _required_model_revision(active_server)
    except (KeyError, TypeError, ValueError, SystemExit):
        return False
    hardware = matrix.get("hardware")
    selected = hardware.get("selected_accelerator") if isinstance(hardware, dict) else None
    selector = selected.get("selector") if isinstance(selected, dict) else None
    harness_execution = matrix.get("harness_execution")
    harness_environment = (
        harness_execution.get("environment") if isinstance(harness_execution, dict) else None
    )
    server_environment = server_execution.get("environment")
    if not (
        isinstance(selector, str)
        and selector
        and isinstance(harness_environment, dict)
        and _fingerprint_valid(harness_environment)
        and isinstance(server_environment, dict)
        and _fingerprint_valid(server_environment)
    ):
        return False
    expected_overrides = _active_environment_overrides(active_server, selector)
    if expected_overrides is None:
        return False
    override_items = sorted(expected_overrides.items())
    return bool(
        binding.get("profile_definition_fingerprint") == profile_contract.get("fingerprint")
        and binding.get("server_execution_fingerprint") == server_execution.get("fingerprint")
        and binding.get("baseline_environment_fingerprint")
        == harness_environment.get("fingerprint")
        and binding.get("resolved_environment_fingerprint") == server_environment.get("fingerprint")
        and binding.get("environment_override_keys") == [key for key, _value in override_items]
        and binding.get("environment_overrides_sha256") == _canonical_sha256(override_items)
        and binding.get("command_template_sha256") == command_template_sha256
        and binding.get("model_contract_sha256")
        == (_canonical_sha256(model_contract) if model_contract is not None else None)
        and binding.get("model_revision_contract_sha256")
        == (
            _canonical_sha256(model_revision_contract)
            if model_revision_contract is not None
            else None
        )
        and _model_revision_contract_matches(
            model_revision_contract,
            required_model_revision,
            model_contract,
        )
        and binding.get("required_source_revision")
        == matrix.get("required_server_source_revision")
        == active_server.get("required_source_revision")
        and binding.get("required_source_role")
        == matrix.get("required_server_source_role")
        == active_server.get("required_source_role")
        and server_execution_matches_profile(
            config,
            profile,
            server_execution,
            model_contract=model_contract,
            model_revision_contract=model_revision_contract,
        )
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_valid(value: dict[str, Any]) -> bool:
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        return False
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    try:
        return fingerprint == _canonical_sha256(payload)
    except (TypeError, ValueError):
        return False


def workload_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    workloads = config.get("workloads", {})
    if name not in workloads:
        known = ", ".join(sorted(workloads))
        raise SystemExit(f"unknown workload {name!r}; known workloads: {known}")
    return dict(workloads[name])


def suite_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a suite to its dict form: ``{"workloads": [...], "compare": [...]}``.

    Plain-list suites remain valid and resolve to ``{"workloads": [...],
    "compare": []}``.
    """
    suites = config.get("suites", {})
    if name not in suites:
        known = ", ".join(sorted(suites))
        raise SystemExit(f"unknown suite {name!r}; known suites: {known}")
    raw = suites[name]
    if isinstance(raw, list):
        return {"workloads": list(raw), "compare": []}
    workloads = list(raw.get("workloads", []))
    compare_groups = [list(group) for group in raw.get("compare", [])]
    return {"workloads": workloads, "compare": compare_groups}


def spec_env(spec: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in dict(expand_profile_value(spec.get("env") or {})).items()
    }


def artifact_root(config: dict[str, Any]) -> Path:
    return ROOT / str(
        expand_profile_value(config.get("artifact_root", "e2e-artifacts/current-verify"))
    )


def server_dir(config: dict[str, Any], name: str) -> Path:
    return artifact_root(config) / "servers" / name


def workload_dir(config: dict[str, Any], name: str) -> Path:
    return artifact_root(config) / "workloads" / name


def server_pid_path(config: dict[str, Any], name: str) -> Path:
    return server_dir(config, name) / "server.pid"


def server_log_path(config: dict[str, Any], name: str) -> Path:
    return server_dir(config, name) / "server.log"


def resolve_server_for_workload(
    config: dict[str, Any], workload: dict[str, Any], override: str | None
) -> tuple[str, dict[str, Any]]:
    server_name = override or workload.get("server")
    if not server_name:
        raise SystemExit("workload has no server; pass --server")
    return str(server_name), server_spec(config, str(server_name))


def workload_env(workload: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in dict(expand_profile_value(workload.get("env") or {})).items()
    }


def merged_env(workload: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(workload_env(workload))
    return env
