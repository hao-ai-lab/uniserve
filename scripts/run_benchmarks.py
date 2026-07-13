#!/usr/bin/env python3
"""Run a profile-declared benchmark matrix.

The runner owns mechanics only: launch one server, run selected harness points
serially, record commands/snapshots, and stop the server. Benchmark content
lives in ``uniserve_eval/profiles.json`` under ``benchmarks``.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This runner and every harness/server child it spawns must resolve packages
# from its own checkout, even when the shared venv's editable uniserve install
# points at a sibling checkout. Re-exec once with PYTHONPATH pinned so the
# import below and all subprocesses agree.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if __name__ == "__main__" and os.environ.get("UNISERVE_BENCH_PYTHONPATH_PINNED") != _REPO_ROOT:
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = f"{_REPO_ROOT}:{existing}" if existing else _REPO_ROOT
    os.environ["UNISERVE_BENCH_PYTHONPATH_PINNED"] = _REPO_ROOT
    os.execv(sys.executable, [sys.executable, *sys.argv])

from uniserve_eval.backends import build_serve_cmd, resolve_cuda_visible_devices  # noqa: E402
from uniserve_eval.harness.cli import spec_from_harness_command  # noqa: E402
from uniserve_eval.harness.comparison import (  # noqa: E402
    compare_pair,
    summarize_runs,
)
from uniserve_eval.harness.comparison import (  # noqa: E402
    render_markdown as render_comparison_markdown,
)
from uniserve_eval.harness.datasets import load_benchmark_inputs  # noqa: E402
from uniserve_eval.harness.provenance import (  # noqa: E402
    effective_environment,
    environment_contract,
    execution_provenance,
    hardware_contract,
    input_path_contract,
    repository_state,
    selected_accelerator_contract,
)
from uniserve_eval.harness.report import (  # noqa: E402
    artifact_bundle_matches,
    attach_execution_contract,
    benchmark_contract,
    benchmark_parity_contract,
    canonical_artifact_bundle_matches,
    canonical_digest,
    write_summary_artifacts,
)
from uniserve_eval.profiles import (  # noqa: E402
    DEFAULT_CONFIG,
    ROOT,
    benchmark_matrix_definition_contract,
    benchmark_matrix_definition_matches,
    command_template_matches,
    expand_profile_value,
    load_config,
    require_resolved_profile_value,
    server_command_template,
    server_execution_matches_profile,
    server_profile_definition_contract,
    server_spec,
    spec_env,
)

PROCESS_RE = re.compile(
    r"uniserve_eval\.harness\.cli|sglang\.launch_server|vllm serve|"
    r"uniserve serve|api_server|openai\.api_server"
)

HARNESS_FLAGS = {
    "dataset": "--dataset",
    "dataset_path": "--dataset-path",
    "dataset_revision": "--dataset-revision",
    "t2i_dataset_revision": "--t2i-dataset-revision",
    "i2t_dataset_revision": "--i2t-dataset-revision",
    "tokenizer": "--tokenizer",
    "endpoint": "--endpoint",
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "min_p": "--min-p",
    "repetition_penalty": "--repetition-penalty",
    "frequency_penalty": "--frequency-penalty",
    "presence_penalty": "--presence-penalty",
    "sampling_seed": "--sampling-seed",
    "chat_template_kwargs": "--chat-template-kwargs",
    "width": "--width",
    "height": "--height",
    "steps": "--steps",
    "denoise_updates": "--denoise-updates",
    "max_images": "--max-images",
    "i2i_mode": "--i2i-mode",
    "wire": "--wire",
    "i2t_question": "--i2t-question",
    "sharegpt_output_len": "--sharegpt-output-len",
    "sharegpt_context_len": "--sharegpt-context-len",
    "guidance_scale": "--guidance-scale",
    "image_guidance_scale": "--image-guidance-scale",
    "cfg_norm": "--cfg-norm",
    "cfg_interval": "--cfg-interval",
    "timestep_shift": "--timestep-shift",
    "image_think": "--image-think",
    "image_t_eps": "--image-t-eps",
    "workload_mix": "--workload-mix",
    "warmup_mix": "--warmup-mix",
    "runtime_profile_id": "--runtime-profile-id",
    "measurement_interface": "--measurement-interface",
    "cache_read_policy": "--cache-read-policy",
    "cache_write_policy": "--cache-write-policy",
    "adapter_selection": "--adapter-selection",
    "structured_output_policy": "--structured-output-policy",
    "output_constraint": "--output-constraint",
    "preprocessing": "--preprocessing",
    "measured_runs": "--measured-runs",
    "server_topology": "--server-topology",
    "plan_evidence_policy": "--plan-evidence-policy",
    "acceptance_min_success": "--acceptance-min-success",
    "acceptance_max_failed": "--acceptance-max-failed",
    "acceptance_min_images_per_success": "--acceptance-min-images-per-success",
}

HARNESS_SWITCHES = {
    "disable_ignore_eos": "--disable-ignore-eos",
}

POINT_SUPPORT_FILES = (
    "command.txt",
    "preflight.txt",
    "postflight.txt",
    "run.json",
    "run.log",
)
SERVER_SUPPORT_FILES = (
    "server_command.txt",
    "server.log",
    "server.exit",
    "pre_server_snapshot.txt",
    "post_server_snapshot.txt",
)


@dataclass(frozen=True)
class ServerRunSpec:
    name: str
    profile: str
    host: str
    port: int
    command: tuple[str, ...]
    env: dict[str, str]
    process_environment: dict[str, str]
    model_contract: dict[str, Any] | None
    baseline_environment: dict[str, str] | None = None
    required_source_revision: str | None = None
    required_source_role: str | None = None
    model_revision_contract: dict[str, Any] | None = None
    profile_contract: dict[str, Any] | None = None
    command_template: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BenchRunSpec:
    name: str
    group: str
    command: tuple[str, ...]
    output_dir: Path
    server_output_dir: Path
    process_environment: dict[str, str]
    harness_contract: dict[str, Any] | None
    parity_group: str | None
    parity_contract: dict[str, Any] | None
    matrix_contract: dict[str, Any] | None


def shell_join(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def numa_binding_prefix(command: tuple[str, ...]) -> tuple[str, ...]:
    cpu = next((part for part in command if part.startswith("--cpunodebind=")), None)
    memory = next((part for part in command if part.startswith("--membind=")), None)
    if cpu is None or memory is None:
        return ()
    return ("numactl", cpu, memory)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = effective_environment() if env is None else dict(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] $ {shell_join(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=full_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rc = proc.wait()
        log.write(f"[{now()}] exit_code={rc}\n")
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {shell_join(command)}")
    return rc


def ps_snapshot() -> str:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if PROCESS_RE.search(line) and Path(__file__).name not in line:
            lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def active_benchmark_processes() -> str:
    """Any benchmark/server process on the host, regardless of checkout.

    Benchmarks share the machine (CPU, memory bandwidth, NVLink fabric) even
    when pinned to different GPUs, so exactly one experiment may run at a
    time host-wide. A sibling checkout's server or harness blocks this runner
    the same as our own.
    """
    return ps_snapshot()


def nvidia_smi(cuda_visible_devices: str | None = None) -> str:
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found"
    command = ["nvidia-smi"]
    if cuda_visible_devices is not None:
        selector = cuda_visible_devices.strip()
        if not selector or "," in selector:
            raise ValueError("clean-GPU inspection requires one physical GPU selector")
        command.append(f"--id={selector}")
    command.extend(
        [
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.strip()


def git_state() -> str:
    chunks = []
    for command in (["git", "rev-parse", "HEAD"], ["git", "status", "--short"]):
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        chunks.append(f"$ {shell_join(command)}\n{proc.stdout.strip()}")
    return "\n\n".join(chunks)


def write_snapshot(path: Path, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"label: {label}",
                f"utc: {now()}",
                "",
                "processes:",
                ps_snapshot(),
                "",
                "nvidia-smi:",
                nvidia_smi(),
                "",
                "git:",
                git_state(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def wait_for_clean_gpu(cuda_visible_devices: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        text = nvidia_smi(cuda_visible_devices)
        used = []
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                try:
                    used.append(int(parts[2]))
                except ValueError:
                    pass
        if used and all(value == 0 for value in used):
            return
        time.sleep(2)
    raise RuntimeError(
        f"GPU {cuda_visible_devices} memory did not return to zero:\n"
        f"{nvidia_smi(cuda_visible_devices)}"
    )


def gpu_numa_nodes() -> dict[str, int]:
    process = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cannot inspect GPU NUMA locality: {process.stdout.strip()}")
    nodes: dict[str, int] = {}
    for line in process.stdout.splitlines():
        columns = line.split()
        if not columns or not re.fullmatch(r"GPU\d+", columns[0]) or len(columns) < 3:
            continue
        try:
            nodes[columns[0][3:]] = int(columns[-2])
        except ValueError:
            continue
    if not nodes:
        raise RuntimeError("nvidia-smi topology did not report GPU NUMA affinity")
    return nodes


def validate_numa_binding(servers: dict[str, ServerRunSpec]) -> None:
    nodes = gpu_numa_nodes()
    for group_name, server in servers.items():
        visible = server.env.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0]
        if visible not in nodes:
            raise SystemExit(
                f"formal server {group_name} has an unresolved physical CUDA device {visible!r}"
            )
        node = nodes[visible]
        command = set(server.command)
        required = {f"--cpunodebind={node}", f"--membind={node}"}
        if not required.issubset(command):
            raise SystemExit(
                f"formal server {group_name} is not bound to GPU {visible}'s NUMA node {node}"
            )


def validate_hardware_requirements(
    servers: dict[str, ServerRunSpec],
    groups: dict[str, list[BenchRunSpec]],
    benchmark: dict[str, Any],
) -> None:
    requirements = benchmark.get("hardware_requirements")
    if not isinstance(requirements, dict) or set(requirements) != {
        "gpu_count_per_process",
        "gpu_model",
    }:
        raise SystemExit("formal benchmark has no exact hardware requirements")
    if requirements.get("gpu_count_per_process") != 1:
        raise SystemExit("formal benchmark must require exactly one GPU per process")
    expected_model = requirements.get("gpu_model")
    if not isinstance(expected_model, str) or not expected_model:
        raise SystemExit("formal benchmark has no GPU model requirement")
    for group_name, server in servers.items():
        visible = server.process_environment.get("CUDA_VISIBLE_DEVICES")
        if not isinstance(visible, str):
            raise SystemExit(f"formal server {group_name} has no explicit visible GPU")
        try:
            selected = selected_accelerator_contract(visible)
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"formal server {group_name}: {error}") from error
        if selected["gpu"].get("name") != expected_model:
            raise SystemExit(
                f"formal server {group_name} selected {selected['gpu'].get('name')!r}; "
                f"protocol requires {expected_model!r}"
            )
        for bench in groups.get(group_name, []):
            harness_visible = bench.process_environment.get("CUDA_VISIBLE_DEVICES")
            if harness_visible != visible:
                raise SystemExit(
                    f"formal harness {bench.name} does not select the paired server GPU"
                )
            try:
                harness_selected = selected_accelerator_contract(harness_visible)
            except (RuntimeError, ValueError) as error:
                raise SystemExit(f"formal harness {bench.name}: {error}") from error
            if harness_selected["gpu"].get("name") != expected_model:
                raise SystemExit(
                    f"formal harness {bench.name} selected "
                    f"{harness_selected['gpu'].get('name')!r}; protocol requires "
                    f"{expected_model!r}"
                )


def paired_harness_environment(server: ServerRunSpec) -> dict[str, str]:
    """Use the paired server's selected accelerator without its backend-only env."""

    environment = (
        dict(server.baseline_environment)
        if server.baseline_environment is not None
        else effective_environment()
    )
    selected = server.process_environment.get("CUDA_VISIBLE_DEVICES")
    if selected is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = selected
    return environment


def wait_for_port(host: str, port: int, proc: subprocess.Popen[str], timeout_s: float) -> None:
    """Wait until the server is inference-ready, not merely listening.

    The Rust HTTP server binds its port while its (tensor-parallel) workers are
    still loading — `/health` returns 503 in that window and generate requests
    500. Poll `/health` for a 200; a non-503 HTTP error (e.g. a backend with no
    `/health`) falls back to the open socket as the readiness signal.
    """
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    health_url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before opening {host}:{port}, rc={proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except OSError as exc:
            last_error = exc
            time.sleep(1)
            continue
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310 - local health probe.
                if resp.status == 200:
                    return
            last_error = RuntimeError("/health returned non-200")
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                return
            last_error = exc
        except OSError as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server did not become ready on {host}:{port}: {last_error}")


def summary_ok(bench: BenchRunSpec) -> bool:
    if bench.harness_contract is None or bench.matrix_contract is None:
        return False
    output_dir = bench.output_dir
    path = output_dir / "summary.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not canonical_artifact_bundle_matches(output_dir, data, bench.harness_contract):
        return False
    try:
        expected_execution_bundle = execution_bundle_contract(bench)
    except OSError:
        return False
    artifact = data["artifact"]
    return bool(
        artifact.get("matrix_contract") == bench.matrix_contract
        and artifact.get("execution_bundle_contract") == expected_execution_bundle
    )


def _file_content_contract(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def execution_bundle_contract(bench: BenchRunSpec) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "benchmark": bench.name,
        "server_group": bench.group,
        "point_files": {
            name: _file_content_contract(bench.output_dir / name) for name in POINT_SUPPORT_FILES
        },
        "server_files": {
            name: _file_content_contract(bench.server_output_dir / name)
            for name in SERVER_SUPPORT_FILES
        },
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def canonicalize_benchmark_point(bench: BenchRunSpec) -> None:
    """Commit one stopped server/harness execution as a matrix point."""
    if bench.harness_contract is None or bench.matrix_contract is None:
        raise RuntimeError(f"benchmark {bench.name} has no resolved artifact contract")
    summary_path = bench.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not artifact_bundle_matches(
        bench.output_dir,
        summary,
        bench.harness_contract,
    ):
        raise RuntimeError(f"benchmark {bench.name} produced a mismatched harness contract")
    attach_execution_contract(
        summary,
        "execution_bundle_contract",
        execution_bundle_contract(bench),
    )
    attach_execution_contract(summary, "matrix_contract", bench.matrix_contract)
    write_summary_artifacts(bench.output_dir, summary)


def materialize_datasets(output_root: Path, benchmark: dict[str, Any]) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for name, spec in dict(benchmark.get("datasets") or {}).items():
        if spec.get("type") != "hf_image_sample":
            raise SystemExit(f"unsupported benchmark dataset type for {name}: {spec.get('type')}")
        out = output_root / str(spec["output_dir"])
        prefix = str(spec.get("prefix", name))
        count = int(spec["count"])
        revision = str(spec["revision"])
        source_identity = {
            "schema_version": 1,
            "repository": str(spec["name"]),
            "revision": revision,
            "split": str(spec.get("split", "train")),
            "selection_seed": int(spec.get("seed", 42)),
            "count": count,
            "encoding": {"format": "JPEG", "quality": 95, "color_mode": "RGB"},
        }
        existing = sorted(out.glob(f"{prefix}_*.jpg"))
        manifest_path = out / "SOURCE.json"
        if len(existing) == count and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = manifest.get("files")
                expected_names = [f"{prefix}_{index:03d}.jpg" for index in range(count)]
                files_match = (
                    isinstance(files, list)
                    and [entry.get("name") for entry in files if isinstance(entry, dict)]
                    == expected_names
                    and all(
                        isinstance(entry, dict)
                        and _file_content_contract(out / str(entry.get("name")))
                        == {
                            "size_bytes": entry.get("size_bytes"),
                            "sha256": entry.get("sha256"),
                        }
                        for entry in files
                    )
                )
            except (OSError, json.JSONDecodeError):
                manifest = {}
                files_match = False
            if manifest.get("source") == source_identity and files_match:
                datasets[name] = out
                continue

        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob(f"{prefix}_*.jpg"):
            old.unlink()

        import random

        from datasets import load_dataset

        ds = load_dataset(
            str(spec["name"]),
            split=str(spec.get("split", "train")),
            revision=revision,
        )
        indices = list(range(len(ds)))
        seed = int(spec.get("seed", 42))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:count]
        files = []
        for n, idx in enumerate(selected_indices):
            image = ds[int(idx)]["image"].convert("RGB")
            image_path = out / f"{prefix}_{n:03d}.jpg"
            image.save(image_path, format="JPEG", quality=95)
            files.append(
                {
                    "name": image_path.name,
                    "source_index": int(idx),
                    **_file_content_contract(image_path),
                }
            )
        manifest_payload = {"source": source_identity, "files": files}
        manifest = {
            **manifest_payload,
            "fingerprint": canonical_digest(manifest_payload),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "SOURCE.txt").unlink(missing_ok=True)
        datasets[name] = out
    return datasets


def harness_command(
    *,
    python: str,
    base_url: str,
    output_dir: Path,
    defaults: dict[str, Any],
    harness: dict[str, Any],
    load_case: dict[str, Any],
    datasets: dict[str, Path],
) -> list[str]:
    harness = expand_profile_value(harness)
    dataset_ref = harness.pop("dataset_ref", None)
    if dataset_ref:
        if str(dataset_ref) not in datasets:
            raise SystemExit(f"unknown dataset_ref {dataset_ref!r}")
        harness["dataset_path"] = str(datasets[str(dataset_ref)])

    task = str(harness.pop("task"))
    model = str(harness.pop("model"))
    num_prompts = int(harness.pop("num_prompts"))
    warmup = int(harness.pop("warmup_requests", defaults.get("warmup_requests", 1)))
    seed = int(harness.pop("seed", defaults.get("seed", 42)))
    controlled_fields = {"request_rate", "max_concurrency"}
    if controlled_fields & (set(defaults) | set(harness)):
        raise SystemExit("load-controlled fields must be declared by the load case")
    request_rate = load_case.get("request_rate")
    max_concurrency = load_case.get("max_concurrency")
    for key in HARNESS_FLAGS:
        if key not in harness and key in defaults:
            harness[key] = defaults[key]

    cmd = [
        python,
        "-m",
        "uniserve_eval.harness.cli",
        "--base-url",
        base_url,
        "--task",
        task,
        "--model",
        model,
        "--num-prompts",
        str(num_prompts),
        "--warmup-requests",
        str(warmup),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--request-rate",
        str(request_rate),
    ]
    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])
    for key, flag in HARNESS_SWITCHES.items():
        if bool(harness.pop(key, False)):
            cmd.append(flag)
    for key, flag in HARNESS_FLAGS.items():
        if key not in harness or harness[key] is None:
            continue
        value = harness.pop(key)
        if key == "cfg_interval" and isinstance(value, list):
            value = ",".join(str(part) for part in value)
        if key in {"chat_template_kwargs", "workload_mix", "warmup_mix"} and isinstance(
            value, dict
        ):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        cmd.extend([flag, str(value)])
    if harness:
        unknown = ", ".join(sorted(harness))
        raise SystemExit(f"unknown harness key(s): {unknown}")
    return cmd


def benchmark_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    benchmarks = config.get("benchmarks", {})
    if name not in benchmarks:
        known = ", ".join(sorted(benchmarks))
        raise SystemExit(f"unknown benchmark {name!r}; known benchmarks: {known}")
    return expand_profile_value(dict(benchmarks[name]))


def server_profile_binding_contract(
    server: ServerRunSpec,
    server_execution: dict[str, Any],
    baseline_environment: dict[str, str],
) -> dict[str, Any]:
    if server.profile_contract is None:
        raise RuntimeError(f"server profile {server.profile!r} has no definition contract")
    if server.command_template is None:
        raise RuntimeError(f"server profile {server.profile!r} has no command template")
    execution_fingerprint = server_execution.get("fingerprint")
    if not isinstance(execution_fingerprint, str):
        raise RuntimeError(f"server profile {server.profile!r} has no execution fingerprint")
    baseline_contract = environment_contract(baseline_environment)
    resolved_environment = dict(baseline_environment)
    resolved_environment.update(server.env)
    if resolved_environment != server.process_environment:
        mismatched_names = sorted(
            name
            for name in set(resolved_environment) | set(server.process_environment)
            if resolved_environment.get(name) != server.process_environment.get(name)
        )
        raise RuntimeError(
            f"server profile {server.profile!r} execution environment does not derive from "
            "the paired harness baseline plus its explicit overrides; mismatched variables: "
            + ", ".join(mismatched_names)
        )
    server_environment = server_execution.get("environment")
    if not isinstance(server_environment, dict):
        raise RuntimeError(f"server profile {server.profile!r} has no environment contract")
    override_items = sorted(server.env.items())
    payload = {
        "schema_version": 3,
        "profile_definition_fingerprint": server.profile_contract["fingerprint"],
        "server_execution_fingerprint": execution_fingerprint,
        "baseline_environment_fingerprint": baseline_contract["fingerprint"],
        "resolved_environment_fingerprint": server_environment.get("fingerprint"),
        "environment_override_keys": [key for key, _value in override_items],
        "environment_overrides_sha256": canonical_digest(override_items),
        "command_template_sha256": canonical_digest(list(server.command_template)),
        "model_contract_sha256": (
            canonical_digest(server.model_contract) if server.model_contract is not None else None
        ),
        "model_revision_contract_sha256": (
            canonical_digest(server.model_revision_contract)
            if server.model_revision_contract is not None
            else None
        ),
        "required_source_revision": server.required_source_revision,
        "required_source_role": server.required_source_role,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def _normalized_huggingface_repository(value: str) -> str | None:
    repository = value.strip().removesuffix(".git").rstrip("/")
    prefixes = (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "git@hf.co:",
        "git@huggingface.co:",
    )
    for prefix in prefixes:
        if repository.startswith(prefix):
            return repository[len(prefix) :]
    return None


def model_revision_contract(
    model_path: str,
    requirement: Any,
    model_contract: dict[str, Any],
) -> dict[str, Any]:
    """Prove a declared model directory is the required immutable revision."""

    if not (
        isinstance(requirement, dict)
        and set(requirement) == {"repository", "revision"}
        and isinstance(requirement.get("repository"), str)
        and requirement["repository"]
        and isinstance(requirement.get("revision"), str)
        and re.fullmatch(r"[0-9a-f]{40}", requirement["revision"])
    ):
        raise SystemExit("server has an invalid required model revision contract")
    repository = requirement["repository"]
    revision = requirement["revision"]
    path = repo_path(model_path).resolve(strict=True)

    proof: dict[str, Any] | None = None
    git = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    lines = git.stdout.splitlines()
    if git.returncode == 0 and len(lines) == 2 and Path(lines[0]).resolve() == path:
        remote = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        remote_repository = _normalized_huggingface_repository(remote.stdout)
        if lines[1] == revision and remote.returncode == 0 and remote_repository == repository:
            proof = {
                "kind": "git_checkout",
                "head": revision,
                "remote_repository": repository,
            }

    cache_name = "models--" + repository.replace("/", "--")
    if proof is None and (
        path.name == revision
        and path.parent.name == "snapshots"
        and path.parent.parent.name == cache_name
    ):
        blobs = path.parent.parent / "blobs"
        if blobs.is_dir():
            entries = [entry for entry in path.rglob("*") if not entry.is_dir()]
            try:
                links = [
                    entry
                    for entry in entries
                    if entry.is_symlink()
                    and entry.resolve(strict=True).is_relative_to(blobs.resolve(strict=True))
                ]
            except OSError:
                links = []
            if entries and len(links) == len(entries):
                proof = {
                    "kind": "huggingface_cache_snapshot",
                    "repository_cache_name": cache_name,
                    "snapshot_revision": revision,
                    "symlink_file_count": len(links),
                }

    if proof is None:
        raise SystemExit(
            f"model path for {repository}@{revision} cannot prove the required revision; "
            "use that exact Git checkout with its Hugging Face origin or its canonical "
            "Hugging Face cache snapshot"
        )
    payload = {
        "schema_version": 1,
        "repository": repository,
        "revision": revision,
        "proof": proof,
        "model_contract_sha256": canonical_digest(model_contract),
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def comparison_profile_roles(benchmark: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return the candidate/reference profile mapping declared by matrix points."""

    point_specs = dict(benchmark.get("points") or {})
    roles: dict[str, dict[str, str]] = {}
    for group_name, group in dict(benchmark.get("groups") or {}).items():
        profile = str(group.get("server"))
        for point_name in list(group.get("points") or []):
            point = dict(point_specs[str(point_name)])
            parity_group = point.get("parity_group")
            comparison_role = point.get("comparison_role")
            if parity_group is None or comparison_role is None:
                continue
            group_roles = roles.setdefault(str(parity_group), {})
            previous = group_roles.get(str(comparison_role))
            if previous is not None and previous != profile:
                raise SystemExit(
                    f"benchmark comparison {parity_group!r} assigns role "
                    f"{comparison_role!r} to multiple server profiles"
                )
            group_roles[str(comparison_role)] = profile
    for parity_group, group_roles in roles.items():
        if set(group_roles) != {"candidate", "reference"}:
            raise SystemExit(
                f"benchmark comparison {parity_group!r} needs candidate and reference profiles"
            )
        if group_roles["candidate"] == group_roles["reference"]:
            raise SystemExit(
                f"benchmark comparison {parity_group!r} must use distinct server profiles"
            )
    return roles


def benchmark_execution_policy_contract(
    *,
    formal: bool,
    selected_groups: list[str],
    output_root_initially_empty: bool,
    build_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "host_lock": True,
        "no_preexisting_benchmark_process": formal,
        "fresh_output_root": formal and output_root_initially_empty,
        "complete_parity_selection": formal,
        "clean_gpu_before_and_after_each_point": formal,
        "clean_source_before_and_after_each_point": formal,
        "pinned_reference_revisions": formal,
        "one_required_gpu_per_process": formal,
        "gpu_local_server_and_harness_numa_binding": formal,
        "fresh_server_per_point": formal,
        "source_build_for_this_root": formal and build_manifest is not None,
    }
    payload = {
        "schema_version": 1,
        "formal": formal,
        "selected_groups": list(selected_groups),
        "checks": checks,
        "build_manifest": build_manifest,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def build_manifest_contract(
    config: dict[str, Any],
    build_log: Path,
) -> dict[str, Any]:
    binary = repo_path(expand_profile_value(config.get("server_bin", "target/release/uniserve")))
    cargo_lock = ROOT / "Cargo.lock"
    if not binary.is_file() or not cargo_lock.is_file() or not build_log.is_file():
        raise RuntimeError("formal build did not produce its required files")

    def version(command: list[str]) -> str:
        process = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0 or not process.stdout.strip():
            raise RuntimeError(f"cannot record build tool version: {shell_join(command)}")
        return process.stdout.strip()

    payload = {
        "schema_version": 1,
        "command": ["cargo", "build", "--release", "--bin", "uniserve"],
        "cargo_version": version(["cargo", "--version"]),
        "rustc_version": version(["rustc", "--version", "--verbose"]),
        "source_state": repository_state(ROOT),
        "cargo_lock": _file_content_contract(cargo_lock),
        "binary": _file_content_contract(binary),
        "build_log": _file_content_contract(build_log),
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def build_servers(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    strict_env: bool,
) -> dict[str, ServerRunSpec]:
    servers: dict[str, ServerRunSpec] = {}
    baseline_environment = effective_environment()
    for group_name, group in dict(benchmark["groups"]).items():
        profile_name = str(group["server"])
        raw = server_spec(config, profile_name)
        spec = expand_profile_value(raw)
        command = tuple(build_serve_cmd(config, raw, strict_env=strict_env))
        env = spec_env(spec)
        profile_value = (
            str(spec["cuda_visible_devices"])
            if spec.get("cuda_visible_devices") is not None
            else None
        )
        cuda_visible_devices = resolve_cuda_visible_devices(profile_value)
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if strict_env:
            require_resolved_profile_value(command, context=f"server {profile_name}")
            require_resolved_profile_value(env, context=f"server env {profile_name}")
        model_contract = None
        resolved_model_revision_contract = None
        if spec.get("model") is not None and strict_env:
            model_contract = input_path_contract(str(spec["model"]), cwd=ROOT)
            if raw.get("required_model_revision") is not None:
                resolved_model_revision_contract = model_revision_contract(
                    str(spec["model"]),
                    raw["required_model_revision"],
                    model_contract,
                )
        required_source_revision = spec.get("required_source_revision")
        required_source_role = spec.get("required_source_role")
        if required_source_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", str(required_source_revision)
        ):
            raise SystemExit(f"server {profile_name} has an invalid required source revision")
        if (required_source_revision is None) != (required_source_role is None):
            raise SystemExit(
                f"server {profile_name} must declare its required source revision and role together"
            )
        if required_source_role is not None and (
            not isinstance(required_source_role, str) or not required_source_role
        ):
            raise SystemExit(f"server {profile_name} has an invalid required source role")
        command_template = tuple(server_command_template(config, profile_name))
        if strict_env and not command_template_matches(command_template, list(command)):
            raise SystemExit(
                f"server {profile_name} launch command does not derive from its active profile"
            )
        servers[group_name] = ServerRunSpec(
            name=group_name,
            profile=profile_name,
            host=str(spec.get("host", "127.0.0.1")),
            port=int(spec["port"]),
            command=command,
            env=env,
            process_environment=effective_environment(env, inherited=baseline_environment),
            model_contract=model_contract,
            baseline_environment=dict(baseline_environment),
            required_source_revision=(
                str(required_source_revision) if required_source_revision is not None else None
            ),
            required_source_role=(
                str(required_source_role) if required_source_role is not None else None
            ),
            model_revision_contract=resolved_model_revision_contract,
            profile_contract=server_profile_definition_contract(config, profile_name),
            command_template=command_template,
        )
    return servers


def build_benches(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    output_root: Path,
    datasets: dict[str, Path],
    servers: dict[str, ServerRunSpec],
    execution_policy: dict[str, Any],
    *,
    strict_env: bool,
    benchmark_profile: str = "main",
) -> dict[str, list[BenchRunSpec]]:
    python = str(repo_path(expand_profile_value(config.get("python", ".venv/bin/python"))))
    defaults = dict(benchmark.get("defaults") or {})
    load_cases = dict(benchmark.get("load_cases") or {})
    point_specs = dict(benchmark.get("points") or {})
    groups: dict[str, list[BenchRunSpec]] = {}
    selected_rows_cache: dict[str, list[dict[str, Any]]] = {}
    server_execution: dict[str, dict[str, Any]] = {}
    declared_parity_groups: dict[str, list[str]] = {}
    declared_parity_roles: dict[str, list[str]] = {}
    for point_name, point in point_specs.items():
        parity_group = point.get("parity_group")
        comparison_role = point.get("comparison_role")
        if parity_group is not None:
            declared_parity_groups.setdefault(str(parity_group), []).append(point_name)
            if comparison_role not in {"candidate", "reference"}:
                raise SystemExit(
                    f"benchmark parity point {point_name!r} needs candidate/reference role"
                )
            declared_parity_roles.setdefault(str(parity_group), []).append(str(comparison_role))
        elif comparison_role is not None:
            raise SystemExit(
                f"non-comparison benchmark point {point_name!r} must not declare a role"
            )
    invalid_groups = {
        group: members for group, members in declared_parity_groups.items() if len(members) < 2
    }
    if invalid_groups:
        raise SystemExit(f"benchmark parity groups need at least two points: {invalid_groups}")
    invalid_roles = {
        group: roles
        for group, roles in declared_parity_roles.items()
        if sorted(roles) != ["candidate", "reference"]
    }
    if invalid_roles:
        raise SystemExit(
            f"benchmark parity groups need exactly one candidate and one reference: {invalid_roles}"
        )
    resolved_parity: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    for group_name, group in dict(benchmark["groups"]).items():
        profile_name = str(group["server"])
        server = expand_profile_value(server_spec(config, profile_name))
        server_run = servers[group_name]
        harness_environment = paired_harness_environment(server_run)
        expected_profile_contract = server_profile_definition_contract(config, profile_name)
        if (
            server_run.profile != profile_name
            or server_run.profile_contract != expected_profile_contract
        ):
            raise SystemExit(
                f"benchmark group {group_name!r} server does not match profile {profile_name!r}"
            )
        base_url = f"http://{server.get('host', '127.0.0.1')}:{server['port']}"
        benches: list[BenchRunSpec] = []
        for point_name in list(group.get("points", [])):
            point = dict(point_specs[str(point_name)])
            parity_group = (
                str(point["parity_group"]) if point.get("parity_group") is not None else None
            )
            comparison_role = (
                str(point["comparison_role"]) if point.get("comparison_role") is not None else None
            )
            load_case_set = str(point.get("load_case_set", ""))
            if load_case_set not in load_cases:
                raise SystemExit(
                    f"point {point_name} references unknown load-case set {load_case_set!r}"
                )
            cases = load_cases[load_case_set]
            if not isinstance(cases, list):
                raise SystemExit(f"load-case set {load_case_set!r} must be a list")
            for load_case in cases:
                if not isinstance(load_case, dict) or not isinstance(load_case.get("id"), str):
                    raise SystemExit(f"load-case set {load_case_set!r} has an invalid case")
                load_id = str(load_case["id"])
                context = {"load_id": load_id}
                name = str(point["name"]).format(**context)
                output_dir = output_root / str(point["output"]).format(**context)
                command = (
                    *numa_binding_prefix(server_run.command),
                    *harness_command(
                        python=python,
                        base_url=base_url,
                        output_dir=output_dir,
                        defaults=defaults,
                        harness=dict(point["harness"]),
                        load_case=load_case,
                        datasets=datasets,
                    ),
                )
                if strict_env:
                    require_resolved_profile_value(command, context=f"benchmark {name}")
                    spec = spec_from_harness_command(command)
                    row_key = canonical_digest(
                        {
                            "task": spec.task.value,
                            "model": spec.model,
                            "dataset": spec.dataset,
                            "dataset_path": spec.dataset_path,
                            "dataset_revision": spec.dataset_revision,
                            "t2i_dataset_revision": spec.t2i_dataset_revision,
                            "i2t_dataset_revision": spec.i2t_dataset_revision,
                            "num_prompts": spec.num_prompts,
                            "workload_mix": spec.workload_mix,
                            "warmup_mix": spec.warmup_mix,
                            "warmup_requests": spec.warmup_requests,
                            "seed": spec.seed,
                            "tokenizer": spec.tokenizer,
                            "sharegpt_context_len": spec.sharegpt_context_len,
                            "sharegpt_output_len": spec.sharegpt_output_len,
                            "i2t_question": spec.i2t_question,
                            "preprocessing": spec.preprocessing,
                        }
                    )
                    if row_key not in selected_rows_cache:
                        selected_rows_cache[row_key] = load_benchmark_inputs(spec).measured
                    harness_contract = benchmark_contract(spec, selected_rows_cache[row_key])
                    harness_parity_contract = benchmark_parity_contract(harness_contract)
                    parity_payload = {
                        "schema_version": 2,
                        "harness": harness_parity_contract,
                        "model": server_run.model_contract,
                    }
                    parity_contract = {
                        **parity_payload,
                        "fingerprint": canonical_digest(parity_payload),
                    }
                    if parity_group is not None:
                        parity_key = (parity_group, load_id)
                        previous = resolved_parity.get(parity_key)
                        if previous is not None and previous[1] != parity_contract:
                            raise SystemExit(
                                f"benchmark parity mismatch for {parity_group!r} at load case "
                                f"{load_id!r}: "
                                f"{previous[0]!r} and {name!r} do not share one protocol/workload contract"
                            )
                        resolved_parity[parity_key] = (name, parity_contract)
                    if group_name not in server_execution:
                        server_execution[group_name] = execution_provenance(
                            server_run.command,
                            server_run.process_environment,
                            cwd=ROOT,
                            workspace_root=ROOT,
                        )
                    selected_accelerator = selected_accelerator_contract(
                        server_run.env.get("CUDA_VISIBLE_DEVICES", "")
                    )
                    hardware_payload = {
                        "schema_version": 2,
                        "host": hardware_contract(),
                        "selected_accelerator": selected_accelerator,
                    }
                    point_hardware = {
                        **hardware_payload,
                        "fingerprint": canonical_digest(hardware_payload),
                    }
                    matrix_payload = {
                        "schema_version": 2,
                        "benchmark": name,
                        "benchmark_profile": benchmark_profile,
                        "benchmark_definition": benchmark_matrix_definition_contract(
                            config,
                            benchmark_profile,
                            group_name=group_name,
                            point_name=str(point_name),
                            load_case_set=load_case_set,
                            load_case_id=load_id,
                        ),
                        "server_profile": server_run.profile,
                        "server_profile_contract": server_run.profile_contract,
                        "server_profile_binding": server_profile_binding_contract(
                            server_run,
                            server_execution[group_name],
                            harness_environment,
                        ),
                        "comparison_role": comparison_role,
                        "required_server_source_revision": (server_run.required_source_revision),
                        "required_server_source_role": server_run.required_source_role,
                        "model_revision_contract": server_run.model_revision_contract,
                        "execution_policy": execution_policy,
                        "server_execution": server_execution[group_name],
                        "harness_execution": execution_provenance(
                            command,
                            harness_environment,
                            cwd=ROOT,
                            workspace_root=ROOT,
                        ),
                        "hardware": point_hardware,
                        "harness_contract_fingerprint": harness_contract["fingerprint"],
                        "parity_group": parity_group,
                        "parity_contract": parity_contract,
                    }
                    matrix_contract = {
                        **matrix_payload,
                        "fingerprint": canonical_digest(matrix_payload),
                    }
                    if not benchmark_matrix_definition_matches(matrix_contract, config):
                        raise SystemExit(
                            f"benchmark {name!r} does not derive from its active matrix definition"
                        )
                    if not server_execution_matches_profile(
                        config,
                        server_run.profile,
                        server_execution[group_name],
                        model_contract=server_run.model_contract,
                        model_revision_contract=server_run.model_revision_contract,
                    ):
                        raise SystemExit(
                            f"benchmark {name!r} server execution does not derive from "
                            f"profile {server_run.profile!r}"
                        )
                else:
                    harness_contract = None
                    parity_contract = None
                    matrix_contract = None
                benches.append(
                    BenchRunSpec(
                        name=name,
                        group=group_name,
                        command=command,
                        output_dir=output_dir,
                        server_output_dir=output_root / "servers" / group_name / name,
                        process_environment=harness_environment,
                        harness_contract=harness_contract,
                        parity_group=parity_group,
                        parity_contract=parity_contract,
                        matrix_contract=matrix_contract,
                    )
                )
        groups[group_name] = benches
    return groups


def write_runbook(
    output_root: Path,
    benchmark_name: str,
    servers: dict[str, ServerRunSpec],
    groups: dict[str, list[BenchRunSpec]],
    benchmark: dict[str, Any],
) -> None:
    lines = [
        "# Benchmark Commands",
        "",
        f"Generated UTC: {now()}",
        f"Benchmark: {benchmark_name}",
        "",
        "Run policy:",
        "- Exactly one server process and one harness process run at a time.",
        "- Every operating point starts from a fresh server process.",
        "- Every point uses the request-arrival or concurrency semantics declared by its load case.",
        "- Server launch commands are emitted from the benchmark profiles.",
        "- Each point binds run.json, command.txt, preflight.txt, postflight.txt, run.log, request records, GPU samples, generated-image samples, and the corresponding server command/log/exit/snapshots.",
        "",
    ]
    if benchmark.get("description"):
        lines.extend(["Description:", "", str(benchmark["description"]), ""])
    for group_name, benches in groups.items():
        server = servers[group_name]
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in server.env.items())
        lines.extend(
            [
                f"## {group_name}",
                "",
                f"Server profile: `{server.profile}`",
                "",
                "Server:",
                "",
                "```bash",
                (env_prefix + " " if env_prefix else "") + shell_join(server.command),
                "```",
                "",
            ]
        )
        for bench in benches:
            lines.extend(
                [f"Harness `{bench.name}`:", "", "```bash", shell_join(bench.command), "```", ""]
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "COMMANDS.md").write_text("\n".join(lines), encoding="utf-8")


def launch_server(
    server: ServerRunSpec, group_dir: Path, timeout_s: float
) -> subprocess.Popen[str]:
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "server_command.txt").write_text(
        shell_join(server.command) + "\n", encoding="utf-8"
    )
    log = (group_dir / "server.log").open("w", encoding="utf-8")
    log.write(f"[{now()}] $ {shell_join(server.command)}\n")
    log.flush()
    proc = subprocess.Popen(
        list(server.command),
        cwd=ROOT,
        env=server.process_environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    (group_dir / "server.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    wait_for_port(server.host, server.port, proc, timeout_s)
    return proc


def stop_server(proc: subprocess.Popen[str] | None, group_dir: Path, grace_s: float) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
    (group_dir / "server.exit").write_text(
        f"exit_code={proc.returncode}\nutc={now()}\n", encoding="utf-8"
    )


def parse_only(value: str | None, all_groups: list[str]) -> list[str]:
    if not value:
        return list(all_groups)
    selected = [part.strip() for part in value.split(",") if part.strip()]
    if len(selected) != len(set(selected)):
        raise SystemExit("--only must not repeat a server group")
    unknown = set(selected) - set(all_groups)
    if unknown:
        raise SystemExit(f"unknown --only group(s): {', '.join(sorted(unknown))}")
    return selected


def parse_name_filter(value: str | None, all_names: set[str], *, flag: str) -> set[str]:
    if not value:
        return set()
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected - all_names
    if unknown:
        raise SystemExit(f"unknown {flag} benchmark(s): {', '.join(sorted(unknown))}")
    return selected


def filter_benchmark_groups(benchmark: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    filtered = dict(benchmark)
    declared = dict(benchmark["groups"])
    filtered["groups"] = {name: declared[name] for name in selected}
    return filtered


def require_clean_source(groups: dict[str, list[BenchRunSpec]]) -> None:
    dirty: list[str] = []
    seen: set[str] = set()
    for benches in groups.values():
        for bench in benches:
            matrix = bench.matrix_contract or {}
            for execution_name in ("server_execution", "harness_execution"):
                execution = matrix.get(execution_name)
                if not isinstance(execution, dict):
                    continue
                for revision in execution.get("source_revisions", []):
                    if not isinstance(revision, dict):
                        continue
                    state = revision.get("state")
                    if not isinstance(state, dict) or state.get("dirty") is not True:
                        continue
                    roles = revision.get("roles")
                    if (
                        isinstance(roles, list)
                        and roles
                        and all(
                            isinstance(role, str) and role.startswith("command_input:")
                            for role in roles
                        )
                    ):
                        continue
                    fingerprint = str(state.get("fingerprint", "unknown"))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    dirty.append(
                        f"{bench.name}:{execution_name}:{','.join(revision.get('roles', []))}"
                    )
    if dirty:
        raise SystemExit(
            "formal benchmark execution requires clean, reconstructable source states: "
            + "; ".join(dirty)
        )


def require_pinned_server_revisions(
    groups: dict[str, list[BenchRunSpec]],
    servers: dict[str, ServerRunSpec],
) -> None:
    mismatches: list[str] = []
    for group_name, benches in groups.items():
        expected = servers[group_name].required_source_revision
        expected_role = servers[group_name].required_source_role
        if expected is None:
            if expected_role is not None:
                mismatches.append(f"{group_name}:missing-revision-for-role:{expected_role}")
            continue
        if expected_role is None:
            mismatches.append(f"{group_name}:{expected}:missing-role")
            continue
        for bench in benches:
            matrix = bench.matrix_contract or {}
            execution = matrix.get("server_execution")
            revisions = execution.get("source_revisions", []) if isinstance(execution, dict) else []
            observed = {
                state.get("head")
                for revision in revisions
                if isinstance(revision, dict)
                and expected_role in revision.get("roles", [])
                and isinstance((state := revision.get("state")), dict)
            }
            if expected not in observed:
                mismatches.append(f"{bench.name}:{expected_role}:{expected}")
    if mismatches:
        raise SystemExit(
            "formal benchmark reference source revision mismatch: " + "; ".join(mismatches)
        )


def require_unchanged_execution_identity(
    bench: BenchRunSpec,
    server: ServerRunSpec,
    *,
    phase: str,
) -> None:
    matrix = bench.matrix_contract
    if not isinstance(matrix, dict):
        raise RuntimeError(f"formal benchmark {bench.name} has no matrix contract")
    actual_server = execution_provenance(
        server.command,
        server.process_environment,
        cwd=ROOT,
        workspace_root=ROOT,
    )
    actual_harness = execution_provenance(
        bench.command,
        bench.process_environment,
        cwd=ROOT,
        workspace_root=ROOT,
    )
    if matrix.get("server_execution") != actual_server:
        raise RuntimeError(
            f"formal benchmark {bench.name} server identity changed {phase} execution"
        )
    if matrix.get("harness_execution") != actual_harness:
        raise RuntimeError(
            f"formal benchmark {bench.name} harness/input identity changed {phase} execution"
        )


def require_complete_parity_selection(
    full_benchmark: dict[str, Any], selected_groups: list[str]
) -> None:
    points = dict(full_benchmark.get("points") or {})
    all_groups = dict(full_benchmark.get("groups") or {})
    declared: dict[str, set[str]] = {}
    selected: dict[str, set[str]] = {}
    for point_name, point in points.items():
        parity_group = point.get("parity_group")
        if parity_group is not None:
            declared.setdefault(str(parity_group), set()).add(str(point_name))
    for group_name in selected_groups:
        for point_name in all_groups[group_name].get("points", []):
            parity_group = points[str(point_name)].get("parity_group")
            if parity_group is not None:
                selected.setdefault(str(parity_group), set()).add(str(point_name))
    incomplete = {
        name: sorted(members) for name, members in selected.items() if members != declared[name]
    }
    if incomplete:
        raise SystemExit(f"formal selection contains incomplete comparison pairs: {incomplete}")


def run_group(
    group_name: str,
    server: ServerRunSpec,
    benches: list[BenchRunSpec],
    *,
    resume: bool,
    server_timeout_s: float,
    server_grace_s: float,
    require_clean_gpu: bool,
    enforce_execution_identity: bool = False,
) -> None:
    for bench in benches:
        if resume and summary_ok(bench):
            print(f"[{now()}] skip complete {bench.name}", flush=True)
            continue
        if enforce_execution_identity:
            require_unchanged_execution_identity(
                bench,
                server,
                phase="before",
            )
        point_server_dir = bench.server_output_dir
        write_snapshot(
            point_server_dir / "pre_server_snapshot.txt",
            label=f"{bench.name} pre-server",
        )
        proc: subprocess.Popen[str] | None = None
        try:
            print(f"[{now()}] launching {group_name} for {bench.name}", flush=True)
            proc = launch_server(server, point_server_dir, server_timeout_s)
            print(f"[{now()}] ready {group_name} port={server.port}", flush=True)
            bench.output_dir.mkdir(parents=True, exist_ok=True)
            (bench.output_dir / "command.txt").write_text(
                shell_join(bench.command) + "\n", encoding="utf-8"
            )
            write_snapshot(bench.output_dir / "preflight.txt", label=f"{bench.name} preflight")
            print(f"[{now()}] run {bench.name}", flush=True)
            run_command(
                list(bench.command),
                log_path=bench.output_dir / "run.log",
                env=bench.process_environment,
            )
            write_snapshot(bench.output_dir / "postflight.txt", label=f"{bench.name} postflight")
        finally:
            print(f"[{now()}] stopping {group_name} after {bench.name}", flush=True)
            stop_server(proc, point_server_dir, server_grace_s)
            write_snapshot(
                point_server_dir / "post_server_snapshot.txt",
                label=f"{bench.name} post-server",
            )
            if require_clean_gpu:
                wait_for_clean_gpu(server.env["CUDA_VISIBLE_DEVICES"])
        if enforce_execution_identity:
            require_unchanged_execution_identity(
                bench,
                server,
                phase="after",
            )
        canonicalize_benchmark_point(bench)
        if not summary_ok(bench):
            raise RuntimeError(f"benchmark did not produce a clean summary: {bench.output_dir}")
        print(f"[{now()}] done {bench.name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark", default="main")
    parser.add_argument("--output-root", help="override benchmark artifact root")
    parser.add_argument(
        "--only", help="comma-separated server groups to run; defaults to the full matrix"
    )
    parser.add_argument(
        "--only-bench", help="comma-separated exact benchmark names to run after group filtering"
    )
    parser.add_argument(
        "--skip-bench", help="comma-separated exact benchmark names to skip after group filtering"
    )
    parser.add_argument("--resume", action="store_true", help="skip complete summary.json points")
    parser.add_argument("--repeat", type=int, default=1, help="number of complete matrix runs")
    parser.add_argument(
        "--text-canary",
        action="store_true",
        help="run the strict redacted text canary for completed text pairs",
    )
    parser.add_argument(
        "--image-smoke",
        action="store_true",
        help="run paired image similarity checks for completed image pairs",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="require a fresh output root, clean source states, clean GPUs, and complete comparison pairs",
    )
    parser.add_argument("--dry-run", action="store_true", help="write COMMANDS.md then exit")
    parser.add_argument(
        "--no-build", action="store_true", help="skip cargo build --release --bin uniserve"
    )
    parser.add_argument(
        "--require-clean-gpu",
        action="store_true",
        help="require all visible GPUs to report zero memory before and after each server group",
    )
    parser.add_argument("--server-timeout-s", type=float, default=1800.0)
    parser.add_argument("--server-grace-s", type=float, default=30.0)
    return parser


def acquire_host_benchmark_lock() -> Any:
    path = Path("/tmp/uniserve-benchmark-runner.lock")
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        holder = handle.read().strip() or "unknown"
        handle.close()
        raise SystemExit(f"another benchmark runner holds the host lock (pid={holder})") from error
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_host_benchmark_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def write_comparisons(
    output_root: Path,
    benches: list[BenchRunSpec],
    *,
    benchmark: str,
    text_canary: bool,
    image_smoke: bool,
) -> dict[str, Any]:
    point_results: list[dict[str, Any]] = []
    paired: dict[tuple[str, str], dict[str, BenchRunSpec]] = {}
    for bench in benches:
        summary_path = bench.output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            artifact = summary.get("artifact") if isinstance(summary.get("artifact"), dict) else {}
            matrix = bench.matrix_contract if isinstance(bench.matrix_contract, dict) else {}
            definition = matrix.get("benchmark_definition")
            metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
            point_results.append(
                {
                    "benchmark": bench.name,
                    "group": bench.group,
                    "task": summary.get("task"),
                    "load_case": (
                        definition.get("load_case_id") if isinstance(definition, dict) else None
                    ),
                    "valid": bool(
                        artifact.get("valid") is True
                        and artifact.get("valid_marker") == "canonical-valid-v2"
                    ),
                    "request_count": summary.get("request_count"),
                    "ok_count": summary.get("ok_count"),
                    "failed_count": summary.get("failed_count"),
                    "elapsed_s": summary.get("elapsed_s"),
                    "metrics": metrics,
                }
            )
        matrix = bench.matrix_contract if isinstance(bench.matrix_contract, dict) else {}
        parity_group = matrix.get("parity_group")
        role = matrix.get("comparison_role")
        definition = matrix.get("benchmark_definition")
        load_case_id = definition.get("load_case_id") if isinstance(definition, dict) else None
        if (
            isinstance(parity_group, str)
            and role in {"candidate", "reference"}
            and isinstance(load_case_id, str)
            and (bench.output_dir / "summary.json").is_file()
        ):
            key = (parity_group, load_case_id)
            if role in paired.setdefault(key, {}):
                raise RuntimeError(
                    f"duplicate {role} point for comparison {parity_group!r} at load case "
                    f"{load_case_id!r}"
                )
            paired[key][role] = bench

    comparisons = []
    for _key, roles in sorted(paired.items()):
        if set(roles) != {"candidate", "reference"}:
            continue
        comparisons.append(
            compare_pair(
                roles["reference"].output_dir,
                roles["candidate"].output_dir,
                text_canary=text_canary,
                image_smoke=image_smoke,
            )
        )
    report = {
        "schema_version": 1,
        "benchmark": benchmark,
        "points": point_results,
        "comparisons": comparisons,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "results.md").write_text(
        render_comparison_markdown(report),
        encoding="utf-8",
    )
    return report


def _run_once(args: argparse.Namespace) -> int:
    if args.formal and args.config.resolve() != DEFAULT_CONFIG.resolve():
        raise SystemExit("--formal requires the canonical uniserve_eval/profiles.json config")
    config = load_config(args.config)
    full_benchmark = benchmark_spec(config, args.benchmark)
    comparison_profile_roles(full_benchmark)
    output_root = repo_path(
        args.output_root or str(full_benchmark.get("artifact_root", "artifacts/benchmark"))
    )
    output_root_initially_empty = not output_root.exists() or not any(output_root.iterdir())
    selected = parse_only(args.only, list(dict(full_benchmark["groups"])))
    if args.formal:
        if args.dry_run or args.resume or args.only_bench or args.skip_bench or args.no_build:
            raise SystemExit(
                "--formal cannot be combined with --dry-run, --resume, --only-bench, "
                "--skip-bench, or --no-build"
            )
        if args.output_root is None:
            raise SystemExit("--formal requires an explicit fresh --output-root")
        if not output_root_initially_empty:
            raise SystemExit(f"formal output root is not empty: {output_root}")
        require_complete_parity_selection(full_benchmark, selected)
        args.require_clean_gpu = True
    benchmark = filter_benchmark_groups(full_benchmark, selected)

    active = active_benchmark_processes()
    if active != "(none)":
        raise SystemExit(f"refusing to start with active benchmark/server processes:\n{active}")
    if not args.no_build and not args.dry_run:
        run_command(
            ["cargo", "build", "--release", "--bin", "uniserve"],
            log_path=output_root / "build.log",
        )
    build_manifest = (
        build_manifest_contract(config, output_root / "build.log")
        if (output_root / "build.log").is_file()
        else None
    )
    execution_policy = benchmark_execution_policy_contract(
        formal=args.formal,
        selected_groups=selected,
        output_root_initially_empty=output_root_initially_empty,
        build_manifest=build_manifest,
    )

    datasets = materialize_datasets(output_root, benchmark) if not args.dry_run else {}
    if args.dry_run:
        for name, spec in dict(benchmark.get("datasets") or {}).items():
            datasets[name] = output_root / str(spec["output_dir"])

    servers = build_servers(config, benchmark, strict_env=not args.dry_run)
    if args.require_clean_gpu and not args.dry_run:
        selectors = {server.env["CUDA_VISIBLE_DEVICES"] for server in servers.values()}
        for selector in sorted(selectors):
            wait_for_clean_gpu(selector)
    groups = build_benches(
        config,
        benchmark,
        output_root,
        datasets,
        servers,
        execution_policy,
        strict_env=not args.dry_run,
        benchmark_profile=args.benchmark,
    )
    if args.formal:
        require_clean_source(groups)
        require_pinned_server_revisions(groups, servers)
        validate_hardware_requirements(servers, groups, benchmark)
        validate_numa_binding(servers)
    write_runbook(output_root, args.benchmark, servers, groups, benchmark)

    all_bench_names = {bench.name for benches in groups.values() for bench in benches}
    only_benches = parse_name_filter(args.only_bench, all_bench_names, flag="--only-bench")
    skipped_benches = parse_name_filter(args.skip_bench, all_bench_names, flag="--skip-bench")

    if args.dry_run:
        print(output_root / "COMMANDS.md")
        return 0

    completed_benches: list[BenchRunSpec] = []
    for group_name in groups:
        if group_name not in selected:
            continue
        benches = [
            bench
            for bench in groups[group_name]
            if (not only_benches or bench.name in only_benches)
            and bench.name not in skipped_benches
        ]
        if not benches:
            print(f"[{now()}] skip {group_name}: no benchmarks selected", flush=True)
            continue
        run_group(
            group_name,
            servers[group_name],
            benches,
            resume=args.resume,
            server_timeout_s=args.server_timeout_s,
            server_grace_s=args.server_grace_s,
            require_clean_gpu=args.require_clean_gpu,
            enforce_execution_identity=args.formal,
        )
        completed_benches.extend(benches)

    write_comparisons(
        output_root,
        completed_benches,
        benchmark=args.benchmark,
        text_canary=args.text_canary,
        image_smoke=args.image_smoke,
    )
    write_snapshot(output_root / "final_snapshot.txt", label="final")
    print(f"[{now()}] complete selected groups: {', '.join(selected)}", flush=True)
    return 0


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least one")
    if args.repeat == 1:
        return _run_once(args)

    config = load_config(args.config)
    benchmark = benchmark_spec(config, args.benchmark)
    output_root = repo_path(
        args.output_root or str(benchmark.get("artifact_root", "artifacts/benchmark"))
    )
    run_reports: list[dict[str, Any]] = []
    for index in range(args.repeat):
        child = copy.copy(args)
        child.repeat = 1
        child.output_root = str(output_root / f"run-{index + 1:03d}")
        status = _run_once(child)
        if status != 0:
            return status
        if args.dry_run:
            continue
        run_reports.append(
            json.loads((Path(child.output_root) / "results.json").read_text(encoding="utf-8"))
        )
    if args.dry_run:
        return 0
    combined = summarize_runs(run_reports)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "results.md").write_text(
        render_comparison_markdown(combined),
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    lock = acquire_host_benchmark_lock()
    try:
        return _main(argv)
    finally:
        release_host_benchmark_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
