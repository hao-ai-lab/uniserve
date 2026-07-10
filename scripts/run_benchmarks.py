#!/usr/bin/env python3
"""Run a profile-declared benchmark matrix.

The runner owns mechanics only: launch one server, run selected harness points
serially, record commands/snapshots, and stop the server. Benchmark content
lives in ``uniserve_eval/profiles.json`` under ``benchmarks``.
"""

from __future__ import annotations

import argparse
import json
import math
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
from uniserve_eval.harness.datasets import load_benchmark_inputs  # noqa: E402
from uniserve_eval.harness.provenance import (  # noqa: E402
    effective_environment,
    execution_provenance,
    input_path_contract,
)
from uniserve_eval.harness.report import (  # noqa: E402
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
    expand_profile_value,
    load_config,
    require_resolved_profile_value,
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
    "tokenizer": "--tokenizer",
    "endpoint": "--endpoint",
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_p": "--top-p",
    "width": "--width",
    "height": "--height",
    "steps": "--steps",
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


@dataclass(frozen=True)
class BenchRunSpec:
    name: str
    group: str
    command: tuple[str, ...]
    output_dir: Path
    process_environment: dict[str, str]
    harness_contract: dict[str, Any] | None
    parity_group: str | None
    parity_contract: dict[str, Any] | None
    matrix_contract: dict[str, Any] | None


def shell_join(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def rate_tag(rate: str | int | float) -> str:
    if isinstance(rate, float) and math.isinf(rate):
        return "inf"
    text = ("%g" % rate) if isinstance(rate, float) else str(rate)
    return text.replace(".", "p")


def run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = effective_environment() if env is None else dict(env)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {shell_join(command)}\n")
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


def nvidia_smi() -> str:
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found"
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
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


def wait_for_clean_gpu(timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        text = nvidia_smi()
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
    raise RuntimeError(f"GPU memory did not return to zero:\n{nvidia_smi()}")


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
    except json.JSONDecodeError:
        return False
    if not canonical_artifact_bundle_matches(output_dir, data, bench.harness_contract):
        return False
    matrix_contract = data["artifact"].get("matrix_contract")
    return matrix_contract == bench.matrix_contract


def attach_matrix_contract(bench: BenchRunSpec) -> None:
    if bench.harness_contract is None or bench.matrix_contract is None:
        raise RuntimeError(f"benchmark {bench.name} has no resolved artifact contract")
    summary_path = bench.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not canonical_artifact_bundle_matches(
        bench.output_dir,
        summary,
        bench.harness_contract,
    ):
        raise RuntimeError(f"benchmark {bench.name} produced a mismatched harness contract")
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
        existing = sorted(out.glob(f"{prefix}_*.jpg"))
        if len(existing) == count and (out / "SOURCE.txt").exists():
            datasets[name] = out
            continue

        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob(f"{prefix}_*.jpg"):
            old.unlink()

        import random

        from datasets import load_dataset

        ds = load_dataset(str(spec["name"]), split=str(spec.get("split", "train")))
        indices = list(range(len(ds)))
        seed = int(spec.get("seed", 42))
        random.Random(seed).shuffle(indices)
        for n, idx in enumerate(indices[:count]):
            image = ds[int(idx)]["image"].convert("RGB")
            image.save(out / f"{prefix}_{n:03d}.jpg", format="JPEG", quality=95)
        (out / "SOURCE.txt").write_text(
            f"{spec['name']} {spec.get('split', 'train')} split, seed={seed}, n={count}\n",
            encoding="utf-8",
        )
        datasets[name] = out
    return datasets


def harness_command(
    *,
    python: str,
    base_url: str,
    output_dir: Path,
    defaults: dict[str, Any],
    harness: dict[str, Any],
    rate: str,
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
        str(rate),
    ]
    for key, flag in HARNESS_FLAGS.items():
        if key not in harness or harness[key] is None:
            continue
        value = harness.pop(key)
        if key == "cfg_interval" and isinstance(value, list):
            value = ",".join(str(part) for part in value)
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


def build_servers(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    strict_env: bool,
) -> dict[str, ServerRunSpec]:
    servers: dict[str, ServerRunSpec] = {}
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
        if spec.get("model") is not None and strict_env:
            model_contract = input_path_contract(str(spec["model"]), cwd=ROOT)
        servers[group_name] = ServerRunSpec(
            name=group_name,
            profile=profile_name,
            host=str(spec.get("host", "127.0.0.1")),
            port=int(spec["port"]),
            command=command,
            env=env,
            process_environment=effective_environment(env),
            model_contract=model_contract,
        )
    return servers


def build_benches(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    output_root: Path,
    datasets: dict[str, Path],
    servers: dict[str, ServerRunSpec],
    *,
    strict_env: bool,
) -> dict[str, list[BenchRunSpec]]:
    python = str(repo_path(expand_profile_value(config.get("python", ".venv/bin/python"))))
    defaults = dict(benchmark.get("defaults") or {})
    axes = dict(benchmark.get("load_axes") or {})
    point_specs = dict(benchmark.get("points") or {})
    groups: dict[str, list[BenchRunSpec]] = {}
    selected_rows_cache: dict[str, list[dict[str, Any]]] = {}
    harness_environment = effective_environment()
    server_execution: dict[str, dict[str, Any]] = {}
    declared_parity_groups: dict[str, list[str]] = {}
    for point_name, point in point_specs.items():
        parity_group = point.get("parity_group")
        if parity_group is not None:
            declared_parity_groups.setdefault(str(parity_group), []).append(point_name)
    invalid_groups = {
        group: members
        for group, members in declared_parity_groups.items()
        if len(members) < 2
    }
    if invalid_groups:
        raise SystemExit(f"benchmark parity groups need at least two points: {invalid_groups}")
    resolved_parity: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    for group_name, group in dict(benchmark["groups"]).items():
        server = expand_profile_value(server_spec(config, str(group["server"])))
        base_url = f"http://{server.get('host', '127.0.0.1')}:{server['port']}"
        benches: list[BenchRunSpec] = []
        for point_name in list(group.get("points", [])):
            point = dict(point_specs[str(point_name)])
            parity_group = (
                str(point["parity_group"]) if point.get("parity_group") is not None else None
            )
            axis = str(point.get("axis", "arrival_rate"))
            if axis not in axes:
                raise SystemExit(f"point {point_name} references unknown axis {axis!r}")
            for rate in axes[axis]:
                context = {"rate": str(rate), "rate_tag": rate_tag(rate)}
                name = str(point["name"]).format(**context)
                output_dir = output_root / str(point["output"]).format(**context)
                command = tuple(
                    harness_command(
                        python=python,
                        base_url=base_url,
                        output_dir=output_dir,
                        defaults=defaults,
                        harness=dict(point["harness"]),
                        rate=str(rate),
                        datasets=datasets,
                    )
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
                            "num_prompts": spec.num_prompts,
                            "seed": spec.seed,
                            "tokenizer": spec.tokenizer,
                            "sharegpt_context_len": spec.sharegpt_context_len,
                            "sharegpt_output_len": spec.sharegpt_output_len,
                            "i2t_question": spec.i2t_question,
                            "preprocessing": spec.preprocessing,
                        }
                    )
                    if row_key not in selected_rows_cache:
                        selected_rows_cache[row_key], _ = load_benchmark_inputs(spec)
                    harness_contract = benchmark_contract(spec, selected_rows_cache[row_key])
                    server_run = servers[group_name]
                    harness_parity_contract = benchmark_parity_contract(harness_contract)
                    parity_payload = {
                        "schema_version": 1,
                        "harness": harness_parity_contract,
                        "model": server_run.model_contract,
                    }
                    parity_contract = {
                        **parity_payload,
                        "fingerprint": canonical_digest(parity_payload),
                    }
                    if parity_group is not None:
                        parity_key = (parity_group, str(rate))
                        previous = resolved_parity.get(parity_key)
                        if previous is not None and previous[1] != parity_contract:
                            raise SystemExit(
                                f"benchmark parity mismatch for {parity_group!r} at rate {rate}: "
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
                    matrix_payload = {
                        "schema_version": 2,
                        "benchmark": name,
                        "server_profile": server_run.profile,
                        "server_execution": server_execution[group_name],
                        "harness_execution": execution_provenance(
                            command,
                            harness_environment,
                            cwd=ROOT,
                            workspace_root=ROOT,
                        ),
                        "harness_contract_fingerprint": harness_contract["fingerprint"],
                        "parity_group": parity_group,
                        "parity_contract": parity_contract,
                    }
                    matrix_contract = {
                        **matrix_payload,
                        "fingerprint": canonical_digest(matrix_payload),
                    }
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
        "- Load axis is open-loop request arrival rate.",
        "- Server launch commands are emitted from the benchmark profiles.",
        "- Each point records command.txt, preflight.txt, postflight.txt, and run.log.",
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
            lines.extend([f"Harness `{bench.name}`:", "", "```bash", shell_join(bench.command), "```", ""])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "COMMANDS.md").write_text("\n".join(lines), encoding="utf-8")


def launch_server(server: ServerRunSpec, group_dir: Path, timeout_s: float) -> subprocess.Popen[str]:
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "server_command.txt").write_text(shell_join(server.command) + "\n", encoding="utf-8")
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
    (group_dir / "server.exit").write_text(f"exit_code={proc.returncode}\nutc={now()}\n", encoding="utf-8")


def parse_only(value: str | None, all_groups: list[str]) -> set[str]:
    if not value:
        return set(all_groups)
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected - set(all_groups)
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


def filter_benchmark_groups(benchmark: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    filtered = dict(benchmark)
    filtered["groups"] = {
        name: group for name, group in dict(benchmark["groups"]).items() if name in selected
    }
    return filtered


def run_group(
    group_name: str,
    server: ServerRunSpec,
    benches: list[BenchRunSpec],
    *,
    output_root: Path,
    resume: bool,
    server_timeout_s: float,
    server_grace_s: float,
    require_clean_gpu: bool,
) -> None:
    group_dir = output_root / "servers" / group_name
    write_snapshot(group_dir / "pre_server_snapshot.txt", label=f"{group_name} pre-server")
    proc: subprocess.Popen[str] | None = None
    try:
        print(f"[{now()}] launching {group_name}", flush=True)
        proc = launch_server(server, group_dir, server_timeout_s)
        print(f"[{now()}] ready {group_name} port={server.port}", flush=True)
        for bench in benches:
            if resume and summary_ok(bench):
                print(f"[{now()}] skip complete {bench.name}", flush=True)
                continue
            bench.output_dir.mkdir(parents=True, exist_ok=True)
            (bench.output_dir / "command.txt").write_text(shell_join(bench.command) + "\n", encoding="utf-8")
            write_snapshot(bench.output_dir / "preflight.txt", label=f"{bench.name} preflight")
            print(f"[{now()}] run {bench.name}", flush=True)
            run_command(
                list(bench.command),
                log_path=bench.output_dir / "run.log",
                env=bench.process_environment,
            )
            write_snapshot(bench.output_dir / "postflight.txt", label=f"{bench.name} postflight")
            attach_matrix_contract(bench)
            if not summary_ok(bench):
                raise RuntimeError(f"benchmark did not produce a clean summary: {bench.output_dir}")
            print(f"[{now()}] done {bench.name}", flush=True)
    finally:
        print(f"[{now()}] stopping {group_name}", flush=True)
        stop_server(proc, group_dir, server_grace_s)
        write_snapshot(group_dir / "post_server_snapshot.txt", label=f"{group_name} post-server")
        if require_clean_gpu:
            wait_for_clean_gpu()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark", default="main")
    parser.add_argument("--output-root", help="override benchmark artifact root")
    parser.add_argument("--only", help="comma-separated server groups to run; defaults to the full matrix")
    parser.add_argument("--only-bench", help="comma-separated exact benchmark names to run after group filtering")
    parser.add_argument("--skip-bench", help="comma-separated exact benchmark names to skip after group filtering")
    parser.add_argument("--resume", action="store_true", help="skip complete summary.json points")
    parser.add_argument("--dry-run", action="store_true", help="write COMMANDS.md then exit")
    parser.add_argument("--no-build", action="store_true", help="skip cargo build --release --bin uniserve")
    parser.add_argument(
        "--require-clean-gpu",
        action="store_true",
        help="require all visible GPUs to report zero memory before and after each server group",
    )
    parser.add_argument("--server-timeout-s", type=float, default=1800.0)
    parser.add_argument("--server-grace-s", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    benchmark = benchmark_spec(config, args.benchmark)
    output_root = repo_path(args.output_root or str(benchmark.get("artifact_root", "artifacts/benchmarks")))
    selected = parse_only(args.only, list(dict(benchmark["groups"])))
    benchmark = filter_benchmark_groups(benchmark, selected)

    active = active_benchmark_processes()
    if active != "(none)":
        raise SystemExit(f"refusing to start with active benchmark/server processes:\n{active}")
    if args.require_clean_gpu and not args.dry_run:
        wait_for_clean_gpu()

    if not args.no_build and not args.dry_run:
        run_command(
            ["cargo", "build", "--release", "--bin", "uniserve"],
            log_path=output_root / "build.log",
        )

    datasets = materialize_datasets(output_root, benchmark) if not args.dry_run else {}
    if args.dry_run:
        for name, spec in dict(benchmark.get("datasets") or {}).items():
            datasets[name] = output_root / str(spec["output_dir"])

    servers = build_servers(config, benchmark, strict_env=not args.dry_run)
    groups = build_benches(
        config,
        benchmark,
        output_root,
        datasets,
        servers,
        strict_env=not args.dry_run,
    )
    write_runbook(output_root, args.benchmark, servers, groups, benchmark)

    all_bench_names = {bench.name for benches in groups.values() for bench in benches}
    only_benches = parse_name_filter(args.only_bench, all_bench_names, flag="--only-bench")
    skipped_benches = parse_name_filter(args.skip_bench, all_bench_names, flag="--skip-bench")

    if args.dry_run:
        print(output_root / "COMMANDS.md")
        return 0

    for group_name in groups:
        if group_name not in selected:
            continue
        benches = [
            bench
            for bench in groups[group_name]
            if (not only_benches or bench.name in only_benches) and bench.name not in skipped_benches
        ]
        if not benches:
            print(f"[{now()}] skip {group_name}: no benchmarks selected", flush=True)
            continue
        run_group(
            group_name,
            servers[group_name],
            benches,
            output_root=output_root,
            resume=args.resume,
            server_timeout_s=args.server_timeout_s,
            server_grace_s=args.server_grace_s,
            require_clean_gpu=args.require_clean_gpu,
        )

    write_snapshot(output_root / "final_snapshot.txt", label="final")
    print(f"[{now()}] complete selected groups: {', '.join(sorted(selected))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
