"""Sanitized provenance for reproducible benchmark execution contracts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the exact environment to pass to a benchmark process."""
    environment = dict(os.environ if inherited is None else inherited)
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def environment_contract(environment: Mapping[str, str]) -> dict[str, Any]:
    """Fingerprint all process variables without persisting their values."""
    variables = sorted((str(key), str(value)) for key, value in environment.items())
    payload = {
        "schema_version": 1,
        "variable_count": len(variables),
        "variables_sha256": _canonical_digest(variables),
    }
    return {**payload, "fingerprint": _canonical_digest(payload)}


_PERFORMANCE_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "NCCL_ALGO",
        "NCCL_BUFFSIZE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_IB_DISABLE",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_NET",
        "NCCL_NTHREADS",
        "NCCL_NVLS_ENABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_PROTO",
        "NCCL_SHM_DISABLE",
        "SGLANG_ENABLE_JIT_DEEPGEMM",
        "SGLANG_ENABLE_TORCH_COMPILE",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_FLASH_ATTN_VERSION",
        "VLLM_OMNI_USE_QUACK_FP8",
        "VLLM_USE_V1",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
)


def performance_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Persist only non-secret variables with known serving-performance impact."""
    return {
        str(key): str(value)
        for key, value in sorted(environment.items())
        if key in _PERFORMANCE_ENVIRONMENT_NAMES
    }


def _command_output(command: Sequence[str]) -> str | None:
    try:
        process = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = process.stdout.strip()
    return output if process.returncode == 0 and output else None


@lru_cache(maxsize=1)
def hardware_contract() -> dict[str, Any]:
    """Capture the accelerator, CPU/NUMA, and kernel identity used by a run."""
    gpu_query = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap,clocks.max.sm,clocks.max.memory,power.limit",
            "--format=csv,noheader,nounits",
        )
    )
    payload = {
        "schema_version": 1,
        "machine": platform.machine(),
        "kernel": platform.release(),
        "system": platform.system(),
        "cpu": _command_output(("lscpu", "--json")),
        "numa": _command_output(("numactl", "--hardware")),
        "gpu": gpu_query,
        "gpu_topology": _command_output(("nvidia-smi", "topo", "-m")),
    }
    return {**payload, "fingerprint": _canonical_digest(payload)}


def selected_accelerator_contract(visible_devices: str) -> dict[str, Any]:
    """Capture one explicitly selected physical accelerator."""
    devices = [value.strip() for value in visible_devices.split(",") if value.strip()]
    if len(devices) != 1:
        raise ValueError("formal execution requires exactly one visible accelerator")
    selector = devices[0]
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "compute_cap",
        "clocks.max.sm",
        "clocks.max.memory",
        "power.limit",
    )
    output = _command_output(
        (
            "nvidia-smi",
            f"--id={selector}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if output is None or len(output.splitlines()) != 1:
        raise RuntimeError(f"cannot resolve selected accelerator {selector!r}")
    values = [value.strip() for value in output.split(",")]
    if len(values) != len(fields):
        raise RuntimeError("selected accelerator query returned an unexpected shape")
    payload = {
        "schema_version": 1,
        "selector": selector,
        "gpu": dict(zip(fields, values, strict=True)),
    }
    return {**payload, "fingerprint": _canonical_digest(payload)}


@lru_cache(maxsize=4096)
def _file_sha256(
    path: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del device, inode, size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_command_file(value: str, environment: Mapping[str, str], cwd: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        lexical = candidate if candidate.is_absolute() else cwd / candidate
    else:
        found = shutil.which(value, path=environment.get("PATH"))
        if found is None:
            raise FileNotFoundError(f"benchmark executable is not resolvable: {value}")
        lexical = Path(found)
    if not lexical.exists():
        raise FileNotFoundError(f"benchmark executable does not exist: {lexical}")
    if not lexical.is_file():
        raise RuntimeError(f"benchmark executable is not a file: {lexical}")
    return lexical.absolute()


def _command_executable_paths(
    command: Sequence[str], environment: Mapping[str, str], cwd: Path
) -> list[tuple[str, Path]]:
    if not command:
        raise ValueError("benchmark command must not be empty")
    paths = [("launcher", _resolve_command_file(str(command[0]), environment, cwd))]
    for index, value in enumerate(command[1:], start=1):
        candidate = Path(str(value))
        if not (candidate.is_absolute() or candidate.parent != Path(".")):
            continue
        lexical = candidate if candidate.is_absolute() else cwd / candidate
        if lexical.is_file() and os.access(lexical, os.X_OK):
            paths.append((f"command_argument:{index}", lexical.absolute()))
    return paths


def executable_contracts(
    command: Sequence[str], environment: Mapping[str, str], *, cwd: str | Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Hash the launcher and executable command arguments such as wrapped servers."""
    contracts: list[dict[str, Any]] = []
    source_candidates: list[Path] = []
    for role, lexical in _command_executable_paths(command, environment, Path(cwd)):
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
        sha256 = _file_sha256(
            str(resolved),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        contracts.append(
            {
                "role": role,
                "requested": str(command[0]) if role == "launcher" else str(lexical),
                "resolved_name": resolved.name,
                "size_bytes": metadata.st_size,
                "sha256": sha256,
            }
        )
        source_candidates.extend((lexical, resolved))
    return contracts, source_candidates


_OUTPUT_PATH_OPTIONS = frozenset({"--cache-dir", "--download-dir", "--log-file", "--output-dir"})


def _hash_file_contract(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "kind": "file",
        "resolved_name": path.name,
        "size_bytes": metadata.st_size,
        "sha256": _file_sha256(
            str(path),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ),
    }


def _hash_directory_contract(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    for entry in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = entry.relative_to(path).as_posix()
        relative_parts = Path(relative).parts
        if ".git" in relative_parts or "__pycache__" in relative_parts:
            continue
        metadata = entry.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not entry.is_symlink():
            continue
        if entry.is_symlink():
            target = os.readlink(entry)
            resolved = entry.resolve(strict=True)
            if resolved.is_dir():
                raise RuntimeError(f"directory input contains a directory symlink: {entry}")
            target_contract = _hash_file_contract(resolved)
            total_size += int(target_contract["size_bytes"])
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target if not Path(target).is_absolute() else Path(target).name,
                    "target_size_bytes": target_contract["size_bytes"],
                    "target_sha256": target_contract["sha256"],
                }
            )
        elif stat.S_ISREG(metadata.st_mode):
            contract = _hash_file_contract(entry)
            total_size += int(contract["size_bytes"])
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "size_bytes": contract["size_bytes"],
                    "sha256": contract["sha256"],
                }
            )
        else:
            raise RuntimeError(f"unsupported command input entry: {entry}")
    payload = {
        "kind": "directory",
        "resolved_name": path.name,
        "file_count": len(entries),
        "total_size_bytes": total_size,
        "tree_sha256": _canonical_digest(entries),
    }
    return payload


def _path_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return _hash_file_contract(resolved)
    if resolved.is_dir():
        return _hash_directory_contract(resolved)
    raise RuntimeError(f"command input is not a file or directory: {path}")


def input_path_contract(path: str | Path, *, cwd: str | Path) -> dict[str, Any]:
    """Return a content identity for one declared file or directory input."""
    candidate = Path(path)
    lexical = candidate if candidate.is_absolute() else Path(cwd) / candidate
    if not lexical.exists():
        raise FileNotFoundError(f"declared benchmark input does not exist: {lexical}")
    return _path_contract(lexical)


def command_input_contracts(
    command: Sequence[str], *, cwd: str | Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Hash every existing file or directory consumed through command arguments."""
    cwd_path = Path(cwd)
    contracts: list[dict[str, Any]] = []
    source_candidates: list[Path] = []
    seen: set[Path] = set()
    for index, raw_value in enumerate(command[1:], start=1):
        value = str(raw_value)
        previous = str(command[index - 1]) if index > 0 else ""
        if previous in _OUTPUT_PATH_OPTIONS or value.startswith("-"):
            continue
        candidate = Path(value)
        lexical = candidate if candidate.is_absolute() else cwd_path / candidate
        if not lexical.exists():
            continue
        resolved = lexical.resolve(strict=True)
        if resolved in seen:
            continue
        seen.add(resolved)
        role = f"option:{previous}" if previous.startswith("--") else f"command_argument:{index}"
        contracts.append({"role": role, **_path_contract(lexical)})
        source_candidates.extend((lexical.absolute(), resolved))
    return contracts, source_candidates


def _python_interpreter(launcher: Path, environment: Mapping[str, str], cwd: Path) -> Path | None:
    if launcher.name.lower().startswith("python"):
        return launcher.absolute()
    try:
        with launcher.open("rb") as handle:
            first_line = handle.readline(4096).decode("utf-8", errors="strict").strip()
    except (OSError, UnicodeError):
        return None
    if not first_line.startswith("#!"):
        return None
    parts = shlex.split(first_line[2:])
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) >= 2:
        found = shutil.which(parts[1], path=environment.get("PATH"))
        interpreter = Path(found) if found else None
    else:
        candidate = Path(parts[0])
        interpreter = candidate if candidate.is_absolute() else cwd / candidate
    if interpreter is None or "python" not in interpreter.name.lower():
        return None
    if not interpreter.exists():
        return None
    return interpreter.absolute()


def _python_invocation(
    command: Sequence[str], environment: Mapping[str, str], cwd: Path
) -> tuple[Path, list[str], bool, str] | None:
    """Find a Python interpreter even when a process wrapper precedes it."""
    for index, raw_value in enumerate(command):
        value = str(raw_value)
        if index > 0:
            candidate = Path(value)
            if not (candidate.is_absolute() or candidate.parent != Path(".")):
                continue
            lexical = candidate if candidate.is_absolute() else cwd / candidate
            if not lexical.is_file() or not os.access(lexical, os.X_OK):
                continue
        else:
            lexical = _resolve_command_file(value, environment, cwd)
        interpreter = _python_interpreter(lexical, environment, cwd)
        if interpreter is not None:
            launcher = "" if lexical.name.lower().startswith("python") else lexical.stem
            return (
                interpreter,
                [str(part) for part in command[index + 1 :]],
                index == 0,
                launcher,
            )
    return None


_PYTHON_PROBE = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys

module = sys.argv[1] or None
launcher = sys.argv[2] or None
distributions = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name") or ""
    version = distribution.version or ""
    record = distribution.read_text("RECORD") or distribution.read_text("installed-files.txt") or ""
    direct = distribution.read_text("direct_url.json") or ""
    distributions.append({
        "name": name.casefold(),
        "version": version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "direct_url_sha256": hashlib.sha256(direct.encode()).hexdigest(),
    })
    if module is None and launcher:
        for entry_point in distribution.entry_points:
            if entry_point.group == "console_scripts" and entry_point.name == launcher:
                module = entry_point.value.partition(":")[0]
                break
if module is None and launcher:
    module = launcher.replace("-", "_")
distributions.sort(key=lambda item: (item["name"], item["version"], item["record_sha256"]))
origin = None
locations = []
if module:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        spec = None
    if spec is not None:
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            origin = str(pathlib.Path(spec.origin).resolve())
        locations = [str(pathlib.Path(value).resolve()) for value in (spec.submodule_search_locations or ())]
print(json.dumps({
    "implementation": sys.implementation.name,
    "cache_tag": sys.implementation.cache_tag,
    "version": list(sys.version_info[:5]),
    "distributions": distributions,
    "sys_path": [str(pathlib.Path(value or ".").resolve()) for value in getattr(sys, "path")],
    "module": module,
    "module_origin": origin,
    "module_locations": locations,
}, sort_keys=True))
"""


def python_runtime_contract(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    cwd: str | Path,
) -> tuple[dict[str, Any] | None, list[Path], list[dict[str, Any]]]:
    """Capture the exact Python environment and target package when applicable."""
    cwd_path = Path(cwd).resolve()
    invocation = _python_invocation(command, environment, cwd_path)
    if invocation is None:
        return None, [], []
    interpreter, python_arguments, interpreter_is_launcher, console_launcher = invocation
    module = ""
    if len(python_arguments) >= 2 and python_arguments[0] == "-m":
        module = python_arguments[1]
    elif interpreter_is_launcher:
        launcher = _resolve_command_file(str(command[0]), environment, cwd_path)
        if not launcher.resolve(strict=True).name.lower().startswith("python"):
            module = launcher.stem.replace("-", "_")
    process = subprocess.run(
        [str(interpreter), "-c", _PYTHON_PROBE, module, console_launcher],
        cwd=cwd_path,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Python provenance probe failed: {detail}")
    probe = json.loads(process.stdout.decode("utf-8", errors="strict"))
    distributions = probe.pop("distributions")
    sys_path = [Path(value) for value in probe.pop("sys_path")]
    module_paths = []
    module_origin = probe.pop("module_origin")
    module_locations = probe.pop("module_locations")
    if module_origin:
        module_paths.append(Path(module_origin))
    module_paths.extend(Path(value) for value in module_locations)
    module_sources = [
        {"role": f"python_module:{index}", **_path_contract(path)}
        for index, path in enumerate(module_paths)
    ]
    payload = {
        "schema_version": 2,
        **probe,
        "distribution_count": len(distributions),
        "distributions_sha256": _canonical_digest(distributions),
        "distributions": distributions,
        "module_source_count": len(module_sources),
    }
    return (
        {**payload, "fingerprint": _canonical_digest(payload)},
        [*sys_path, *module_paths, interpreter],
        module_sources,
    )


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed in {root}: {detail}")
    return process.stdout


def _git_root(path: Path) -> Path | None:
    candidate = path if path.is_dir() else path.parent
    process = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        return None
    return Path(process.stdout.decode("utf-8", errors="strict").strip()).resolve()


def _hash_untracked_files(root: Path, paths: list[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"untracked-files-v1\0")
    for raw_path in paths:
        path = root / os.fsdecode(raw_path)
        metadata = path.lstat()
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(stat.S_IFMT(metadata.st_mode).to_bytes(8, "big"))
        digest.update(metadata.st_size.to_bytes(8, "big"))
        if path.is_symlink():
            target = os.fsencode(os.readlink(path))
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise RuntimeError(f"unsupported untracked source entry: {path}")
    return digest.hexdigest()


def repository_state(path: str | Path) -> dict[str, Any]:
    """Record a repository revision plus all tracked and untracked changes."""
    root = _git_root(Path(path).absolute())
    if root is None:
        raise RuntimeError(f"source path is not in a Git repository: {path}")
    head = _run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    tracked_changes = _run_git(root, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    raw_untracked = _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z", "--"])
    untracked_paths = sorted(path for path in raw_untracked.split(b"\0") if path)
    payload = {
        "schema_version": 1,
        "head": head,
        "dirty": bool(tracked_changes or untracked_paths),
        "tracked_changes_sha256": hashlib.sha256(tracked_changes).hexdigest(),
        "untracked_files_sha256": _hash_untracked_files(root, untracked_paths),
        "untracked_file_count": len(untracked_paths),
    }
    return {**payload, "fingerprint": _canonical_digest(payload)}


def _source_revisions(
    *,
    workspace_root: Path,
    source_paths: Sequence[tuple[str, Path]],
    environment: Mapping[str, str],
    cwd: Path,
) -> list[dict[str, Any]]:
    repositories: dict[Path, set[str]] = {}

    def add(role: str, path: Path) -> None:
        root = _git_root(path)
        if root is not None:
            repositories.setdefault(root, set()).add(role)

    add("workspace", workspace_root)
    for role, path in source_paths:
        add(role, path)
    for index, value in enumerate(environment.get("PYTHONPATH", "").split(os.pathsep)):
        if not value:
            continue
        path = Path(value)
        add(f"pythonpath:{index}", path if path.is_absolute() else cwd / path)

    revisions = []
    for root, roles in sorted(repositories.items(), key=lambda item: sorted(item[1])):
        revisions.append({"roles": sorted(roles), "state": repository_state(root)})
    return revisions


def execution_provenance(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    cwd: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Build a relocation-safe identity for one exact process invocation."""
    cwd_path = Path(cwd).resolve()
    workspace_path = Path(workspace_root).resolve()
    executables, executable_paths = executable_contracts(command, environment, cwd=cwd_path)
    command_inputs, command_input_paths = command_input_contracts(command, cwd=cwd_path)
    python_runtime, python_paths, python_module_sources = python_runtime_contract(
        command,
        environment,
        cwd=cwd_path,
    )
    source_paths = [
        *((f"executable:{index}", path) for index, path in enumerate(executable_paths)),
        *((f"command_input:{index}", path) for index, path in enumerate(command_input_paths)),
        *((f"python:{index}", path) for index, path in enumerate(python_paths)),
    ]
    payload = {
        "schema_version": 2,
        "command": [str(part) for part in command],
        "working_directory": str(cwd_path),
        "environment": environment_contract(environment),
        "performance_environment": performance_environment(environment),
        "executables": executables,
        "command_inputs": command_inputs,
        "python_runtime": python_runtime,
        "python_module_sources": python_module_sources,
        "source_revisions": _source_revisions(
            workspace_root=workspace_path,
            source_paths=source_paths,
            environment=environment,
            cwd=cwd_path,
        ),
    }
    return {**payload, "fingerprint": _canonical_digest(payload)}
