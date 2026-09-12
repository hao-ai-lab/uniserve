"""Checkpoint discovery and closed file-set resolution for local and remote models."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from safetensors.torch import safe_open

from .config import LoadConfig, LoadFormat, LoadRequest

__all__ = [
    "WeightSourceConfig",
    "WeightSourceSet",
    "read_model_config",
    "resolve_model_root",
    "resolve_weight_sources",
]

_TRAINING_FILES = frozenset(
    {
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    }
)
_SAFETENSORS_INDEX = "*.safetensors.index.json"
_PT_INDEX = "*.bin.index.json"


@dataclass(frozen=True, slots=True)
class WeightSourceConfig:
    """Declare a component directory or ordered checkpoint filename choices."""

    name: str = "primary"
    directory: str = ""
    filenames: tuple[str, ...] = ()
    entry: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("weight source requires a component name")
        for value in (self.directory, *self.filenames):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("weight sources must stay within the checkpoint root")


@dataclass(frozen=True, slots=True)
class WeightSourceSet:
    """Exact files and logical namespace belonging to one checkpoint source."""

    root: Path
    weight_files: tuple[Path, ...]
    relative_paths: tuple[str, ...]
    name_prefix: str = ""
    source_name: str = "primary"

    def __post_init__(self) -> None:
        """Validate one-to-one file identities and require every local artifact to exist."""

        if len(self.weight_files) != len(self.relative_paths):
            raise ValueError("weight files and relative paths must align")
        if len(set(self.relative_paths)) != len(self.relative_paths):
            raise ValueError("a weight source cannot contain duplicate relative paths")
        missing = [path for path in self.weight_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"checkpoint source is missing {missing[0]}")

    def preview_shape(self, tensor_name: str) -> tuple[int, ...]:
        """Read one tensor's shape while avoiding safetensors payload materialization."""

        # Safetensors exposes shape metadata independently from tensor storage.
        for path in self.weight_files:
            if path.suffix == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    if tensor_name in checkpoint.keys():
                        return tuple(
                            int(value) for value in checkpoint.get_slice(tensor_name).get_shape()
                        )
                continue

            # PT containers require deserialization before their tensor metadata is visible.
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
                state = state["state_dict"]
            if isinstance(state, dict) and tensor_name in state:
                value = state[tensor_name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{tensor_name!r} in {path} is not a tensor")
                return tuple(int(size) for size in value.shape)
        raise KeyError(f"checkpoint source has no tensor {tensor_name!r}")


def resolve_model_root(model_path: str, load: LoadConfig) -> tuple[Path, str | None]:
    """Resolve a local model root or fetch remote configuration into the hub cache.

    Return the resolved directory together with the repository identifier needed for
    subsequent remote file discovery.
    """

    # Existing paths are complete local identities, including direct weight files.
    candidate = Path(model_path)
    if candidate.exists():
        return (candidate.parent if candidate.is_file() else candidate), None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    # Either supported configuration file is sufficient to establish the cached root.
    config_path: Path | None = None
    for filename in ("config.json", "modular_model_index.json"):
        try:
            config_path = Path(
                hf_hub_download(
                    repo_id=model_path,
                    filename=filename,
                    cache_dir=load.download_dir,
                    revision=load.revision,
                )
            )
            break
        except EntryNotFoundError:
            continue
    if config_path is None:
        raise FileNotFoundError(
            f"checkpoint {model_path!r} has neither config.json nor modular_model_index.json"
        )
    return config_path.parent, model_path


def read_model_config(root: Path) -> dict[str, Any]:
    """Read and normalize a checkpoint root's architecture configuration object."""

    # Modular pipelines carry their architecture in a pipeline index rather than config.json.
    path = root / "config.json"
    if not path.is_file():
        path = root / "modular_model_index.json"
    if not path.is_file():
        raise FileNotFoundError("checkpoint is missing 'config.json' or 'modular_model_index.json'")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint {path.name} must contain an object")

    # Normalize the supported modular pipeline onto the architecture dispatch contract.
    if path.name == "modular_model_index.json":
        if value.get("_class_name") != "MiniMaxH3ModularPipeline":
            raise ValueError("checkpoint modular_model_index.json declares an unsupported pipeline")
        return {**value, "architectures": ["MiniMaxH3Transformer3DModel"]}
    return value


def resolve_weight_sources(
    request: LoadRequest,
    *,
    sources: tuple[WeightSourceConfig, ...],
    sidecars: tuple[str, ...],
    root: Path,
    repository_id: str | None,
) -> tuple[WeightSourceSet, ...]:
    """Resolve declared component sources using one closed repository inventory."""

    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("checkpoint sources require unique nonempty component names")
    sources = tuple(
        source
        for source in sources
        if source.entry is None
        or (source.entry in request.bindings and request.bindings[source.entry].owns)
    )
    # Synthetic loading carries a source identity without checkpoint files.
    if request.load.load_format is LoadFormat.DUMMY:
        return tuple(
            WeightSourceSet(root=root, weight_files=(), relative_paths=(), source_name=source.name)
            for source in sources
        )

    # Freeze the visible repository inventory before selecting or fetching artifacts.
    available = _available_files(root, repository_id, request.load)
    _fetch_sidecars(
        sidecars,
        available=available,
        repository_id=repository_id,
        load=request.load,
    )

    resolved = []
    for source in sources:
        directory = PurePosixPath(source.directory)
        names: tuple[str, ...]
        if source.filenames:
            selected = next(
                (
                    (directory / name).as_posix()
                    for name in source.filenames
                    if (directory / name).as_posix() in available
                    and _accepts_suffix(name, request.load.load_format)
                ),
                None,
            )
            if selected is None:
                raise FileNotFoundError(
                    f"component {source.name!r} requires a {request.load.load_format.value} "
                    f"checkpoint from {source.filenames!r} under {root / source.directory}"
                )
            names = (selected,)
        else:
            names = _select_primary(
                model_path=request.model_path,
                root=root,
                available=available,
                load=request.load,
                repository_id=repository_id,
                directory=directory,
            )
        paths = tuple(_fetch_file(name, root, repository_id, request.load) for name in names)
        resolved.append(WeightSourceSet(root, paths, names, source_name=source.name))
    return tuple(resolved)


def _available_files(root: Path, repository_id: str | None, load: LoadConfig) -> tuple[str, ...]:
    """List non-ignored checkpoint files using stable repository-relative names."""

    if repository_id is None:
        return tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and not _ignored(path.relative_to(root).as_posix(), load.ignore_patterns)
            )
        )
    from huggingface_hub import HfApi

    names = HfApi().list_repo_files(repo_id=repository_id, revision=load.revision)
    return tuple(sorted(name for name in names if not _ignored(name, load.ignore_patterns)))


def _select_primary(
    *,
    model_path: str,
    root: Path,
    available: tuple[str, ...],
    load: LoadConfig,
    repository_id: str | None,
    directory: PurePosixPath = PurePosixPath("."),
) -> tuple[str, ...]:
    """Select an explicit file, indexed shards, or top-level files for the load format."""

    # A direct local file bypasses repository-wide candidate selection.
    local_candidate = Path(model_path)
    if repository_id is None and local_candidate.is_file() and directory == PurePosixPath("."):
        relative = local_candidate.relative_to(root).as_posix()
        if not _accepts_suffix(relative, load.load_format):
            raise ValueError(
                f"load format {load.load_format.value!r} does not accept checkpoint file {relative!r}"
            )
        return (relative,)

    # Prefer an index when present; otherwise collect one supported top-level format.
    for index_name, pattern in _format_candidates(load.load_format):
        if index_name is not None:
            indexes = tuple(
                name
                for name in available
                if PurePosixPath(name).parent == directory
                and fnmatch.fnmatchcase(PurePosixPath(name).name, index_name)
            )
            if len(indexes) > 1:
                raise ValueError(f"component directory {directory} has multiple checkpoint indexes")
            if indexes:
                return _index_weight_files(indexes[0], root, available, load, repository_id)
        if pattern is None:
            continue
        matches = tuple(
            name
            for name in available
            if PurePosixPath(name).parent == directory
            and fnmatch.fnmatchcase(PurePosixPath(name).name, pattern)
            and Path(name).name not in _TRAINING_FILES
        )
        if matches:
            return tuple(sorted(matches))
    raise FileNotFoundError(
        f"no {load.load_format.value} checkpoint weight files found under {root}"
    )


def _format_candidates(load_format: LoadFormat) -> tuple[tuple[str | None, str | None], ...]:
    """Return index and filename candidates in checkpoint format preference order."""

    if load_format in {LoadFormat.AUTO, LoadFormat.LAYERED}:
        return (
            (_SAFETENSORS_INDEX, None),
            (None, "*.safetensors"),
            (_PT_INDEX, None),
            (None, "*.bin"),
            (None, "*.pt"),
        )
    if load_format is LoadFormat.SAFETENSORS:
        return ((_SAFETENSORS_INDEX, None), (None, "*.safetensors"))
    if load_format is LoadFormat.PT:
        return ((_PT_INDEX, None), (None, "*.pt"), (None, "*.bin"))
    return ()


def _index_weight_files(
    index_name: str,
    root: Path,
    available: tuple[str, ...],
    load: LoadConfig,
    repository_id: str | None,
) -> tuple[str, ...]:
    """Read a checkpoint index and validate its unique shard file set."""

    index_path = _fetch_file(index_name, root, repository_id, load)
    value = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"checkpoint index {index_name!r} has no weight_map")
    directory = PurePosixPath(index_name).parent
    names = tuple(sorted({(directory / str(name)).as_posix() for name in mapping.values()}))
    unavailable = [name for name in names if name not in available]
    if unavailable:
        raise FileNotFoundError(f"checkpoint index references missing shard {unavailable[0]!r}")
    return names


def _fetch_sidecars(
    patterns: tuple[str, ...],
    *,
    available: tuple[str, ...],
    repository_id: str | None,
    load: LoadConfig,
) -> None:
    """Fetch remote architecture sidecars that match the declared glob patterns."""

    if repository_id is None:
        return
    for name in available:
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            _fetch_file(name, Path(), repository_id, load)


def _fetch_file(name: str, root: Path, repository_id: str | None, load: LoadConfig) -> Path:
    """Resolve one required local file or materialize it through the hub cache."""

    if repository_id is None:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint source is missing {name!r}")
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repository_id,
            filename=name,
            cache_dir=load.download_dir,
            revision=load.revision,
        )
    )


def _ignored(name: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a repository-relative path matches an ignore pattern."""

    return any(
        fnmatch.fnmatchcase(name, pattern) or PurePosixPath(name).match(pattern)
        for pattern in patterns
    )


def _accepts_suffix(name: str, load_format: LoadFormat) -> bool:
    """Return whether a direct checkpoint file is compatible with the load format."""

    suffix = Path(name).suffix
    if load_format in {LoadFormat.AUTO, LoadFormat.LAYERED}:
        return suffix in {".safetensors", ".bin", ".pt"}
    if load_format is LoadFormat.SAFETENSORS:
        return suffix == ".safetensors"
    if load_format is LoadFormat.PT:
        return suffix in {".bin", ".pt"}
    return False
