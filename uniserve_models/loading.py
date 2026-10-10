"""Discover concrete models and normalize their public loading inputs.

``read_config`` resolves a local checkpoint directory or a Hub repository to
the model package named in ``_catalog``, reads the package's typed
configuration, fetches the payload files the selected modules need (plus the
tensor headers the package reads to build its configuration), and returns an
immutable ``Config``. ``load_model`` then materializes the model
through ``uniserve.loading.load_model``. The worker bootstrap
(``uniserve_worker.bootstrap.model_loader``) and direct Python callers use the
same two calls.

A model package provides ``config_sources``, ``read_config``, ``Model``,
``checkpoint_sources``, ``checkpoint_mappings``, ``entry_points``,
``image_processor``, ``flow_prompt``, ``precisions`` and
``checkpoint_precision``. A package whose checkpoints may omit components
that a base revision supplies also provides ``base_checkpoint(root)``, which
returns that revision's ``(repository, revision)`` or ``None``; a directory
the checkpoint omits is then read from the base revision. This module also
recognizes calibrated ModelOpt NVFP4 exports.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

import torch
from torch import nn

from uniserve import loading
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.loading import checkpoint
from uniserve.loading import weights as weight_options
from uniserve.model import ComponentEntry
from uniserve.nn.attention import AttentionParallelConfig
from uniserve.nn.linear import Linear
from uniserve.nn.moe import ExpertLinear
from uniserve.processing import FlowPrompt, ImageProcessor
from uniserve.quantization import QuantizationConfig, Quantizer

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT", bound=nn.Module)

# Architecture a checkpoint declares, mapped to the package implementing it.
# A model checkpoint declares `architectures` in its root `config.json`; a
# diffusers pipeline declares its pipeline class in a root index instead.
_catalog: Mapping[str, str] = MappingProxyType(
    {
        "Qwen3ForCausalLM": "uniserve_models.qwen3",
        "Qwen3MoeForCausalLM": "uniserve_models.qwen3",
        "BagelForConditionalGeneration": "uniserve_models.bagel",
        "DiffusionGemmaForBlockDiffusion": "uniserve_models.diffusion_gemma",
        "NEOChatModel": "uniserve_models.sensenova_u1",
        "MiniMaxH3ModularPipeline": "uniserve_models.minimax_h3",
    }
)

# Root metadata files, in the order they identify a checkpoint: a model
# configuration, then the classic and modular diffusers pipeline indexes.
_metadata_files = (
    "config.json",
    "model_index.json",
    "modular_model_index.json",
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """One checkpoint directory and where its files come from.

    ``repository`` and ``revision`` name the Hub snapshot that ``root`` is
    the local cache of, or are ``None`` for a local directory; ``inventory``
    lists the files ``read_config`` may read, relative to ``root``.
    """

    root: Path
    repository: str | None
    revision: str | None
    inventory: frozenset[str]


@dataclass(frozen=True, slots=True)
class Config(Generic[ConfigT, ModelT]):
    """Resolved architecture, closed checkpoint sources, and caller-owned
    assets.

    No tokenizer, reader, resource context, or model instance is retained.
    Module selection uses actual module paths and also selects shared
    descendants by identity. Loading can narrow an already resolved selection.

    Attributes:
        model: The package's typed model configuration.
        model_class: Constructor that builds the model from ``model``.
        checkpoint: Resolved sources the selected modules read.
        mapping: The package's ``checkpoint_mappings``.
        entry_points: Component entry points the package declares.
        weights: Default weight configuration for ``load_model``.
        precisions: Named presets ``load_model`` accepts; empty for a
            calibrated ModelOpt checkpoint.
        checkpoint_format: ``"modelopt_nvfp4"`` for a calibrated ModelOpt
            checkpoint, otherwise ``None``. The worker bootstrap refuses
            launch quantization overrides when it is set.
        io: Checkpoint IO policy used for resolution and loading.
        tokenizer: Directory holding tokenizer files, or ``None``.
        image_processor: Image processor the package builds from ``model``,
            with its feature-injection token IDs resolved, or ``None``.
        flow_prompt: The package's classifier-free-guidance prompt framing,
            or ``None``.
        modules: Selected module paths; ``None`` selects the whole model.
        exclude_modules: Subtrees excluded from that selection by identity.
    """  # noqa: D205

    model: ConfigT
    model_class: Callable[[ConfigT], ModelT]
    checkpoint: tuple[checkpoint.Source, ...]
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]]
    entry_points: Mapping[str, ComponentEntry]
    weights: weight_options.Config
    precisions: Mapping[str, weight_options.Config]
    checkpoint_format: str | None
    io: loading.Config
    tokenizer: Path | None
    image_processor: ImageProcessor | None
    flow_prompt: FlowPrompt | None
    modules: frozenset[str] | None
    exclude_modules: frozenset[str] = frozenset()

    def __post_init__(self):
        # Freeze the caller's containers; ``ComponentEntry`` values are
        # already immutable.
        object.__setattr__(
            self, "entry_points", MappingProxyType(dict(self.entry_points))
        )
        object.__setattr__(
            self, "precisions", MappingProxyType(dict(self.precisions))
        )
        if self.modules is not None:
            object.__setattr__(self, "modules", frozenset(self.modules))
        object.__setattr__(
            self, "exclude_modules", frozenset(self.exclude_modules)
        )


def _json(path: Path) -> dict:
    """Read a JSON object; invalid JSON or a non-object raises ValueError."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint metadata {path} must contain an object")
    return value


def _root(path: str | Path, io: loading.Config):
    """Resolve a local checkpoint directory or pin a Hub snapshot to a local snapshot root."""  # noqa: E501
    # Returns ``(root, repository, revision)``. A local path yields its
    # directory (the parent of a named file) with no repository or revision;
    # a Hub repository yields its snapshot directory, the repository id and
    # the snapshot's commit revision. A ``Path`` or absolute string that does
    # not exist is never treated as a Hub repository id; it raises
    # ``FileNotFoundError``, as does a repository publishing none of
    # ``_metadata_files``.
    candidate = Path(path).expanduser()
    if candidate.exists():
        return (
            (candidate.parent if candidate.is_file() else candidate),
            None,
            None,
        )
    if isinstance(path, Path) or candidate.is_absolute():
        raise FileNotFoundError(candidate)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in _metadata_files:
        try:
            file = Path(
                hf_hub_download(
                    repo_id=str(path),
                    filename=name,
                    revision=io.revision,
                    cache_dir=io.download_dir,
                )
            )
            # Subsequent metadata and payload reads use this immutable snapshot,
            # even if the original branch name moves while downloading.
            return file.parent, str(path), file.parent.name
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(
        f"checkpoint {path!r} has no supported model metadata"
    )


def _inventory(root, repository, revision, io):
    """List checkpoint files, excluding caller-configured ignore patterns."""
    if repository is None:
        names: Iterable[str] = (
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    else:
        from huggingface_hub import HfApi

        names = HfApi().list_repo_files(repo_id=repository, revision=revision)

    return frozenset(
        name
        for name in names
        if not any(
            fnmatch.fnmatchcase(name, pattern)
            or PurePosixPath(name).match(pattern)
            for pattern in io.ignore_patterns
        )
    )


def _fetch(root, names, repository, revision, io):
    """Download the named files of a Hub revision into the cache.

    A local checkpoint needs no fetch. Downloads run on ``io.num_threads``
    threads (8 when unset); a download failure propagates to the caller.
    """
    if repository is None:
        return

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def fetch(name):
        hf_hub_download(
            repo_id=repository,
            filename=name,
            revision=revision,
            cache_dir=io.download_dir,
        )

    with ThreadPoolExecutor(max_workers=io.num_threads or 8) as executor:
        tuple(executor.map(fetch, sorted(names)))


def _source_files(source, root, inventory, io):
    """Choose the declared file encoding before downloading checkpoint payloads."""  # noqa: E501
    # This applies the selection rule of ``checkpoint.Config.resolve`` to the
    # inventory before the payloads exist locally: explicit filenames first,
    # then a sharded index, then every weight file of the format. The two
    # must stay in agreement, or ``resolve`` looks for a file this function
    # did not fetch. An index is read from ``root``, so the sidecar fetch
    # must precede this call.
    directory = PurePosixPath(source.directory)
    available = {
        name for name in inventory if PurePosixPath(name).parent == directory
    }
    formats = ("safetensors", "pt") if io.format == "auto" else (io.format,)
    suffixes = {"safetensors": {".safetensors"}, "pt": {".pt", ".bin"}}

    # An explicit filename list wins over any index or layout convention.
    if source.filenames:
        for filename in source.filenames:
            name = (directory / filename).as_posix()
            if name in available and any(
                Path(name).suffix in suffixes[format] for format in formats
            ):
                return (name,)
        raise FileNotFoundError(
            f"source {source.name!r} requires one of {source.filenames!r}"
        )

    for format in formats:
        pattern = (
            "*.safetensors.index.json"
            if format == "safetensors"
            else "*.bin.index.json"
        )
        indexes = sorted(
            name
            for name in available
            if fnmatch.fnmatchcase(PurePosixPath(name).name, pattern)
        )
        if len(indexes) > 1:
            raise ValueError(
                f"source {source.name!r} has multiple {format} indexes"
            )

        if indexes:
            # A sharded index declares its payload files relative to the
            # source directory.
            mapping = _json(root / indexes[0]).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "checkpoint index requires a nonempty weight_map"
                )
            paths = tuple(PurePosixPath(name) for name in set(mapping.values()))
            if any(path.is_absolute() or ".." in path.parts for path in paths):
                raise ValueError(
                    "checkpoint index paths must stay inside their directory"
                )
            names = tuple((directory / path).as_posix() for path in paths)
            if set(names).difference(inventory):
                raise FileNotFoundError(
                    "checkpoint index references a missing or excluded shard"
                )
            return names

        # Without an index, every weight file in the directory is a payload;
        # training state saved alongside the model is not part of it.
        names = tuple(
            name
            for name in available
            if Path(name).suffix in suffixes[format]
            and Path(name).name
            not in {
                "training_args.bin",
                "optimizer.pt",
                "optimizer.bin",
                "scheduler.pt",
                "scaler.pt",
            }
        )
        if names:
            return names

    raise FileNotFoundError(
        f"source {source.name!r} has no {io.format} checkpoint weights"
    )


def _safetensors_headers(repository, revision, names, io):
    """Read the tensor entries of Hub safetensors files from their headers.

    Each file costs HTTP range requests over its length-prefixed JSON header,
    never its tensor payload, with the saved Hub token as for every other Hub
    request here; files are read on ``io.num_threads`` threads (8 when
    unset). Returns the ``(name, shape, dtype)`` entries of all files, which
    ``checkpoint.Source`` accepts as ``headers``. A header that does not
    parse raises ``ValueError``; a request failure propagates.
    """
    if not names:
        return ()

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi
    from huggingface_hub.errors import SafetensorsParsingError

    api = HfApi()

    def read(name):
        return api.parse_safetensors_file_metadata(
            repo_id=repository, filename=name, revision=revision
        )

    try:
        with ThreadPoolExecutor(max_workers=io.num_threads or 8) as executor:
            files = tuple(executor.map(read, sorted(names)))
    except SafetensorsParsingError as error:
        raise ValueError(f"malformed safetensors header: {error}") from error

    return tuple(
        (name, tuple(tensor.shape), tensor.dtype)
        for file in files
        for name, tensor in sorted(file.tensors.items())
    )


def _config_sources(package, located, io):
    """Resolve the sources ``package.read_config`` reads tensor headers from.

    Returns ``package.config_sources`` resolved by name, each in the
    snapshot ``located(directory)`` holds its directory in. A local
    checkpoint resolves in place. A Hub checkpoint downloads the files of
    each source, except under dummy loading, which synthesizes weight values
    and needs only the headers: a file the Hub cache already holds is read
    locally, a safetensors file contributes only its header
    (``_safetensors_headers``), and a PyTorch container, whose tensor
    metadata is not separable from its payload, is downloaded.
    """
    sources = {}
    for declaration in package.config_sources:
        snapshot = located(declaration.directory)
        root, repository, revision, inventory = (
            snapshot.root,
            snapshot.repository,
            snapshot.revision,
            snapshot.inventory,
        )
        if repository is None:
            sources[declaration.name] = declaration.resolve(root, io=io)
            continue

        names = _source_files(declaration, root, inventory, io)
        if io.mode != "dummy":
            _fetch(root, names, repository, revision, io)
            sources[declaration.name] = declaration.resolve(root, io=io)
            continue

        # ``root`` is the pinned snapshot directory of the Hub cache, so a
        # present file is the cached copy of this revision's file.
        missing = {name for name in names if not (root / name).is_file()}
        remote = {
            name
            for name in missing
            if PurePosixPath(name).suffix == ".safetensors"
        }
        _fetch(root, missing - remote, repository, revision, io)
        sources[declaration.name] = checkpoint.Source(
            declaration.name,
            root,
            tuple(sorted(root / name for name in set(names) - remote)),
            declaration.prefix,
            headers=_safetensors_headers(repository, revision, remote, io),
        )
    return sources


def _tokenizer_vocabulary(path: Path) -> dict[str, object]:
    """Map token text to ID from a ``tokenizer.json`` model and added tokens.

    The model vocabulary is an object from token text to ID or, for a
    Unigram model, a list of ``[text, score]`` rows whose position is the ID;
    ``added_tokens`` is a list of objects carrying ``content`` and ``id``.
    Added tokens override the model vocabulary. Any other layout raises
    ``ValueError``; IDs are returned unchecked for ``_tokens`` to validate.
    """
    data = _json(path)
    model = data.get("model", {})
    vocabulary = model.get("vocab", {}) if isinstance(model, dict) else None
    added = data.get("added_tokens", [])

    if isinstance(vocabulary, list) and all(
        isinstance(row, list) and row and isinstance(row[0], str)
        for row in vocabulary
    ):
        vocabulary = {row[0]: index for index, row in enumerate(vocabulary)}
    if (
        not isinstance(vocabulary, dict)
        or not isinstance(added, list)
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("content"), str)
            or "id" not in entry
            for entry in added
        )
    ):
        raise ValueError(f"tokenizer metadata {path} has a malformed layout")

    return {
        **vocabulary,
        **{entry["content"]: entry["id"] for entry in added},
    }


def _tokens(processor, root):
    """Fill declared start/end feature-injection token IDs from checkpoint tokenizer files."""  # noqa: E501
    # Only tokens declared by text without an ID are resolved; a processor
    # with nothing to resolve is returned unchanged. A declared token that no
    # vocabulary source maps to a nonnegative integer, or a tokenizer file
    # with a malformed layout, raises ``ValueError``.
    if processor is None or processor.feature_injection is None:
        return processor
    injection = processor.feature_injection
    required = {
        name: getattr(injection, name)
        for name in ("start_token", "end_token")
        if getattr(injection, name) is not None
        and getattr(injection, name + "_id") is None
    }
    if not required:
        return processor

    # Merge every vocabulary spelling a checkpoint may carry; later files
    # override earlier ones for the same token text.
    vocabulary: dict[str, object] = {}
    if (root / "tokenizer.json").is_file():
        vocabulary.update(_tokenizer_vocabulary(root / "tokenizer.json"))
    for filename in ("vocab.json", "added_tokens.json"):
        if (root / filename).is_file():
            vocabulary.update(_json(root / filename))

    path = root / "tokenizer_config.json"
    if path.is_file():
        # ``added_tokens_decoder`` maps each decimal ID to an object whose
        # ``content`` is the token text.
        decoder = _json(path).get("added_tokens_decoder", {})
        if not isinstance(decoder, dict) or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("content"), str)
            for entry in decoder.values()
        ):
            raise ValueError(
                f"tokenizer metadata {path} has a malformed layout"
            )
        vocabulary.update(
            {entry["content"]: int(index) for index, entry in decoder.items()}
        )

    updates = {}
    for name, token in required.items():
        index = vocabulary.get(token)
        if type(index) is not int or index < 0:
            raise ValueError(
                f"tokenizer metadata does not define declared token {token!r}"
            )
        updates[name + "_id"] = index
    return replace(processor, feature_injection=replace(injection, **updates))


def _exclusions(model, declarations, sources, ignored, io):
    """Translate checkpoint exclusions through the same explicit assignments.

    Each entry of ``ignored`` names either a model module path or a checkpoint
    tensor prefix. The result maps every module path that directly owns an
    excluded parameter to ``None``, which leaves that module unquantized in
    ``uniserve.loading.weights.Config.quantization``. Opens each source in
    ``sources`` to read its assignments.

    Raises:
        ValueError: ``ignored`` is not a tuple or list of strings, or an entry
            matches neither a module path nor any assigned tensor.
    """
    if not isinstance(ignored, (tuple, list)) or any(
        not isinstance(name, str) for name in ignored
    ):
        raise ValueError("checkpoint ignored_layers must contain module paths")

    # One parameter can be reachable under several module paths (tied or
    # shared weights); excluding any alias must exclude the shared parameter
    # everywhere.
    aliases: dict[int, set[str]] = {}
    paths = dict(model.named_modules(remove_duplicate=False))
    for path, module in paths.items():
        for parameter in module.parameters(recurse=False):
            aliases.setdefault(id(parameter), set()).add(path)

    # Exclusions given as module paths match directly against the model tree.
    targets, matched = set(), set()
    for name in ignored:
        if name in paths:
            matched.add(name)
            for parameter in paths[name].parameters():
                targets.update(aliases[id(parameter)])

    # Exclusions given as checkpoint tensor prefixes match through each source's
    # weight assignments back to the target parameters they would load into.
    for source in sources:
        with source.open(io=io) as reader:
            for component in declarations:
                if component.source != source.name:
                    continue
                for assignment in component.map_weights(reader):
                    for name in ignored:
                        if assignment.source.name.startswith(name + "."):
                            matched.add(name)
                            targets.update(aliases[id(assignment.target)])

    if set(ignored).difference(matched):
        raise ValueError(
            f"checkpoint exclusions do not match numerical weights: "
            f"{sorted(set(ignored) - matched)}"
        )
    return dict.fromkeys(targets)


def _component_quantization(
    root: Path, declaration: checkpoint.Config
) -> dict | None:
    """Return the ModelOpt quantization a component folder declares.

    A diffusers pipeline records each component's quantization in that
    component's config.json. Only ModelOpt exports are accepted there;
    runtime quantization is a deployment choice, not a component property.

    Returns ``None`` for a source at the checkpoint root, whose quantization
    the root metadata declares, and for a component folder without a
    config.json or without a ``quantization_config``.

    Raises:
        ValueError: The component declares a non-ModelOpt quantization.
    """
    if not declaration.directory:
        return None
    path = root / declaration.directory / "config.json"
    if not path.is_file():
        return None
    declared = _json(path).get("quantization_config")
    if declared is None:
        return None
    if not isinstance(declared, dict) or declared.get("quant_method") != (
        "modelopt"
    ):
        raise ValueError(
            f"component {declaration.name} declares unsupported quantization "
            f"{declared!r}"
        )
    return declared


def _require_static_nvfp4(source: str, declared: Mapping) -> None:
    """Accept only ModelOpt's static NVFP4 W4A4 recipe with K16 blocks."""
    groups = declared.get("config_groups")
    expected = {"num_bits": 4, "type": "float", "group_size": 16}
    if (
        declared.get("quant_algo") != "NVFP4"
        or not isinstance(groups, dict)
        or not groups
        or any(
            not isinstance(group, dict)
            or any(
                not isinstance(group.get(role), dict)
                or group[role].get("dynamic", False)
                or any(
                    group[role].get(key) != value
                    for key, value in expected.items()
                )
                for role in ("weights", "input_activations")
            )
            for group in groups.values()
        )
    ):
        raise ValueError(
            f"ModelOpt source {source} must use static NVFP4 weights and "
            "activations with 16-element blocks"
        )


def _manifest_scales(root: Path, model: nn.Module) -> dict[str, float] | None:
    """Read native module calibration from a packed ModelOpt manifest.

    The manifest owns the W4A4 representation, including on ranks without a
    quantized component. Per-module amax is converted to ModelOpt's tensor
    scale; K16 block encoding remains a numerical layer operation.
    """
    path = root / "modelopt_manifest.json"
    if not path.is_file():
        return None
    manifest = _json(path)
    expected = {
        "activation": "a4",
        "block_scale": "fp8_e4m3",
        "block_size": 16,
        "output": "bf16",
        "tensor_scale": "fp32",
        "values": "e2m1",
        "weight": "w4",
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("numerical_format") != expected
    ):
        raise ValueError("ModelOpt manifest must declare static NVFP4 W4A4 K16")
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("ModelOpt manifest requires component calibration")
    paths = dict(model.named_modules(remove_duplicate=False))
    scales = {}
    for name, component in components.items():
        if not isinstance(component, dict) or not isinstance(
            component.get("enabled"), bool
        ):
            raise ValueError(f"invalid ModelOpt component {name!r}")
        modules = component.get("modules")
        if (
            not isinstance(modules, dict)
            or bool(modules) != component["enabled"]
        ):
            raise ValueError(f"invalid ModelOpt calibration for {name!r}")
        for path, calibration in modules.items():
            amax = (
                calibration.get("activation_amax")
                if isinstance(calibration, dict)
                else None
            )
            if (
                not isinstance(paths.get(path), Linear)
                or path in scales
                or not isinstance(amax, (int, float))
                or isinstance(amax, bool)
                or not math.isfinite(amax)
                or amax <= 0
            ):
                raise ValueError(
                    f"invalid ModelOpt activation calibration for {path!r}"
                )
            scales[path] = amax / (6 * 448)
    if not scales:
        raise ValueError("ModelOpt manifest declares no calibrated modules")
    return scales


def _calibrated_quantization(
    model, declarations, sources, declared, io, manifest_scales=None
):
    """Configure every Linear a ModelOpt NVFP4 export stores packed.

    The checkpoint tensors decide which modules are quantized: a module whose
    weight is stored as packed NVFP4 executes with that weight and its
    calibrated static activation scale; every other module stays dense. The
    activation's per-block K16 encoding is computed at run time against that
    fixed tensor scale.

    Args:
        model: Meta-device model whose module paths receive the result.
        declarations: The package's checkpoint mappings for ``model``.
        sources: Resolved sources; unified exports read those named in
            ``declared`` and manifest exports inspect every resolved source.
        declared: ModelOpt ``quantization_config`` per source name, each
            checked against ``_require_static_nvfp4``.
        io: Checkpoint IO policy for opening the sources.
        manifest_scales: Native module activation scales from a packed
            manifest, or ``None`` for unified exports with input_scale tensors.

    Returns:
        A ``QuantizationConfig`` for every module path that owns a packed
        weight, aliases included.

    Raises:
        ValueError: A declared recipe is not static NVFP4 with 16-element
            blocks, a packed weight lacks a positive finite input scale, maps
            onto a non-``Linear`` module or gives one module conflicting
            scales, or a read source stores no packed weight.
    """
    for name, value in declared.items():
        _require_static_nvfp4(name, value)

    paths = dict(model.named_modules(remove_duplicate=False))
    owners: dict[int, set[str]] = {}
    for path, module in paths.items():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), set()).add(path)

    scales: dict[str, float] = {}
    for source in sources:
        if manifest_scales is None and source.name not in declared:
            continue
        packed = 0
        with source.open(io=io) as reader:
            for component in declarations:
                if component.source != source.name:
                    continue
                for assignment in component.map_weights(reader):
                    weight = assignment.source
                    targets = owners[id(assignment.target)]
                    if not isinstance(weight, checkpoint.NVFP4Weight):
                        if manifest_scales is not None and any(
                            path in manifest_scales
                            and assignment.target is paths[path].weight
                            for path in targets
                        ):
                            raise ValueError(
                                f"ModelOpt calibrated weight {weight.name!r} "
                                "is not packed"
                            )
                        continue
                    stored_scale = weight.input_scale()
                    for path in targets:
                        value = (
                            stored_scale
                            if manifest_scales is None
                            else manifest_scales.get(path)
                        )
                        if (
                            value is None
                            or not math.isfinite(value)
                            or value <= 0
                        ):
                            raise ValueError(
                                f"ModelOpt weight {weight.name!r} requires a "
                                "positive static input_scale"
                            )
                        if stored_scale is not None and stored_scale != value:
                            raise ValueError(
                                f"{path} has conflicting activation calibration"
                            )
                        if isinstance(paths[path], ExpertLinear):
                            # Stacked experts share one activation encoding;
                            # its static scale is the largest calibrated
                            # expert scale, so no expert's input saturates.
                            scales[path] = max(scales.get(path, 0.0), value)
                            continue
                        if not isinstance(paths[path], Linear):
                            raise ValueError(
                                f"ModelOpt weight {weight.name!r} maps onto "
                                f"{path}, which is not a Linear"
                            )
                        if scales.setdefault(path, value) != value:
                            raise ValueError(
                                f"{path} receives conflicting calibrated "
                                "activation scales"
                            )
                    packed += 1
        if not packed and source.name in declared:
            raise ValueError(
                f"ModelOpt source {source.name} stores no packed NVFP4 weights"
            )
    return {
        path: QuantizationConfig(
            Quantizer("nvfp4"), Quantizer("nvfp4", calibrated_scale=value)
        )
        for path, value in scales.items()
    }


def _base_snapshot(repository: str, revision: str, io) -> _Snapshot:
    """Resolve the Hub snapshot of the base revision a checkpoint pins.

    The snapshot is read from the Hub cache and fetched into it when absent.

    Raises:
        FileNotFoundError: The revision publishes no model metadata.
    """
    root, repository, revision = _root(
        repository, replace(io, revision=revision)
    )
    return _Snapshot(
        root, repository, revision, _inventory(root, repository, revision, io)
    )


def _root_metadata(root: Path) -> dict:
    """Read the first root metadata file the checkpoint publishes."""
    for name in _metadata_files:
        if (root / name).is_file():
            return _json(root / name)
    raise FileNotFoundError(
        f"checkpoint {root} has no model metadata; expected one of "
        f"{', '.join(_metadata_files)}"
    )


def _architectures(metadata: Mapping) -> tuple[str, ...]:
    """Return the architectures a root metadata file declares.

    A pipeline index declares its pipeline class; a model configuration
    declares its architectures. A class name that is not a string, or
    architectures that are not a list of strings, raise ``ValueError``.
    """
    declared = (
        [metadata["_class_name"]]
        if "_class_name" in metadata
        else metadata.get("architectures", [])
    )
    if not isinstance(declared, list) or any(
        not isinstance(name, str) for name in declared
    ):
        raise ValueError(
            f"checkpoint must declare architectures as names; "
            f"found {declared!r}"
        )
    return tuple(declared)


def read_config(
    path: str | Path,
    *,
    io: loading.Config = loading.Config(),
    modules: frozenset[str] | None = None,
    exclude_modules: frozenset[str] = frozenset(),
) -> Config:
    """Normalize one local or immutable Hub snapshot without materializing a
    model.

    Architecture sidecars are read for the complete model. Payload downloads
    follow the selected numerical modules; shared aliases retain the same
    source. Tokenizer files remain paths until the caller loads a tokenizer.

    Args:
        path: A local checkpoint directory, a file inside one, or a Hub
            repository id, pinned to ``io.revision`` when it is set.
        io: Checkpoint IO policy. With ``mode="dummy"`` no payload of the
            selected modules is downloaded from the Hub, and the sources the
            package's ``config_sources`` names contribute only their
            safetensors headers (see ``_config_sources``).
        modules: Module paths to resolve sources for; ``None`` selects the
            whole model, and an empty set resolves no payload source.
        exclude_modules: Subtrees to leave unmaterialized, including their
            aliases. Their architecture metadata remains available.

    Raises:
        FileNotFoundError: The checkpoint, its metadata, its base revision,
            or a declared source's files cannot be found.
        ValueError: The checkpoint metadata or selection is invalid or
            unsupported, including an architecture outside the catalog, a
            selected path that is not a module, and malformed architecture,
            model configuration, index, tokenizer or quantization metadata
            (a required field that is missing or a field of the wrong type).
    """  # noqa: D205
    root, repository, revision = _root(path, io)
    metadata = _root_metadata(root)
    architectures = _architectures(metadata)
    if len(architectures) != 1 or architectures[0] not in _catalog:
        raise ValueError(
            f"checkpoint must declare one supported architecture; "
            f"found {architectures!r}"
        )
    package = import_module(_catalog[architectures[0]])

    inventory = _inventory(root, repository, revision, io)
    # All architecture/index/tokenizer sidecars are small and required to
    # normalize the same complete configuration on every rank, whatever its
    # module selection.
    sidecars = {
        name
        for name in inventory
        if Path(name).suffix in {".json", ".jinja", ".model", ".txt"}
        and not name.startswith(("optimizer/", "original/"))
    }
    _fetch(root, sidecars, repository, revision, io)

    # A checkpoint that pins a base revision reads the directories it omits
    # from it, whose sidecars are fetched like the checkpoint's own.
    export = _Snapshot(root, repository, revision, inventory)
    declares_base = hasattr(package, "base_checkpoint")
    pinned = package.base_checkpoint(root) if declares_base else None
    supplier = None
    if pinned is not None:
        base_repository, base_revision = pinned
        supplier = _base_snapshot(base_repository, base_revision, io)
    if supplier is not None:
        _fetch(
            supplier.root,
            {
                name
                for name in supplier.inventory
                if Path(name).suffix in {".json", ".jinja", ".model", ".txt"}
            },
            supplier.repository,
            supplier.revision,
            io,
        )

    held = {PurePosixPath(name).parts[0] for name in inventory}

    def located(directory: str) -> _Snapshot:
        parts = PurePosixPath(directory).parts
        if supplier is not None and parts and parts[0] not in held:
            return supplier
        return export

    # Some architectures derive dimensions from checkpoint tensor headers.
    # The package declares those sources before module selection is known.
    config_sources = _config_sources(package, located, io)

    # A meta-device skeleton resolves module selection and the mappings
    # without allocating weights. A source is needed when its mapping's
    # module is selected or when the mapping names any selected parameter as
    # required or optional, which covers parameters shared into an unselected
    # mapping's module.
    model_config = (
        package.read_config(
            root,
            io,
            sources=config_sources,
            base=None if supplier is None else supplier.root,
        )
        if declares_base
        else package.read_config(root, io, sources=config_sources)
    )
    with torch.device("meta"):
        model = package.Model(model_config)
    selected = loading.select_modules(
        model, modules, exclude_modules=exclude_modules
    )
    declarations = package.checkpoint_mappings(model)
    parameters = {
        id(parameter)
        for module in model.modules()
        if module in selected
        for parameter in module.parameters(recurse=False)
    }
    source_names = {
        component.source
        for component in declarations
        if component.module in selected
        or any(
            id(parameter) in parameters
            and name in component.required | component.optional
            for name, parameter in component.module.named_parameters(
                remove_duplicate=False
            )
        )
    }
    sources = []
    for declaration in package.checkpoint_sources:
        if declaration.name not in source_names:
            continue
        snapshot = located(declaration.directory)
        if snapshot.repository is not None and io.mode != "dummy":
            _fetch(
                snapshot.root,
                _source_files(
                    declaration, snapshot.root, snapshot.inventory, io
                ),
                snapshot.repository,
                snapshot.revision,
                io,
            )
        sources.append(declaration.resolve(snapshot.root, io=io))

    processor = (
        None
        if package.image_processor is None
        else package.image_processor(model_config)
    )
    processor = _tokens(processor, root)
    tokenizer_root = located("tokenizer").root
    tokenizer_root = (
        tokenizer_root / "tokenizer"
        if (tokenizer_root / "tokenizer").is_dir()
        else root
    )
    tokenizer = (
        tokenizer_root
        if any(
            (tokenizer_root / filename).is_file()
            for filename in (
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "vocab.json",
                "spiece.model",
            )
        )
        else None
    )

    # The package's "default" preset, else its "bf16" preset, else the
    # ``weights.Config`` defaults; the quantization metadata below refines
    # this base or, for a ModelOpt export, replaces it.
    precisions = package.precisions(model_config)
    precision = precisions.get(
        "default", precisions.get("bf16", weight_options.Config())
    )
    checkpoint_format = None
    quantization = metadata.get("quantization_config")
    if quantization is not None and not isinstance(quantization, dict):
        raise ValueError("checkpoint quantization_config must be an object")

    # A unified ModelOpt export declares `quant_method: modelopt` in the root
    # configuration of a single-model checkpoint, or in each quantized
    # component's config.json of a diffusers pipeline.
    modelopt = {}
    if quantization is not None and (
        quantization.get("quant_method") == "modelopt"
    ):
        modelopt = {
            declaration.name: quantization
            for declaration in package.checkpoint_sources
        }
        quantization = None

    # Every rank of the checkpoint agrees that it is calibrated, including a
    # rank that loads none of the quantized components; only the resolved
    # sources contribute calibrated modules.
    for declaration in package.checkpoint_sources:
        declared = _component_quantization(
            located(declaration.directory).root, declaration
        )
        if declared is not None:
            modelopt[declaration.name] = declared
    # Packed exports instead publish native module calibration in a root
    # manifest. Its metadata is fetched with the architecture sidecars.
    manifest_scales = _manifest_scales(root, model)
    if modelopt and manifest_scales is not None:
        raise ValueError(
            "checkpoint declares both unified and manifest ModelOpt calibration"
        )
    if (modelopt or manifest_scales is not None) and quantization is not None:
        raise ValueError(
            "a calibrated ModelOpt checkpoint cannot also declare a "
            "dynamic quantization_config"
        )
    if modelopt or manifest_scales is not None:
        # Packed weights and their calibrated scales form one immutable
        # checkpoint contract. Runtime precision presets apply only to dense
        # checkpoints and must not be offered for this source.
        checkpoint_precision: weight_options.Config = (
            package.checkpoint_precision(model_config)
        )
        precision = replace(
            checkpoint_precision,
            quantization={
                **checkpoint_precision.quantization,
                **_calibrated_quantization(
                    model, declarations, sources, modelopt, io, manifest_scales
                ),
            },
        )
        precisions = MappingProxyType({})
        checkpoint_format = "modelopt_nvfp4"

    # A non-ModelOpt root quantization_config with a supported method other
    # than "unquantized" applies one quantizer to the weights and activations
    # of every Linear (the empty prefix matches every module path).
    # Independently of the method, its ignored_layers map the listed modules
    # to no quantization.
    if quantization is not None:
        method = quantization.get("quant_method", "unquantized")
        if method != "unquantized":
            formats: dict[str, Literal["fp8", "mxfp8", "nvfp4"]] = {
                "fp8": "fp8",
                "mxfp8": "mxfp8",
                "nvfp4": "nvfp4",
            }
            quant_format = (
                formats.get(method) if isinstance(method, str) else None
            )
            if quant_format is None:
                raise ValueError(
                    f"unsupported checkpoint quantization {method!r}"
                )
            quantizer = Quantizer(
                quant_format, axis=0 if quant_format == "fp8" else None
            )
            precision = replace(
                precision,
                quantization={"": QuantizationConfig(quantizer, quantizer)},
            )
        ignored = quantization.get("ignored_layers", ())
        if ignored:
            precision = replace(
                precision,
                quantization={
                    **precision.quantization,
                    **_exclusions(model, declarations, sources, ignored, io),
                },
            )

    return Config(
        model_config,
        package.Model,
        tuple(sources),
        package.checkpoint_mappings,
        package.entry_points(model_config),
        precision,
        precisions,
        checkpoint_format,
        io,
        tokenizer,
        processor,
        package.flow_prompt,
        modules,
        exclude_modules,
    )


def load_model(
    config: Config[ConfigT, ModelT],
    *,
    device: torch.device | str,
    precision: str | None = None,
    weights: weight_options.Config | None = None,
    meshes: Mapping[str, DeviceMesh] | None = None,
    attention: Mapping[str, AttentionParallelConfig] | None = None,
    devices: Mapping[str, torch.device | str] | None = None,
    modules: frozenset[str] | None = None,
    exclude_modules: frozenset[str] = frozenset(),
    experts: Communicator | None = None,
) -> loading.Result[ModelT]:
    """Materialize the selected capability modules through the common loader.

    ``precision`` names one of ``config.precisions`` and ``weights`` supplies
    a weight configuration directly; with neither, ``config.weights``
    applies. ``modules`` may narrow ``config.modules`` but not widen it.
    ``exclude_modules`` adds to the exclusions already resolved in ``config``.
    The remaining arguments pass through to ``uniserve.loading.load_model``.

    Raises:
        ValueError: Both ``precision`` and ``weights`` are given, the
            precision is unknown, or the selection exceeds ``config.modules``.
    """
    if precision is not None and weights is not None:
        raise ValueError(
            "precision and weights are mutually exclusive numerical choices"
        )
    if precision is not None:
        if precision not in config.precisions:
            raise ValueError(
                f"unknown precision {precision!r}; "
                f"choose from {tuple(config.precisions)}"
            )
        weights = config.precisions[precision]

    # A caller may narrow the resolved module selection but never widen it:
    # read_config fetched payloads only for the resolved selection.
    selected = config.modules if modules is None else modules
    excluded = config.exclude_modules | frozenset(exclude_modules)
    if config.modules is not None or config.exclude_modules:
        with torch.device("meta"):
            model = config.model_class(config.model)
        if not loading.select_modules(
            model, selected, exclude_modules=excluded
        ).issubset(
            loading.select_modules(
                model, config.modules, exclude_modules=config.exclude_modules
            )
        ):
            raise ValueError(
                "load selection exceeds the modules resolved by read_config"
            )

    return loading.load_model(
        config.model_class,
        config.model,
        checkpoint=config.checkpoint,
        mapping=config.mapping,
        device=device,
        weights=config.weights if weights is None else weights,
        io=config.io,
        meshes=meshes,
        attention=attention,
        devices=devices,
        modules=selected,
        exclude_modules=excluded,
        experts=experts,
    )
