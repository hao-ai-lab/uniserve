"""Discover concrete models and normalize their public loading inputs."""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint
from uniserve.loading import weights as weight_options
from uniserve.model import EntryPoint
from uniserve.nn.attention import AttentionParallelConfig
from uniserve.quantization import QuantizationConfig, Quantizer

from .processing import FlowPrompt, ImageProcessor

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT", bound=nn.Module)

_catalog: Mapping[str, str] = MappingProxyType(
    {
        "Qwen3ForCausalLM": "uniserve_models.qwen3",
        "Qwen3MoeForCausalLM": "uniserve_models.qwen3",
        "BagelForConditionalGeneration": "uniserve_models.bagel",
        "NEOChatModel": "uniserve_models.sensenova_u1",
        "MiniMaxH3Transformer3DModel": "uniserve_models.minimax_h3",
    }
)


@dataclass(frozen=True, slots=True)
class Config(Generic[ConfigT, ModelT]):
    """Resolved architecture, closed checkpoint sources, and caller-owned assets.

    No tokenizer, reader, resource context, or model instance is retained.
    Module selection uses actual module paths and also selects shared
    descendants by identity. Loading can narrow an already resolved selection.
    """

    model: ConfigT
    model_class: Callable[[ConfigT], ModelT]
    checkpoint: tuple[checkpoint.Source, ...]
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]]
    entry_points: Mapping[str, tuple[EntryPoint, ...]]
    weights: weight_options.Config
    precisions: Mapping[str, weight_options.Config]
    io: loading.Config
    tokenizer: Path | None
    image_processor: ImageProcessor | None
    flow_prompt: FlowPrompt | None
    modules: frozenset[str] | None

    def __post_init__(self):
        object.__setattr__(
            self,
            "entry_points",
            MappingProxyType({path: tuple(entries) for path, entries in self.entry_points.items()}),
        )
        object.__setattr__(self, "precisions", MappingProxyType(dict(self.precisions)))
        if self.modules is not None:
            object.__setattr__(self, "modules", frozenset(self.modules))


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint metadata {path} must contain an object")
    return value


def _root(path: str | Path, io: loading.Config):
    """Resolve a local checkpoint directory or pin a Hub snapshot to a local snapshot root."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return (candidate.parent if candidate.is_file() else candidate), None, None
    if isinstance(path, Path) or candidate.is_absolute():
        raise FileNotFoundError(candidate)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in ("config.json", "modular_model_index.json"):
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
    raise FileNotFoundError(f"checkpoint {path!r} has no supported model metadata")


def _inventory(root, repository, revision, io):
    """List checkpoint files, excluding caller-configured ignore patterns."""
    if repository is None:
        names = (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    else:
        from huggingface_hub import HfApi

        names = HfApi().list_repo_files(repo_id=repository, revision=revision)

    return frozenset(
        name
        for name in names
        if not any(
            fnmatch.fnmatchcase(name, pattern) or PurePosixPath(name).match(pattern)
            for pattern in io.ignore_patterns
        )
    )


def _fetch(root, names, repository, revision, io):
    if repository is None:
        return
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def fetch(name):
        hf_hub_download(
            repo_id=repository, filename=name, revision=revision, cache_dir=io.download_dir
        )

    with ThreadPoolExecutor(max_workers=io.num_threads or 8) as executor:
        tuple(executor.map(fetch, sorted(names)))


def _source_files(source, root, inventory, io):
    """Choose the declared file encoding before downloading checkpoint payloads."""
    directory = PurePosixPath(source.directory)
    available = {name for name in inventory if PurePosixPath(name).parent == directory}
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
        raise FileNotFoundError(f"source {source.name!r} requires one of {source.filenames!r}")

    for format in formats:
        pattern = "*.safetensors.index.json" if format == "safetensors" else "*.bin.index.json"
        indexes = sorted(
            name for name in available if fnmatch.fnmatchcase(PurePosixPath(name).name, pattern)
        )
        if len(indexes) > 1:
            raise ValueError(f"source {source.name!r} has multiple {format} indexes")

        if indexes:
            # A sharded index declares its payload files relative to the source directory.
            mapping = _json(root / indexes[0]).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("checkpoint index requires a nonempty weight_map")
            paths = tuple(PurePosixPath(name) for name in set(mapping.values()))
            if any(path.is_absolute() or ".." in path.parts for path in paths):
                raise ValueError("checkpoint index paths must stay inside their directory")
            names = tuple((directory / path).as_posix() for path in paths)
            if set(names).difference(inventory):
                raise FileNotFoundError("checkpoint index references a missing or excluded shard")
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

    raise FileNotFoundError(f"source {source.name!r} has no {io.format} checkpoint weights")


def _tokens(processor, root):
    """Fill declared start/end feature-injection token IDs from checkpoint tokenizer files."""
    if processor is None or processor.feature_injection is None:
        return processor
    injection = processor.feature_injection
    required = {
        name: getattr(injection, name)
        for name in ("start_token", "end_token")
        if getattr(injection, name) is not None and getattr(injection, name + "_id") is None
    }
    if not required:
        return processor

    # Merge every vocabulary spelling a checkpoint may carry; later files
    # override earlier ones for the same token text.
    vocabulary = {}
    path = root / "tokenizer.json"
    if path.is_file():
        data = _json(path)
        raw_vocab = data.get("model", {}).get("vocab", {})
        vocabulary.update(
            raw_vocab
            if isinstance(raw_vocab, dict)
            else {row[0]: i for i, row in enumerate(raw_vocab)}
        )
        vocabulary.update({entry["content"]: entry["id"] for entry in data.get("added_tokens", ())})
    for filename in ("vocab.json", "added_tokens.json"):
        if (root / filename).is_file():
            vocabulary.update(_json(root / filename))
    if (root / "tokenizer_config.json").is_file():
        vocabulary.update(
            {
                entry["content"]: int(index)
                for index, entry in _json(root / "tokenizer_config.json")
                .get("added_tokens_decoder", {})
                .items()
            }
        )

    updates = {}
    for name, token in required.items():
        index = vocabulary.get(token)
        if type(index) is not int or index < 0:
            raise ValueError(f"tokenizer metadata does not define declared token {token!r}")
        updates[name + "_id"] = index
    return replace(processor, feature_injection=replace(injection, **updates))


def _selection(model, modules):
    if modules is None:
        return {id(child) for child in model.modules()}
    try:
        return {id(child) for path in modules for child in model.get_submodule(path).modules()}
    except AttributeError as error:
        raise ValueError("selected modules must be actual model module paths") from error


def _exclusions(model, declarations, sources, ignored, io):
    """Translate checkpoint exclusions through the same explicit assignments."""
    if not isinstance(ignored, (tuple, list)) or any(not isinstance(name, str) for name in ignored):
        raise ValueError("checkpoint ignored_layers must contain module paths")

    # One parameter can be reachable under several module paths (tied or shared
    # weights); excluding any alias must exclude the shared parameter everywhere.
    aliases = {}
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
            f"checkpoint exclusions do not match numerical weights: {sorted(set(ignored) - matched)}"
        )
    return {name: None for name in targets}


def read_config(
    path: str | Path,
    *,
    io: loading.Config = loading.Config(),
    modules: frozenset[str] | None = None,
) -> Config:
    """Normalize one local or immutable Hub snapshot without materializing a model.

    Architecture sidecars are read for the complete model. Payload downloads
    follow the selected numerical modules; shared aliases retain the same
    source. Tokenizer files remain paths until the caller loads a tokenizer.
    """
    root, repository, revision = _root(path, io)
    metadata = (
        _json(root / "config.json")
        if (root / "config.json").is_file()
        else _json(root / "modular_model_index.json")
    )
    architectures = metadata.get("architectures", ())
    if metadata.get("_class_name") == "MiniMaxH3ModularPipeline":
        architectures = ("MiniMaxH3Transformer3DModel",)
    if len(architectures) != 1 or architectures[0] not in _catalog:
        raise ValueError(
            f"checkpoint must declare one supported architecture; found {architectures!r}"
        )
    architecture = architectures[0]
    package = import_module(_catalog[architecture])

    inventory = _inventory(root, repository, revision, io)
    # All architecture/index/tokenizer sidecars are small and required to
    # normalize the same complete configuration on participating and remote ranks.
    sidecars = {
        name
        for name in inventory
        if Path(name).suffix in {".json", ".jinja", ".model", ".txt"}
        and not name.startswith(("optimizer/", "original/"))
    }
    _fetch(root, sidecars, repository, revision, io)
    if repository is not None and io.mode != "dummy":
        # Some architectures derive dimensions from checkpoint tensor headers.
        # The package declares those sources before module selection is known.
        for declaration in package.config_sources:
            _fetch(
                root,
                _source_files(declaration, root, inventory, io),
                repository,
                revision,
                io,
            )

    model_config = package.read_config(root, io)
    with torch.device("meta"):
        model = package.Model(model_config)
    selected = _selection(model, modules)
    declarations = package.checkpoint_mappings(model)
    parameters = {
        id(parameter)
        for module in model.modules()
        if id(module) in selected
        for parameter in module.parameters(recurse=False)
    }
    source_names = {
        component.source
        for component in declarations
        if id(component.module) in selected
        or any(
            id(parameter) in parameters and name in component.required | component.optional
            for name, parameter in component.module.named_parameters(remove_duplicate=False)
        )
    }
    sources = []
    for declaration in package.checkpoint_sources:
        if declaration.name not in source_names:
            continue
        if repository is not None and io.mode != "dummy":
            _fetch(root, _source_files(declaration, root, inventory, io), repository, revision, io)
        sources.append(declaration.resolve(root, io=io))

    processor = None if package.image_processor is None else package.image_processor(model_config)
    processor = _tokens(processor, root)
    tokenizer_root = root / "tokenizer" if (root / "tokenizer").is_dir() else root
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

    precision = package.precisions.get(
        "default", package.precisions.get("bf16", weight_options.Config())
    )
    quantization = metadata.get("quantization_config")
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise ValueError("checkpoint quantization_config must be an object")
        method = quantization.get("quant_method", "unquantized")
        if method != "unquantized":
            if method not in {"fp8", "mxfp8", "nvfp4"}:
                raise ValueError(f"unsupported checkpoint quantization {method!r}")
            quantizer = Quantizer(method, axis=0 if method == "fp8" else None)
            precision = replace(
                precision, quantization={"": QuantizationConfig(quantizer, quantizer)}
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
        package.precisions,
        io,
        tokenizer,
        processor,
        package.flow_prompt,
        modules,
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
) -> loading.Result[ModelT]:
    """Materialize the selected capability modules through the common loader."""
    if precision is not None and weights is not None:
        raise ValueError("precision and weights are mutually exclusive numerical choices")
    if precision is not None:
        if precision not in config.precisions:
            raise ValueError(
                f"unknown precision {precision!r}; choose from {tuple(config.precisions)}"
            )
        weights = config.precisions[precision]

    # A caller may narrow the resolved module selection but never widen it:
    # read_config fetched payloads only for the resolved selection.
    selected = config.modules if modules is None else modules
    if config.modules is not None and selected is not None:
        with torch.device("meta"):
            model = config.model_class(config.model)
        if not _selection(model, selected).issubset(_selection(model, config.modules)):
            raise ValueError("load selection exceeds the modules resolved by read_config")

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
    )
