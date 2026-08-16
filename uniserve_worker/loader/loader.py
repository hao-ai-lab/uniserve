"""Registered model loaders and the canonical construction pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

import torch
from torch import nn

from ..foundation.errors import capability_mismatch
from ..nn.layer import LayerSpec
from ..nn.quant import QuantizationConfig
from ..nn.quant.base import process_quantized_modules
from .audit import audit_load_report
from .config import LoadFormat, LoadRequest
from .handles import TensorWeightHandle, WeightHandle, weight_handle_materialization
from .io import iter_weight_handles
from .mapping import LoadReport
from .source import WeightSourceSet, resolve_weight_sources
from .weight_loaders import (
    DeferredWeightPlacement,
    attach_parameter_loaders,
    default_weight_loader,
    defer_parameter_weights,
)
from .weight_set import WeightSet, dummy_weight_digest, source_weight_digest

logger = logging.getLogger(__name__)

__all__ = [
    "BaseModelLoader",
    "DefaultModelLoader",
    "LoadedModel",
    "get_model_loader",
]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: nn.Module
    tokenizer: Any | None
    device: str
    weights: WeightSet
    sources: tuple[WeightSourceSet, ...]
    architecture_config: dict[str, Any]


class BaseModelLoader(ABC):
    @abstractmethod
    def load(
        self,
        entry: Any,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    def load(
        self,
        entry: Any,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        sources = resolve_weight_sources(
            request,
            architecture=entry.architecture,
            sidecars=entry.sidecars,
            root=root,
            repository_id=repository_id,
        )
        _verify_checksums(sources, request.load.checksum_manifest)
        prepared = _prepare_architecture_config(entry.architecture, config, root, sources)
        model, tokenizer = _construct_model(entry, prepared, request, root)
        report = _load_primary(model, entry.architecture, sources[0], request)
        _materialize_scope_buffers(model, report.loaded, request.device)
        _audit_primary(model, entry.architecture, report)
        if len(sources) > 1:
            _load_secondary(model, entry.architecture, sources[1:], request)
        _warn_skips(entry.architecture, report)
        _process_quantization_for_scope(model, entry.architecture, report.loaded)
        model.eval()
        digest = source_weight_digest(
            entry.architecture,
            request.scope.value,
            request.load.load_format,
            sources,
        )
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=request.device,
            weights=WeightSet.from_module(model, digest=digest),
            sources=sources,
            architecture_config=_canonical_architecture_config(prepared),
        )


class DummyModelLoader(BaseModelLoader):
    def load(
        self,
        entry: Any,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        del repository_id
        sources = (WeightSourceSet(root, (), ()),)
        prepared = _prepare_architecture_config(entry.architecture, config, root, sources)
        model, tokenizer = _construct_model(entry, prepared, request, root)
        included = (
            getattr(model, "checkpoint_parameter_names")()
            if entry.architecture == "NEOChatModel"
            else {name for name, _ in model.named_parameters()}
        )
        loaded: set[str] = set()
        with torch.no_grad():
            for name in sorted(included):
                current = dict(model.named_parameters()).get(name)
                if current is None:
                    continue
                seed = int.from_bytes(
                    hashlib.sha256(f"{entry.architecture}\0{name}".encode()).digest()[:8],
                    "little",
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                value = torch.empty(
                    tuple(current.shape),
                    dtype=current.dtype,
                    device="cpu",
                )
                if current.is_floating_point():
                    value.normal_(mean=0.0, std=0.02, generator=generator)
                else:
                    value.zero_()
                default_weight_loader(current, TensorWeightHandle(name, value))
                loaded.add(name)
        _materialize_scope_buffers(model, loaded, request.device)
        _zero_dummy_vocab_padding(model, loaded)
        _process_loaded_quantization(model, loaded)
        model.eval()
        digest = dummy_weight_digest(
            entry.architecture,
            request.scope.value,
            request.execution.model_dtype,
            model,
        )
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=request.device,
            weights=WeightSet.from_module(model, digest=digest),
            sources=sources,
            architecture_config=_canonical_architecture_config(prepared),
        )


class ShardedStateLoader(BaseModelLoader):
    def load(
        self,
        entry: Any,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        sources = resolve_weight_sources(
            request,
            architecture=entry.architecture,
            sidecars=entry.sidecars,
            root=root,
            repository_id=repository_id,
        )
        _verify_checksums(sources, request.load.checksum_manifest)
        prepared = _prepare_architecture_config(entry.architecture, config, root, sources)
        model, tokenizer = _construct_model(entry, prepared, request, root)
        report = LoadReport()
        parameters = dict(model.named_parameters())
        names = set(parameters)
        included = (
            set(getattr(model, "checkpoint_parameter_names")())
            if entry.architecture == "NEOChatModel"
            else names
        )
        for handle in iter_weight_handles(sources[0], request.load):
            if handle.name not in names:
                report.unexpected.append(handle.name)
                continue
            if handle.name not in included:
                continue
            parameter = parameters[handle.name]
            default_weight_loader(parameter, handle)
            report.loaded.add(handle.name)
        _materialize_scope_buffers(model, report.loaded, request.device)
        audit_load_report(
            model,
            report,
            included=included,
            label="sharded-state checkpoint",
            require_packed_shards=False,
        )
        _process_quantization_for_scope(model, entry.architecture, report.loaded)
        model.eval()
        digest = source_weight_digest(
            entry.architecture,
            request.scope.value,
            request.load.load_format,
            sources,
        )
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=request.device,
            weights=WeightSet.from_module(model, digest=digest),
            sources=sources,
            architecture_config=_canonical_architecture_config(prepared),
        )


class LayeredModelLoader(BaseModelLoader):
    """Materialize and finalize one architecture-resolved module subtree at a time."""

    def load(
        self,
        entry: Any,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        sources = resolve_weight_sources(
            request,
            architecture=entry.architecture,
            sidecars=entry.sidecars,
            root=root,
            repository_id=repository_id,
        )
        _verify_checksums(sources, request.load.checksum_manifest)
        prepared = _prepare_architecture_config(entry.architecture, config, root, sources)
        model, tokenizer = _construct_model(entry, prepared, request, root)
        report = _load_layered_primary(model, entry.architecture, sources[0], request)
        _materialize_scope_buffers(model, report.loaded, request.device)
        _audit_primary(model, entry.architecture, report)
        if len(sources) > 1:
            _load_layered_secondary(model, entry.architecture, sources[1:], request)
        _warn_skips(entry.architecture, report)
        model.eval()
        digest = source_weight_digest(
            entry.architecture,
            request.scope.value,
            request.load.load_format,
            sources,
        )
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=request.device,
            weights=WeightSet.from_module(model, digest=digest),
            sources=sources,
            architecture_config=_canonical_architecture_config(prepared),
        )


_LOADERS: dict[LoadFormat, type[BaseModelLoader]] = {
    LoadFormat.AUTO: DefaultModelLoader,
    LoadFormat.SAFETENSORS: DefaultModelLoader,
    LoadFormat.PT: DefaultModelLoader,
    LoadFormat.DUMMY: DummyModelLoader,
    LoadFormat.SHARDED_STATE: ShardedStateLoader,
    LoadFormat.LAYERED: LayeredModelLoader,
}


def get_model_loader(load_format: LoadFormat | str) -> BaseModelLoader:
    try:
        selected = LoadFormat(str(load_format))
        loader = _LOADERS[selected]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unregistered model load format {load_format!r}") from error
    return loader()


def _construct_model(
    entry: Any,
    config: Any,
    request: LoadRequest,
    root: Path,
) -> tuple[nn.Module, Any | None]:
    dtype = _serving_dtype(request.execution.model_dtype)
    quantization = QuantizationConfig.from_model_config(config)
    _validate_quantization(quantization, request.device, dtype)
    spec = LayerSpec(parallel=request.parallel, quantization=quantization)
    use_meta = (
        request.load.load_format is LoadFormat.LAYERED
        or (
            entry.architecture == "NEOChatModel"
            and request.scope.value != "whole"
        )
    )
    construction_device = torch.device("meta" if use_meta else request.device)
    with _default_dtype(dtype), torch.device(construction_device):
        if entry.architecture == "NEOChatModel":
            model = entry.model_class(config, layer_spec=spec, scope=request.scope.value)
        else:
            model = entry.model_class(config, layer_spec=spec)
    if not isinstance(model, nn.Module):
        raise capability_mismatch("catalog model constructor did not return torch.nn.Module")
    attach_parameter_loaders(model, device=request.device, dtype=dtype)
    tokenizer = None
    if entry.architecture == "NEOChatModel":
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                root,
                use_fast=False,
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to load the configured SenseNova tokenizer from {str(root)!r}: {error}"
            ) from error
    return model, tokenizer


def _prepare_architecture_config(
    architecture: str,
    config: dict[str, Any],
    root: Path,
    sources: tuple[WeightSourceSet, ...],
) -> Any:
    if architecture == "NEOChatModel":
        from ..models.sensenova.config import NeoChatConfig
        from ..models.sensenova.model import _check_checkpoint_code_version

        resolved = NeoChatConfig.from_dict(config)
        _check_checkpoint_code_version(resolved)
        return resolved
    if architecture != "BagelForConditionalGeneration":
        return config
    from ..models.bagel import BagelConfig

    raw = dict(config)
    for field, filename in (
        ("llm_config", "llm_config.json"),
        ("vit_config", "vit_config.json"),
        ("vae_config", "vae_config.json"),
    ):
        if field in raw:
            continue
        path = root / filename
        if not path.is_file():
            raise capability_mismatch(f"BAGEL checkpoint is missing {filename!r} for {field!r}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise capability_mismatch(f"BAGEL checkpoint file {filename!r} must contain an object")
        raw[field] = value
    positions = sources[0].preview_shape("latent_pos_embed.pos_embed")[0]
    max_latent_size = math.isqrt(positions)
    if max_latent_size * max_latent_size != positions:
        raise capability_mismatch(f"BAGEL latent position count {positions} is not square")
    raw["max_latent_size"] = max_latent_size
    return BagelConfig.from_mapping(raw)


def _load_primary(
    model: nn.Module,
    architecture: str,
    source: WeightSourceSet,
    request: LoadRequest,
) -> LoadReport:
    load_weights = getattr(model, "load_weights", None)
    if not callable(load_weights):
        raise capability_mismatch(f"{architecture} must implement load_weights")
    report = load_weights(iter_weight_handles(source, request.load))
    if not isinstance(report, LoadReport):
        raise capability_mismatch(f"{architecture}.load_weights must return LoadReport")
    return report


def _load_layered_primary(
    model: nn.Module,
    architecture: str,
    source: WeightSourceSet,
    request: LoadRequest,
) -> LoadReport:
    return _load_layered_weights(
        model,
        architecture,
        iter_weight_handles(source, request.load),
    )


def _load_layered_weights(
    model: nn.Module,
    architecture: str,
    handles: Iterable[WeightHandle],
) -> LoadReport:
    load_weights = getattr(model, "load_weights", None)
    if not callable(load_weights):
        raise capability_mismatch(f"{architecture} must implement load_weights")
    with defer_parameter_weights() as placements:
        report = load_weights(handles)
    if not isinstance(report, LoadReport):
        raise capability_mismatch(f"{architecture}.load_weights must return LoadReport")
    _materialize_layered_placements(model, placements)
    return report


def _materialize_layered_placements(
    model: nn.Module,
    placements: list[DeferredWeightPlacement],
) -> None:
    owners: dict[int, tuple[str, nn.Module]] = {}
    ordered_modules = tuple(model.named_modules())
    for module_name, module in ordered_modules:
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), (module_name, module))
    grouped: dict[str, list[DeferredWeightPlacement]] = defaultdict(list)
    for placement in placements:
        try:
            owner_name, _owner = owners[id(placement.parameter)]
        except KeyError as error:
            raise RuntimeError("deferred checkpoint placement has no model owner") from error
        grouped[owner_name].append(placement)
    for module_name, module in ordered_modules:
        unit = grouped.get(module_name)
        if not unit:
            continue
        with torch.no_grad(), weight_handle_materialization():
            for placement in unit:
                placement.apply()
        process_quantized_modules((module,))


def _audit_primary(model: nn.Module, architecture: str, report: LoadReport) -> None:
    if architecture == "NEOChatModel":
        included = getattr(model, "checkpoint_parameter_names")()
        audit_load_report(model, report, included=included, label="SenseNova checkpoint")
        return
    if architecture == "BagelForConditionalGeneration":
        target = getattr(model, "model")
        included = getattr(model, "checkpoint_parameter_names")()
        audit_load_report(target, report, included=included, label="BAGEL checkpoint")
        return
    audit_load_report(model, report, label="Qwen3 checkpoint")


def _load_secondary(
    model: nn.Module,
    architecture: str,
    sources: tuple[WeightSourceSet, ...],
    request: LoadRequest,
) -> None:
    if architecture != "BagelForConditionalGeneration" or len(sources) != 1:
        raise capability_mismatch(f"{architecture} does not declare these secondary sources")
    load_autoencoder = getattr(model, "load_autoencoder_weights", None)
    if not callable(load_autoencoder):
        raise capability_mismatch("BAGEL must implement load_autoencoder_weights")
    report = load_autoencoder(iter_weight_handles(sources[0], request.load))
    target = getattr(getattr(model, "model"), "vae")
    optional = _bagel_vae_optional(target)
    audit_load_report(
        target,
        report,
        optional=optional,
        label="BAGEL autoencoder checkpoint",
    )
    _warn_skips("BAGEL autoencoder", report)


def _load_layered_secondary(
    model: nn.Module,
    architecture: str,
    sources: tuple[WeightSourceSet, ...],
    request: LoadRequest,
) -> None:
    if architecture != "BagelForConditionalGeneration" or len(sources) != 1:
        raise capability_mismatch(f"{architecture} does not declare these secondary sources")
    load_autoencoder = getattr(model, "load_autoencoder_weights", None)
    if not callable(load_autoencoder):
        raise capability_mismatch("BAGEL must implement load_autoencoder_weights")
    target = getattr(getattr(model, "model"), "vae")
    with defer_parameter_weights() as placements:
        report = load_autoencoder(iter_weight_handles(sources[0], request.load))
    if not isinstance(report, LoadReport):
        raise capability_mismatch("BAGEL autoencoder loader must return LoadReport")
    _materialize_layered_placements(target, placements)
    optional = _bagel_vae_optional(target)
    audit_load_report(
        target,
        report,
        optional=optional,
        label="BAGEL autoencoder checkpoint",
    )
    _warn_skips("BAGEL autoencoder", report)


def _bagel_vae_optional(module: nn.Module) -> set[str]:
    return {
        name
        for name, _ in module.named_parameters()
        if name == "reg" or name.startswith("reg.")
    }


def _verify_checksums(
    sources: tuple[WeightSourceSet, ...],
    manifest_location: str | None,
) -> None:
    if manifest_location is None:
        return
    if manifest_location.startswith(("http://", "https://")):
        with urlopen(manifest_location) as response:  # noqa: S310 - explicit configured URI
            payload = response.read().decode("utf-8")
    else:
        payload = Path(manifest_location).read_text(encoding="utf-8")
    value = json.loads(payload)
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if not isinstance(value, dict):
        raise TypeError("checksum manifest must contain a relative-path mapping")
    for source in sources:
        for relative, path in zip(source.relative_paths, source.weight_files):
            expected = value.get(relative)
            if not isinstance(expected, str):
                raise ValueError(f"checksum manifest has no digest for {relative!r}")
            actual = _file_sha256(path)
            if actual.lower() != expected.lower().removeprefix("sha256:"):
                raise ValueError(f"checksum mismatch for checkpoint file {relative!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _warn_skips(architecture: str, report: LoadReport) -> None:
    if report.skipped:
        logger.warning(
            "%s declared %d checkpoint tensors outside its load contract",
            architecture,
            len(report.skipped),
        )


def _process_loaded_quantization(model: nn.Module, loaded: set[str]) -> None:
    selected: list[nn.Module] = []
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if any(name.startswith(prefix) for name in loaded):
            selected.append(module)
    process_quantized_modules(selected)


def _materialize_scope_buffers(
    model: nn.Module,
    loaded: set[str],
    device: str,
) -> None:
    active_modules: set[str] = set()
    for name in loaded:
        parts = name.split(".")[:-1]
        active_modules.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    for module_name, module in model.named_modules():
        parts = module_name.split(".")
        if not any(
            ".".join(parts[:end]) in active_modules
            for end in range(1, len(parts) + 1)
        ):
            continue
        materialize = getattr(module, "materialize_load_buffers", None)
        if callable(materialize):
            materialize(device)


def _process_quantization_for_scope(
    model: nn.Module,
    architecture: str,
    loaded: set[str],
) -> None:
    if architecture == "NEOChatModel":
        _process_loaded_quantization(model, loaded)
    else:
        process_quantized_modules(model.modules())


def _zero_dummy_vocab_padding(model: nn.Module, loaded: set[str]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in loaded:
                continue
            layout = getattr(parameter, "_uniserve_vocab_layout", None)
            if not isinstance(layout, tuple) or len(layout) != 3:
                continue
            real_size, start, _end = (int(value) for value in layout)
            padding_start = max(0, real_size - start)
            if padding_start < int(parameter.shape[0]):
                parameter[padding_start:].zero_()


def _canonical_architecture_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(config):
        value = asdict(cast(Any, config))
    else:
        raise TypeError("load-time architecture configuration is not serializable")
    if not isinstance(value, dict):
        raise TypeError("load-time architecture configuration must serialize to an object")
    return value


def _serving_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown model dtype {name!r}; expected bfloat16, float16, or float32"
        ) from error


def _validate_quantization(
    config: QuantizationConfig | None,
    device: str,
    dtype: torch.dtype,
) -> None:
    if config is None or config.method == "unquantized":
        return
    if dtype not in {torch.float16, torch.bfloat16}:
        raise capability_mismatch("W8A8 FP8 requires float16 or bfloat16 activations")
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.get_device_capability(target) < (8, 9):
        raise capability_mismatch("W8A8 FP8 requires CUDA compute capability 8.9 or newer")


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
