"""Resolve checkpoint metadata into concrete model configs and caller-owned assets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uniserve.loading.config import LoadConfig
from uniserve.loading.source import WeightSourceSet, resolve_model_root, resolve_weight_sources
from uniserve.model.model import Model
from uniserve.nn.layer import LayerConfig
from uniserve.nn.quant import QuantizationConfig
from uniserve.nn.quant.config import LinearPrecision
from uniserve_models.processing import FlowPrompt, ImageProcessor, resolve_input_tokens

if TYPE_CHECKING:
    from uniserve_models.catalog import CatalogEntry


@dataclass(frozen=True, slots=True)
class ModelSource:
    """A resolved model definition, checkpoint files and input-processing assets.

    Raw checkpoint dictionaries are consumed by the architecture reader here.
    The numerical loader receives the resulting config without parsing it again.
    Quantization and component precision describe loading, not model architecture.
    """

    entry: CatalogEntry
    config: Any
    weights: tuple[WeightSourceSet, ...]
    tokenizer: Any | None = None
    quantization: QuantizationConfig | None = None
    precisions: Mapping[str, LinearPrecision] = field(default_factory=dict)
    image_processor: ImageProcessor | None = None
    flow_prompt: FlowPrompt | None = None

    @property
    def model_class(self) -> type[Model]:
        return self.entry.model_class

    def configure_layers(self, layers: Mapping[str, LayerConfig]) -> Mapping[str, LayerConfig]:
        """Apply resolved checkpoint representation to the caller's borrowed layers."""

        configured = {
            name: replace(layer, quantization=self.quantization) for name, layer in layers.items()
        }
        if self.entry.configure_layers is not None:
            configured = dict(self.entry.configure_layers(configured, self.precisions))
        return configured


def resolve_model(
    model_path: str,
    *,
    load: LoadConfig = LoadConfig(),
    components: frozenset[str] | None = None,
    quantization: Mapping[str, object] | None = None,
) -> ModelSource:
    """Read one checkpoint snapshot and normalize its architecture exactly once.

    ``components`` selects resident numerical module paths for checkpoint I/O;
    all architecture sidecars are still read to validate the complete model.
    Omitting it resolves every component. No placement or process groups are
    created. A tokenizer, when required for inputs, belongs to the caller.
    """

    from uniserve_models.catalog import resolve_catalog_entry

    if not model_path:
        raise ValueError("model_path must not be empty")
    root, repository_id = resolve_model_root(model_path, load)
    metadata = read_model_config(root)
    entry = resolve_catalog_entry(
        tuple(str(value) for value in metadata.get("architectures") or ())
    )
    weights = resolve_weight_sources(
        model_path,
        load,
        sources=entry.sources,
        sidecars=entry.sidecars,
        root=root,
        repository_id=repository_id,
        components=components,
    )
    overrides = {} if quantization is None else quantization
    quantized = QuantizationConfig.from_model_config(
        metadata,
        overrides=overrides if entry.component_precisions is None else {},
    )
    config = entry.prepare_config(metadata, root, weights)
    tokenizer = None
    if entry.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            root, use_fast=False, trust_remote_code=False, local_files_only=True
        )
    precisions = {} if entry.component_precisions is None else entry.component_precisions(overrides)
    processor = None if entry.image_processor is None else entry.image_processor(config)
    return ModelSource(
        entry,
        config,
        weights,
        tokenizer,
        quantized,
        precisions,
        resolve_input_tokens(processor, tokenizer),
        entry.flow_prompt,
    )


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
