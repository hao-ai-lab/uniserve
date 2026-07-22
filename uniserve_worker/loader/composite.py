"""Composite checkpoint loader: a root graph file plus sidecar submodule files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Type

import torch

from ..contracts.model_protocols import UniModel
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.quant.base import process_quantized_modules
from .base import BaseModelLoader, LoadResult
from .registry import register_loader
from .transformers import dtype_from_name
from .weight_spec import Sidecar, WeightSpec, weight_spec_of
from .weight_utils import iter_weights, stacked_params_mapping_loop

__all__ = [
    "CompositeCheckpointLoader",
]

logger = logging.getLogger(__name__)


class CompositeCheckpointLoader(BaseModelLoader):
    """Loads an eagerly constructed neural graph from a composite checkpoint.

    Registered under the ``composite`` load_format. The model's ``WeightSpec``
    declares the graph bindings, the candidate root weight files (first
    existing wins), the rename/stack rules, and the sidecar submodule files.
    The loader builds the graph from its resolved config, streams the root
    file through the declared rules, validates that every non-sidecar
    parameter loaded, loads each sidecar into its submodule, casts the graph
    to the declared serving dtype on the target device, and constructs the
    serving wrapper around the ready graph.
    """

    def load_model(
        self,
        model_cls: Type[UniModel],
        config: Any,
        *,
        device: str = "cpu",
        model_path: str | None = None,
        **kwargs: Any,
    ) -> LoadResult:
        del config  # the composite path resolves its own config from model_path
        if model_path is None:
            raise ValueError("CompositeCheckpointLoader requires model_path")
        spec: WeightSpec = weight_spec_of(model_cls)
        source = spec.graph
        if source is None:
            raise ValueError(
                f"{model_cls.__name__} declares no graph checkpoint source in its weight_spec"
            )
        serving_dtype = dtype_from_name(source.serving_dtype)
        cfg = source.config_cls.from_pretrained(model_path)
        graph = source.module_cls(cfg).eval()
        loaded, ignored = stacked_params_mapping_loop(
            graph,
            iter_weights([_root_weights_file(model_path, spec.checkpoint_files)]),
            spec.stacked,
            name_mapper=spec.map_name,
            dtype=serving_dtype,
        )
        sidecar_prefixes = tuple(f"{sidecar.module}." for sidecar in spec.sidecars)
        expected = {
            name
            for name, _ in graph.named_parameters()
            if not (sidecar_prefixes and name.startswith(sidecar_prefixes))
        }
        missing = sorted(expected - loaded)
        if missing:
            raise capability_mismatch(
                "composite checkpoint load mismatch: "
                f"missing={missing[:8]} ({len(missing)}) ignored={ignored[:8]}"
            )
        logger.info("loaded composite graph weights (%d params)", len(loaded))
        for sidecar in spec.sidecars:
            _load_sidecar(graph, model_path, sidecar)
        graph.to(device=device, dtype=serving_dtype)
        process_quantized_modules(graph.modules())
        model = model_cls(  # type: ignore[call-arg]
            cfg,
            model=graph,
            device=device,
            block_size=kwargs.get("block_size", DEFAULT_BLOCK_SIZE),
            kv_token_capacity=kwargs.get("kv_token_capacity"),
            attention_backend=kwargs.get("attention_backend"),
        )
        return LoadResult(model=model, tokenizer=None, device=device)


def _root_weights_file(model_dir: str, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = Path(model_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no root weight file among {candidates!r} under {model_dir}")


def _load_sidecar(graph: torch.nn.Module, model_dir: str, sidecar: Sidecar) -> None:
    state_dict = dict(iter_weights([Path(model_dir) / sidecar.file]))
    submodule = graph.get_submodule(sidecar.module)
    missing, unexpected = submodule.load_state_dict(state_dict, strict=False)
    real_missing = [
        name
        for name in missing
        if not any(substring in name for substring in sidecar.optional_substrings)
    ]
    if real_missing or unexpected:
        raise capability_mismatch(
            f"sidecar {sidecar.file!r} load mismatch: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    logger.info("loaded sidecar %s weights (%d tensors)", sidecar.file, len(state_dict))


register_loader("composite", CompositeCheckpointLoader())
