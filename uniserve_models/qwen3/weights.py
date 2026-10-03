"""Qwen3 checkpoint assignments and numerical precision presets.

Checkpoint names follow the Transformers ``Qwen3ForCausalLM`` and
``Qwen3MoeForCausalLM`` layouts in one checkpoint source, ``primary``, at the
checkpoint root. Parameters map to complete logical tensors; the loader
applies each layer's tensor-parallel partition.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from uniserve.loading import checkpoint, weights

from .config import Config

if TYPE_CHECKING:
    from .model import Model


checkpoint_sources = (checkpoint.Config(name="primary"),)

# Named presets that load_model accepts; a calibrated ModelOpt checkpoint
# offers none. With no "default" entry, uniserve_models.loading.read_config
# starts from "bf16".
_PRECISIONS = MappingProxyType(
    {"bf16": weights.Config(), "fp16": weights.Config(dtype=torch.float16)}
)


def precisions(config: Config) -> Mapping[str, weights.Config]:
    """Named presets ``load_model`` accepts for ``config``'s checkpoint."""
    return _PRECISIONS


def checkpoint_precision(config: Config) -> weights.Config:
    """Base precision of a calibrated ModelOpt checkpoint.

    The loader adds the calibrated quantization on top of it, and every
    module the checkpoint does not store packed stays dense in this
    precision.
    """
    return _PRECISIONS["bf16"]


def parameter_sources(config: Config) -> Mapping[str, str]:
    """Map complete logical Qwen parameters to checkpoint names before partitioning.

    The result covers every parameter of a complete, unbound ``Model``,
    keyed by module path. A tied head maps to the embedding tensor. The
    MiniMax H3 text encoder reuses the ``backbone.`` entries under its own
    prefixes.
    """  # noqa: E501
    names = {
        "backbone.embedding.weight": "model.embed_tokens.weight",
        "backbone.norm.weight": "model.norm.weight",
        "lm_head.weight": "model.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight",
    }
    for index in range(config.num_hidden_layers):
        target, source = f"backbone.layers.{index}", f"model.layers.{index}"
        names[f"{target}.input_norm.weight"] = (
            f"{source}.input_layernorm.weight"
        )
        names[f"{target}.post_attention_norm.weight"] = (
            f"{source}.post_attention_layernorm.weight"
        )

        for field in (
            ("weight", "bias") if config.attention_bias else ("weight",)
        ):
            for branch in ("q", "k", "v"):
                names[
                    f"{target}.attention.qkv.projection.projections.{branch}.{field}"
                ] = f"{source}.self_attn.{branch}_proj.{field}"
            names[f"{target}.attention.output.{field}"] = (
                f"{source}.self_attn.o_proj.{field}"
            )

        for branch in ("q", "k"):
            norm = "query_norm" if branch == "q" else "key_norm"
            names[f"{target}.attention.qkv.{norm}.weight"] = (
                f"{source}.self_attn.{branch}_norm.weight"
            )

        if config.num_experts:
            names[f"{target}.mlp.router.weight"] = f"{source}.mlp.gate.weight"
            mlps = tuple(
                (
                    f"{target}.mlp.experts.experts.{expert}",
                    f"{source}.mlp.experts.{expert}",
                )
                for expert in range(config.num_experts)
            )
        else:
            mlps = ((f"{target}.mlp", f"{source}.mlp"),)
        for target_mlp, source_mlp in mlps:
            for branch in ("gate", "up"):
                names[f"{target_mlp}.gate_up.projections.{branch}.weight"] = (
                    f"{source_mlp}.{branch}_proj.weight"
                )
            names[f"{target_mlp}.down.weight"] = (
                f"{source_mlp}.down_proj.weight"
            )

    return names


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    """Declare how the resident parameters of ``model`` load from ``primary``.

    ``uniserve_models.loading.read_config`` calls this on the complete meta
    skeleton, and ``uniserve.loading.load_model`` calls it again after
    ``parallelize_`` has bound the pipeline stage, so ``model`` may lack
    layers, the embedding or the head. Checkpoint tensors of those absent
    parameters are declared nonresident rather than unexpected. A parameter
    whose tensor the checkpoint lacks receives no assignment; outside dummy
    loading, the load fails reporting it missing when it is selected.
    """
    names = parameter_sources(model.config)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    off_stage = frozenset(
        source for target, source in names.items() if target not in parameters
    )
    # Tied heads can have a redundant checkpoint copy; the embedding is the
    # unique source for their shared Parameter on either pipeline endpoint.
    if model.config.tie_word_embeddings:
        off_stage |= {"lm_head.weight"}
    # A tensor still read by a resident parameter is never nonresident; with
    # tied embeddings the last stage's head reads the embedding tensor.
    used_sources = {names[target] for target in parameters}
    off_stage -= used_sources

    # ``parameters`` lists a tied Parameter under both of its paths, as the
    # loader's own remove_duplicate=False view does. Both paths assign the
    # same tensor, and the loader skips the identical repeat.
    def assign(reader):
        available = frozenset(reader.names())
        result = []
        for target, parameter in parameters.items():
            source = names[target]
            # Some checkpoints serialize the same tensors without the
            # "model." prefix used by the canonical naming.
            if source not in available and source.startswith("model.layers."):
                source = source.removeprefix("model.")
            if source in available:
                result.append(weights.Assignment(parameter, reader.get(source)))
        return tuple(result)

    return (
        weights.ModuleMapping(
            model,
            "primary",
            assign,
            frozenset(parameters),
            nonresident=off_stage,
        ),
    )
