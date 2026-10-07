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
from uniserve.quantization import QuantizationConfig, Quantizer

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
    experts = expert_sources(config)
    if not experts:
        return _PRECISIONS
    quantizer = Quantizer("mxfp8")
    return MappingProxyType(
        {
            **_PRECISIONS,
            "mxfp8-experts": weights.Config(
                quantization=dict.fromkeys(
                    experts, QuantizationConfig(quantizer, quantizer)
                )
            ),
        }
    )


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

        if config.sparse(index):
            # Stacked expert parameters receive every expert's projections
            # through ``expert_sources``.
            names[f"{target}.mlp.router.weight"] = f"{source}.mlp.gate.weight"
            continue
        for branch in ("gate", "up"):
            names[f"{target}.mlp.gate_up.projections.{branch}.weight"] = (
                f"{source}.mlp.{branch}_proj.weight"
            )
        names[f"{target}.mlp.down.weight"] = f"{source}.mlp.down_proj.weight"

    return names


def expert_sources(config: Config) -> Mapping[str, str]:
    """Map each MoE layer's stacked expert module to its checkpoint prefix.

    Keys are ``FusedMoE`` module paths of a complete ``Model``; values are
    the Transformers prefix whose ``{expert}.{gate,up,down}_proj.weight``
    tensors hold each expert's matrices.
    """
    return {
        f"backbone.layers.{index}.mlp.experts": (
            f"model.layers.{index}.mlp.experts"
        )
        for index in range(config.num_hidden_layers)
        if config.sparse(index)
    }


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
    experts = expert_sources(model.config)
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters(remove_duplicate=False))
    off_stage = frozenset(
        source for target, source in names.items() if target not in parameters
    ) | frozenset(
        f"{prefix}.{expert}.{projection}_proj.weight"
        for path, prefix in experts.items()
        for expert in range(model.config.num_experts)
        if path not in modules
        or not (
            modules[path].expert_slice.start
            <= expert
            < modules[path].expert_slice.stop
        )
        for projection in ("gate", "up", "down")
    )
    # Tied heads can have a redundant checkpoint copy; the embedding is the
    # unique source for their shared Parameter on either pipeline endpoint.
    if model.config.tie_word_embeddings:
        off_stage |= {"lm_head.weight"}
    # A tensor still read by a resident parameter is never nonresident; with
    # tied embeddings the last stage's head reads the embedding tensor.
    used_sources = {names[target] for target in parameters if target in names}
    off_stage -= used_sources
    # Prefix normalization applies equally to resident and nonresident
    # expert tensors. Declaring peer-owned experts does not consume or load
    # them, and every unrelated checkpoint name remains an error.
    off_stage |= {
        source.removeprefix("model.")
        for source in off_stage
        if source.startswith("model.layers.")
    }

    # ``parameters`` lists a tied Parameter under both of its paths, as the
    # loader's own remove_duplicate=False view does. Both paths assign the
    # same tensor, and the loader skips the identical repeat.
    def assign(reader):
        available = frozenset(reader.names())

        def resolve(source):
            # Some checkpoints serialize the same tensors without the
            # "model." prefix used by the canonical naming.
            if source not in available and source.startswith("model.layers."):
                source = source.removeprefix("model.")
            return reader.get(source) if source in available else None

        result: list[weights.Assignment] = []
        for path, prefix in experts.items():
            module = modules.get(path)
            if module is None:
                continue
            for expert in range(model.config.num_experts):
                sources = {
                    projection: resolve(
                        f"{prefix}.{expert}.{projection}_proj.weight"
                    )
                    for projection in ("gate", "up", "down")
                }
                if any(value is None for value in sources.values()):
                    continue
                result.extend(
                    weights.expert_assignments(
                        module,
                        up=(sources["up"], 0),
                        gate=(sources["gate"], 0),
                        down=sources["down"],
                        expert=expert,
                    )
                )
        for target, parameter in parameters.items():
            weight = resolve(names[target]) if target in names else None
            if weight is not None:
                result.append(weights.Assignment(parameter, weight))
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
