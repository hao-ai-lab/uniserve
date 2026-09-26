"""DiffusionGemma checkpoint assignments.

Checkpoint names follow the Transformers ``DiffusionGemmaForBlockDiffusion``
layout in one source, ``primary``, at the checkpoint root. The text stack is
stored once, under ``model.decoder``; the prompt encoder's text layers are
tied to it except for their ``layer_scalar``, which the checkpoint repeats
under ``model.encoder.language_model`` and which must equal the decoder's.
The head is tied to the token embedding. Experts come either as stacked
``[E, ...]`` tensors (``experts.gate_up_proj`` with each expert's gate rows
before its up rows, and ``experts.down_proj``) or, in ModelOpt exports, as
separate ``experts.{e}.{gate,up,down}_proj`` matrices; both load into the
same stacked ``FusedMoE`` rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from uniserve.loading import checkpoint, weights

from .config import Config

if TYPE_CHECKING:
    from .model import Model


checkpoint_sources = (checkpoint.Config(name="primary"),)

_TEXT = "model.decoder"
_ENCODER_TEXT = "model.encoder.language_model"
_VISION = "model.encoder.vision_tower"


def parameter_sources(config: Config) -> Mapping[str, str]:
    """Map complete logical parameters to checkpoint names.

    Keys are parameter paths of a complete, unbound ``Model``: the text
    backbone and head under ``text``, which the denoiser shares, and the
    self-conditioning block under ``denoiser``. The tied head maps to the
    embedding tensor, so the last pipeline stage, which holds the head but
    not the embedding, still loads it. Stacked expert parameters are absent;
    ``expert_sources`` names them.
    """
    text = config.text
    embedding = f"{_TEXT}.embed_tokens.weight"
    names = {
        "text.backbone.embedding.weight": embedding,
        "text.lm_head.weight": embedding,
        "text.backbone.norm.weight": f"{_TEXT}.norm.weight",
    }
    for index, layer in enumerate(text.layers):
        target, source = (
            f"text.backbone.layers.{index}",
            f"{_TEXT}.layers.{index}",
        )
        for name, stored in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_feedforward_norm", "pre_feedforward_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm_1"),
            ("moe.input_norm", "pre_feedforward_layernorm_2"),
            ("moe.output_norm", "post_feedforward_layernorm_2"),
            ("post_feedforward_norm", "post_feedforward_layernorm"),
            ("attention.query_norm", "self_attn.q_norm"),
            ("attention.key_norm", "self_attn.k_norm"),
            ("attention.output", "self_attn.o_proj"),
            ("mlp.down", "mlp.down_proj"),
            ("moe.router.projection", "router.proj"),
        ):
            names[f"{target}.{name}.weight"] = f"{source}.{stored}.weight"
        for branch in ("q", "k", "v") if layer.value_projection else ("q", "k"):
            names[f"{target}.attention.qkv.projections.{branch}.weight"] = (
                f"{source}.self_attn.{branch}_proj.weight"
            )
        for branch in ("gate", "up"):
            names[f"{target}.mlp.gate_up.projections.{branch}.weight"] = (
                f"{source}.mlp.{branch}_proj.weight"
            )
        names[f"{target}.moe.router.scale"] = f"{source}.router.scale"
        names[f"{target}.moe.router.per_expert_scale"] = (
            f"{source}.router.per_expert_scale"
        )
        names[f"{target}.layer_scalar"] = f"{source}.layer_scalar"

    conditioning = "denoiser.self_conditioning"
    names[f"{conditioning}.pre_norm.weight"] = (
        f"{_TEXT}.self_conditioning.pre_norm.weight"
    )
    for branch in ("gate", "up"):
        names[f"{conditioning}.mlp.gate_up.projections.{branch}.weight"] = (
            f"{_TEXT}.self_conditioning.{branch}_proj.weight"
        )
    names[f"{conditioning}.mlp.down.weight"] = (
        f"{_TEXT}.self_conditioning.down_proj.weight"
    )

    network = "vision_encoder.network"
    names[f"{network}.patch_embedder.projection.weight"] = (
        f"{_VISION}.patch_embedder.input_proj.weight"
    )
    names[f"{network}.patch_embedder.position_table"] = (
        f"{_VISION}.patch_embedder.position_embedding_table"
    )
    if config.vision.standardize:
        names[f"{network}.pooler.std_bias"] = f"{_VISION}.std_bias"
        names[f"{network}.pooler.std_scale"] = f"{_VISION}.std_scale"
    for index in range(config.vision.num_hidden_layers):
        target = f"{network}.transformer.layers.{index}"
        source = f"{_VISION}.encoder.layers.{index}"
        for name, stored in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_feedforward_norm", "pre_feedforward_layernorm"),
            ("post_feedforward_norm", "post_feedforward_layernorm"),
            ("attention.query_norm", "self_attn.q_norm"),
            ("attention.key_norm", "self_attn.k_norm"),
        ):
            names[f"{target}.{name}.weight"] = f"{source}.{stored}.weight"
        for branch in ("q", "k", "v"):
            names[f"{target}.attention.qkv.projections.{branch}.weight"] = (
                f"{source}.self_attn.{branch}_proj.linear.weight"
            )
        names[f"{target}.attention.output.weight"] = (
            f"{source}.self_attn.o_proj.linear.weight"
        )
        for branch in ("gate", "up"):
            names[f"{target}.mlp.gate_up.projections.{branch}.weight"] = (
                f"{source}.mlp.{branch}_proj.linear.weight"
            )
        names[f"{target}.mlp.down.weight"] = (
            f"{source}.mlp.down_proj.linear.weight"
        )
    names["vision_encoder.connector.projection.weight"] = (
        "model.encoder.embed_vision.embedding_projection.weight"
    )
    return names


def expert_sources(config: Config) -> Mapping[str, str]:
    """Map each layer's ``FusedMoE`` module path to its checkpoint prefix.

    The prefix holds either the stacked ``gate_up_proj`` and ``down_proj``
    tensors or each expert's ``{e}.{gate,up,down}_proj.weight`` matrices.
    """
    return {
        f"text.backbone.layers.{index}.moe.experts": (
            f"{_TEXT}.layers.{index}.experts"
        )
        for index in range(config.text.num_hidden_layers)
    }


def _expert_names(prefix: str, count: int) -> frozenset[str]:
    """Every checkpoint name either expert layout may use under ``prefix``.

    Per-expert matrices of a ModelOpt export carry their NVFP4 block, tensor
    and input scales beside each weight.
    """
    return frozenset(
        {f"{prefix}.gate_up_proj", f"{prefix}.down_proj"}
        | {
            f"{prefix}.{expert}.{projection}_proj.{field}"
            for expert in range(count)
            for projection in ("gate", "up", "down")
            for field in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            )
        }
    )


def _check_layer_scalars(reader: checkpoint.Reader, layers: int) -> None:
    """Require the encoder's layer scalars to equal the decoder's.

    The model runs one backbone for both roles, so a checkpoint whose two
    copies disagree has no faithful single-backbone representation. Dummy
    loading synthesizes unrelated values for the two names and is not
    compared.
    """
    if reader.io.mode == "dummy":
        return
    available = frozenset(reader.names())
    for index in range(layers):
        decoder = f"{_TEXT}.layers.{index}.layer_scalar"
        encoder = f"{_ENCODER_TEXT}.layers.{index}.layer_scalar"
        if encoder not in available or decoder not in available:
            continue
        if not torch.equal(
            reader.get(encoder).read(), reader.get(decoder).read()
        ):
            raise ValueError(
                f"DiffusionGemma encoder and decoder layer_scalar of layer "
                f"{index} differ; one backbone cannot represent both"
            )


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    """Declare how the resident parameters of ``model`` load from ``primary``.

    ``uniserve_models.loading.read_config`` calls this on the complete meta
    skeleton, and ``uniserve.loading.load_model`` calls it again after
    ``parallelize_`` has bound the pipeline stage, so ``model`` may lack
    layers, the embedding or the head. Checkpoint tensors of absent
    parameters are declared nonresident rather than unexpected. The
    assignment reads the encoder and decoder layer scalars and raises
    ``ValueError`` when they differ.
    """
    config = model.config
    names = parameter_sources(config)
    experts = expert_sources(config)
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters(remove_duplicate=False))
    off_stage = frozenset(
        source for target, source in names.items() if target not in parameters
    ) | frozenset(
        name
        for path, prefix in experts.items()
        if path not in modules
        for name in _expert_names(prefix, config.text.num_experts)
    )
    # A tensor still read by a resident parameter is never nonresident.
    off_stage -= {names[target] for target in parameters if target in names}

    def assign(reader):
        _check_layer_scalars(reader, config.text.num_hidden_layers)
        available = frozenset(reader.names())
        result = []
        for path, prefix in experts.items():
            module = modules.get(path)
            if module is None:
                continue
            if f"{prefix}.gate_up_proj" in available:
                # Transformers stacks each expert's gate rows before its up
                # rows; the module keeps up rows first.
                gate_up = reader.get(f"{prefix}.gate_up_proj")
                result.extend(
                    weights.expert_assignments(
                        module,
                        up=(gate_up, config.text.moe_intermediate_size),
                        gate=(gate_up, 0),
                        down=reader.get(f"{prefix}.down_proj"),
                    )
                )
                continue
            for expert in range(config.text.num_experts):
                sources = {
                    projection: f"{prefix}.{expert}.{projection}_proj.weight"
                    for projection in ("gate", "up", "down")
                }
                if not available.issuperset(sources.values()):
                    continue
                result.extend(
                    weights.expert_assignments(
                        module,
                        up=(reader.get(sources["up"]), 0),
                        gate=(reader.get(sources["gate"]), 0),
                        down=reader.get(sources["down"]),
                        expert=expert,
                    )
                )

        # ``parameters`` lists shared Parameters under every path. Paths in
        # ``names`` assign them; a tied Parameter reached through both the
        # embedding and the head receives the same tensor twice, and the
        # loader skips the identical repeat.
        for target, parameter in parameters.items():
            source = names.get(target)
            if source is not None and source in available:
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
