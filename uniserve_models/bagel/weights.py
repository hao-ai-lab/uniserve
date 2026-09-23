"""BAGEL checkpoint assignments and numerical precision presets."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from uniserve.loading import checkpoint, weights
from uniserve_models import siglip

from . import vae

if TYPE_CHECKING:
    from .model import Model
    from .transformer import Transformer


checkpoint_sources = (
    checkpoint.Config(
        "primary", filenames=("ema.safetensors", "model.safetensors")
    ),
    checkpoint.Config("autoencoder", filenames=("ae.safetensors",)),
)


precisions = MappingProxyType(
    {
        "bf16": weights.Config(
            dtypes={
                "latent_encoder": torch.float32,
                "image_decoder.decoder": torch.float32,
            }
        )
    }
)

# Dense modules of a calibrated checkpoint keep its stored representation.
checkpoint_precision = precisions["bf16"]


def _backbone_names(backbone: Transformer) -> dict[str, str]:
    """Translate backbone parameter paths to checkpoint language_model names.

    Text experts keep the plain checkpoint names; flow experts carry the
    ``_moe_gen`` suffix. Only layers resident on this pipeline rank appear.
    """
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": "norm_moe_gen.weight",
    }

    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        _, index, kind, route, *parts = path.split(".")
        suffix = "" if route == "text" else "_moe_gen"
        tail = ".".join(parts)
        if kind in ("input_norms", "post_attention_norms"):
            target = (
                (
                    "input_layernorm"
                    if kind == "input_norms"
                    else "post_attention_layernorm"
                )
                + suffix
                + "."
                + tail
            )
        elif kind == "outputs":
            target = f"self_attn.o_proj{suffix}.{tail}"
        elif kind == "projections":
            if parts[0] == "projection":
                target = f"self_attn.{parts[2]}_proj{suffix}.{parts[3]}"
            else:
                target = (
                    f"self_attn.{'q' if parts[0] == 'query_norm' else 'k'}"
                    f"_norm{suffix}.{parts[1]}"
                )
        elif kind == "mlps":
            tail = (
                tail.replace("gate_up.projections.gate", "gate_proj")
                .replace("gate_up.projections.up", "up_proj")
                .replace("down.", "down_proj.")
            )
            target = f"mlp{suffix}.{tail}"
        else:
            raise ValueError(f"unmapped BAGEL backbone parameter {path}")

        names[path] = f"layers.{index}.{target}"

    return {
        target: "language_model.model." + source
        for target, source in names.items()
    }


def _mapped(module, source_names, *, nonresident=frozenset()):
    """Build a primary-source mapping that skips tensors absent from the file."""  # noqa: E501

    def map_weights(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(source_names[name]))
            for name, parameter in module.named_parameters()
            if name in source_names and source_names[name] in available
        )

    return weights.ModuleMapping(
        module,
        "primary",
        map_weights,
        frozenset(
            name
            for name, _ in module.named_parameters()
            if name in source_names
        ),
        nonresident=nonresident,
    )


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    """Assign every resident module its checkpoint tensors per pipeline rank."""
    backbone = model.text.backbone
    source_names = _backbone_names(backbone)

    # PP omits only the source layers and terminal modules assigned elsewhere.
    template = tuple(
        source.split(".", 4)[-1]
        for target, source in source_names.items()
        if target.startswith(f"layers.{next(iter(backbone.layers))}.")
    )
    nonresident = {
        f"language_model.model.layers.{index}.{tail}"
        for index in range(model.config.text.num_hidden_layers)
        if str(index) not in backbone.layers
        for tail in template
    }
    if backbone.embedding is None:
        nonresident.add("language_model.model.embed_tokens.weight")
    if backbone.norm is None:
        nonresident.update(
            (
                "language_model.model.norm.weight",
                "language_model.model.norm_moe_gen.weight",
            )
        )

    components = [
        _mapped(backbone, source_names, nonresident=frozenset(nonresident))
    ]
    if model.text.lm_head is not None:
        components.append(
            _mapped(
                model.text, {"lm_head.weight": "language_model.lm_head.weight"}
            )
        )
    else:
        # Without a resident head, declare its tensor nonresident on the
        # backbone.
        first = components[0]
        components[0] = weights.ModuleMapping(
            first.module,
            first.source,
            first.map_weights,
            first.required,
            nonresident=first.nonresident | {"language_model.lm_head.weight"},
        )

    denoiser_names = {}
    for path, prefix in (
        ("input", "vae2llm."),
        ("prediction", "llm2vae."),
        ("time_embedding.projection", "time_embedder.mlp."),
    ):
        denoiser_names.update(
            {
                f"{path}.{name}": prefix + name
                for name, _ in model.denoiser.get_submodule(
                    path
                ).named_parameters()
            }
        )
    denoiser_names["position.weight"] = "latent_pos_embed.pos_embed"
    components.append(_mapped(model.denoiser, denoiser_names))

    vision = model.vision_encoder.network
    vision_names = {
        "network.connector." + name: "connector."
        + name.replace("projection.0.", "fc1.").replace("projection.2.", "fc2.")
        for name, _ in vision.connector.named_parameters()
    }
    vision_names["network.position.weight"] = "vit_pos_embed.pos_embed"
    projected = _mapped(model.vision_encoder, vision_names)
    components.append(
        weights.ModuleMapping(
            model.vision_encoder,
            "primary",
            lambda reader: (
                projected.map_weights(reader)
                + siglip.assignments(
                    vision.encoder, reader, prefix="vit_model.vision_model."
                )
            ),
            frozenset(dict(model.vision_encoder.named_parameters())),
        )
    )

    components.append(
        weights.ModuleMapping(
            model.latent_encoder,
            "autoencoder",
            lambda reader: vae.assignments(model.latent_encoder, reader),
            frozenset(dict(model.latent_encoder.named_parameters())),
        )
    )
    return tuple(components)
