"""BAGEL checkpoint assignments and numerical precision presets.

A BAGEL checkpoint stores the language model, denoiser heads, vision tower
and connector in one ``primary`` file (``ema.safetensors`` when present,
otherwise ``model.safetensors``) and the FLUX autoencoder in a separate
``autoencoder`` file (``ae.safetensors``). ``checkpoint_mappings`` translates
UniServe module paths to those tensor names. ``uniserve_models.loading``
calls it on the complete meta-device model to choose sources;
``uniserve.loading.load_model`` calls it again after binding parallel
partitions; under pipeline parallelism only the rank's resident layers and
terminal modules then remain.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch

from uniserve.loading import checkpoint, weights
from uniserve.model import TransformerDecoder
from uniserve_models import siglip

from . import vae

if TYPE_CHECKING:
    from .model import Model
    from .vision import Encoder


# ``config.read_config`` opens ``checkpoint_sources[0]`` to size the latent
# position grid from a tensor header, and ``config.config_sources`` is the
# first entry, so the primary source must stay first.
checkpoint_sources = (
    checkpoint.Config(
        "primary", filenames=("ema.safetensors", "model.safetensors")
    ),
    checkpoint.Config("autoencoder", filenames=("ae.safetensors",)),
)


# The same ``PatchAutoencoder`` is reachable as ``latent_encoder`` and as
# ``image_decoder.decoder``. The loader chooses a dtype per module path and
# refuses a shared parameter whose alias paths disagree, so both paths are
# named float32; every other module takes the default bfloat16 dtype.
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

# Dense base for a calibrated ModelOpt checkpoint: ``read_config`` in
# ``uniserve_models.loading`` overlays the checkpoint's packed NVFP4 modules
# onto it and offers no runtime presets.
checkpoint_precision = precisions["bf16"]


def _backbone_names(backbone: TransformerDecoder) -> dict[str, str]:
    """Translate backbone parameter paths to checkpoint language_model names.

    Text experts keep the plain checkpoint names; flow experts carry the
    ``_moe_gen`` suffix. Only layers the backbone holds appear; after
    pipeline partitioning that is this rank's share.

    Returns:
        A mapping from backbone parameter paths to full checkpoint tensor names
        under ``language_model.model.``. The embedding and final norm entries
        are always present, even on a rank that does not hold those modules.

    Raises:
        ValueError: A layer parameter has no known checkpoint name.
    """
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": "norm_moe_gen.weight",
    }

    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        # Layer paths read ``layers.<index>.<kind>.<route>.<parts...>``, where
        # route is ``text`` or ``flow``.
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
            # parts is ``projection.projections.<q|k|v>.<weight|bias>`` for
            # the fused QKV linear, or ``<query|key>_norm.weight`` for the
            # QK norms.
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

    # ``source_names`` maps module parameter paths to checkpoint names. Every
    # parameter in ``source_names`` is required, so a tensor absent from the
    # file yields no assignment and the loader reports that parameter as
    # missing.
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
    # Every layer has the same tensor set, so the first resident layer's
    # tails name the tensors of each nonresident layer. Declaring them
    # nonresident keeps the loader from rejecting them as unexpected. Every
    # pipeline stage holds at least one layer (``parallelize_`` enforces it),
    # so ``next`` has a value.
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
        # backbone mapping, which reads the same primary file.
        first = components[0]
        components[0] = weights.ModuleMapping(
            first.module,
            first.source,
            first.map_weights,
            first.required,
            nonresident=first.nonresident | {"language_model.lm_head.weight"},
        )

    # The denoiser also holds the shared backbone; only its own input,
    # prediction, timestep and position tensors are named here, so the
    # backbone keeps a single mapping.
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

    # The model always composes its vision encoder around BAGEL's Encoder.
    vision = cast("Encoder", model.vision_encoder.network)

    # ``projection.0`` and ``projection.2`` are the two linears of
    # ``MLPConnector``'s sequential; index 1 is its activation.
    vision_names = {
        "network.connector." + name: "connector."
        + name.replace("projection.0.", "fc1.").replace("projection.2.", "fc2.")
        for name, _ in vision.connector.named_parameters()
    }
    vision_names["network.position.weight"] = "vit_pos_embed.pos_embed"
    projected = _mapped(model.vision_encoder, vision_names)

    # One mapping merges the connector and position assignments with the
    # SigLIP tower's, so its required set covers every vision parameter.
    # ``projected`` contributes only its ``map_weights``.
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
