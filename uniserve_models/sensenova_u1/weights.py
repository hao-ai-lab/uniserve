"""SenseNova U1 checkpoint assignments and numerical precision presets.

All tensors come from one ``primary`` source. Module parameter paths are
translated into the checkpoint's names: the shared backbone under
``language_model.model``, the denoiser's generation modules under
``fm_modules`` and the input vision tower under ``vision_model``. The
backbone mapping covers only the layers and terminal modules resident on this
pipeline rank and declares the checkpoint names of the others nonresident.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from uniserve.loading import checkpoint, weights
from uniserve.model import TransformerDecoder

if TYPE_CHECKING:
    from .model import Model


checkpoint_sources = (checkpoint.Config("primary"),)


def _backbone_names(backbone: TransformerDecoder):
    """Translate backbone parameter paths to checkpoint language_model names.

    Text experts keep the plain checkpoint names; flow experts carry the
    ``_mot_gen`` suffix. QK norms split into temporal and spatial (``_hw``)
    checkpoint tensors. Only layers resident on this pipeline rank appear.
    """
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": "norm_mot_gen.weight",
    }

    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        _, index, kind, route, *parts = path.split(".")
        suffix = "" if route == "text" else "_mot_gen"
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
                spatial = "_hw" if parts[1] == "1" else ""
                target = (
                    f"self_attn.{'q' if parts[0] == 'query_norm' else 'k'}"
                    f"_norm{spatial}{suffix}.{parts[2]}"
                )
        elif kind == "mlps":
            tail = (
                tail.replace("gate_up.projections.gate", "gate_proj")
                .replace("gate_up.projections.up", "up_proj")
                .replace("down.", "down_proj.")
            )
            target = f"mlp{suffix}.{tail}"
        else:
            raise ValueError(f"unmapped SenseNova backbone parameter {path}")

        names[path] = f"layers.{index}.{target}"

    return {
        target: "language_model.model." + source
        for target, source in names.items()
    }


def _mapped(module, names, *, nonresident=frozenset()):
    """Build a primary-source mapping that skips tensors absent from the file.

    Every parameter named in ``names`` is required. A tensor missing from the
    file yields no assignment, so the loader reports that parameter as
    missing and fails with a load mismatch instead of a reader lookup error.
    ``nonresident`` names checkpoint tensors intentionally left unused.
    """  # noqa: E501

    def map_weights(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(names[name]))
            for name, parameter in module.named_parameters()
            if name in names and names[name] in available
        )

    return weights.ModuleMapping(
        module,
        "primary",
        map_weights,
        frozenset(
            name for name, _ in module.named_parameters() if name in names
        ),
        nonresident=nonresident,
    )


def checkpoint_mappings(model: Model):
    """Assign every resident module its checkpoint tensors per pipeline rank."""
    backbone = model.text.backbone
    names = _backbone_names(backbone)

    # PP omits only the source layers and terminal modules assigned elsewhere.
    # Every layer has the same tensors, so the first resident layer's
    # checkpoint names, minus the layer prefix, enumerate each absent layer.
    template = tuple(
        source.split(".", 4)[-1]
        for target, source in names.items()
        if target.startswith(f"layers.{next(iter(backbone.layers))}.")
    )
    nonresident = {
        f"language_model.model.layers.{index}.{tail}"
        for index in range(model.config.text.num_hidden_layers)
        if str(index) not in backbone.layers
        for tail in template
    }
    # With tied embeddings the embedding tensor is never declared unused:
    # on the last stage the tied head reads it as ``head_name``.
    if backbone.embedding is None and not model.config.text.tie_word_embeddings:
        nonresident.add("language_model.model.embed_tokens.weight")
    if backbone.norm is None:
        nonresident.update(
            (
                "language_model.model.norm.weight",
                "language_model.model.norm_mot_gen.weight",
            )
        )

    # A tied head reads the embedding tensor instead of a separate lm_head.
    head_name = (
        "language_model.model.embed_tokens.weight"
        if model.config.text.tie_word_embeddings
        else "language_model.lm_head.weight"
    )
    if model.text.lm_head is None or model.config.text.tie_word_embeddings:
        nonresident.add("language_model.lm_head.weight")

    components = [_mapped(backbone, names, nonresident=frozenset(nonresident))]
    if model.text.lm_head is not None:
        components.append(_mapped(model.text, {"lm_head.weight": head_name}))

    denoiser_names = {}
    for path, prefix in (
        ("input", "fm_modules.vision_model_mot_gen.embeddings."),
        ("time_embedding.projection", "fm_modules.timestep_embedder.mlp."),
    ):
        denoiser_names.update(
            {
                f"{path}.{name}": prefix + name
                for name, _ in model.denoiser.get_submodule(
                    path
                ).named_parameters()
            }
        )
    if model.denoiser.noise_embedding is not None:
        projection = model.denoiser.noise_embedding.projection
        denoiser_names.update(
            {
                "noise_embedding.projection."
                + name: "fm_modules.noise_scale_embedder.mlp." + name
                for name, _ in projection.named_parameters()
            }
        )

    # Each head variant has its own module structure and checkpoint naming
    # under fm_modules.fm_head; translate the active variant's paths. The
    # shallow MLP's nn.Sequential indices already match the checkpoint.
    for name, _ in model.denoiser.prediction.named_parameters():
        if model.config.flow.use_pixel_head:
            source = name.replace("decoder.blocks.1.", "conv1.").replace(
                "decoder.output.", "conv2."
            )
        elif model.config.flow.head.num_layers <= 2:
            source = name.removeprefix("head.")
        else:
            source = name.removeprefix("head.")
            source = source.replace(
                "time_embedding.projection.", "time_embed.mlp."
            )
            source = source.replace("input.", "input_proj.")
            source = source.replace("blocks.", "res_blocks.")
            source = (
                source.replace(".norm.", ".in_ln.")
                if source.startswith("res_blocks.")
                else source
            )
            source = source.replace(".modulation.", ".adaLN_modulation.")
            source = source.replace(
                "output.projection.", "final_layer.linear."
            ).replace("output.", "final_layer.")
            source = "net." + source
        denoiser_names["prediction." + name] = "fm_modules.fm_head." + source
    components.append(_mapped(model.denoiser, denoiser_names))

    components.append(
        _mapped(
            model.vision_encoder,
            {
                "network." + name: "vision_model.embeddings." + name
                for name, _ in model.vision_encoder.network.named_parameters()
            },
        )
    )
    return tuple(components)


precisions = MappingProxyType({"bf16": weights.Config()})

# Base precision for a calibrated ModelOpt checkpoint:
# ``uniserve_models.loading`` overlays the calibrated quantization, and every
# other module stays BF16.
checkpoint_precision = precisions["bf16"]
