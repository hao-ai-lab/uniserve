"""Qwen3-VL checkpoint assignments.

Names follow the Transformers ``Qwen3VLForConditionalGeneration`` layout:
the language model below ``model.language_model.`` (Qwen3 decoder names) and
the vision tower below ``model.visual.``. The composing model owns the
``weights.ModuleMapping`` these assignments feed and declares every other
checkpoint tensor nonresident; ``sources`` names the tensors a module's
resident parameters read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve.loading import checkpoint, weights
from uniserve_models.qwen3.weights import parameter_sources

if TYPE_CHECKING:
    from .encoder import Encoder
    from .vision import VisionTower

LANGUAGE_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."

# A rectangle of a checkpoint tensor; None reads the whole tensor.
_Region = tuple[slice, ...] | None

# Vision module names below each block and merger, as checkpoint names.
_BLOCK_NAMES = {
    "input_norm": "norm1",
    "output_norm": "norm2",
    "attention.output": "attn.proj",
    "mlp.0": "mlp.linear_fc1",
    "mlp.2": "mlp.linear_fc2",
}
_MERGER_NAMES = {
    "norm": "norm",
    "mlp.0": "linear_fc1",
    "mlp.2": "linear_fc2",
}


def _vision_source(tower: VisionTower, name: str) -> tuple[str, _Region]:
    """Map one ``VisionTower`` parameter to its checkpoint name and region.

    The checkpoint fuses each block's Q/K/V projection into one ``qkv``
    tensor with Q, K and V rows in that order; each branch reads its rows.
    """
    module, _, field = name.rpartition(".")
    if module == "patch_embedding":
        return f"patch_embed.proj.{field}", None
    if module == "position_embedding":
        return "pos_embed.weight", None

    kind, _, path = module.partition(".")
    if kind == "merger":
        return f"merger.{_MERGER_NAMES[path]}.{field}", None

    # Blocks and DeepStack mergers are numbered module lists.
    index, _, path = path.partition(".")
    if kind == "deepstack":
        return (
            f"deepstack_merger_list.{index}.{_MERGER_NAMES[path]}.{field}",
            None,
        )
    if kind != "layers":
        raise ValueError(f"unmapped Qwen3-VL vision parameter {name!r}")
    if path.startswith("attention.qkv.projections."):
        branch = ("q", "k", "v").index(path.rpartition(".")[2])
        width = tower.config.hidden_size
        rows = slice(branch * width, (branch + 1) * width)
        region = (rows, slice(0, width)) if field == "weight" else (rows,)
        return f"blocks.{index}.attn.qkv.{field}", region
    return f"blocks.{index}.{_BLOCK_NAMES[path]}.{field}", None


def _sources(encoder: Encoder) -> dict[str, tuple[str, _Region]]:
    """Map every parameter of ``encoder`` to its checkpoint tensor region."""
    language = {
        "network." + target.removeprefix("backbone."): source.replace(
            "model.", LANGUAGE_PREFIX, 1
        )
        for target, source in parameter_sources(encoder.network.config).items()
        if target.startswith("backbone.")
    }
    tower = encoder.vision.network
    result: dict[str, tuple[str, _Region]] = {}
    for name, _ in encoder.named_parameters():
        if name.startswith("network."):
            result[name] = (language[name], None)
        else:
            source, region = _vision_source(
                tower, name.removeprefix("vision.network.")
            )
            result[name] = (VISION_PREFIX + source, region)
    return result


def assignments(
    encoder: Encoder, reader: checkpoint.Reader
) -> tuple[weights.Assignment, ...]:
    """Assign every resident ``Encoder`` parameter its checkpoint region.

    Pipeline and tensor-parallel binding may already have pruned or
    partitioned the encoder; only its remaining parameters are assigned.
    Parameters whose tensor the reader lacks receive no assignment, which
    the loader reports as missing.
    """
    available = frozenset(reader.names())
    parameters = dict(encoder.named_parameters())
    return tuple(
        weights.Assignment(
            parameters[name], reader.get(source), source_slice=region
        )
        for name, (source, region) in _sources(encoder).items()
        if source in available
    )


def sources(encoder: Encoder) -> frozenset[str]:
    """Name the checkpoint tensors the resident parameters read.

    A composing model declares every other checkpoint tensor of the
    Qwen3-VL checkpoint nonresident: the language head, the final norm,
    decoder layers outside the retained ones or this pipeline stage.
    """
    return frozenset(source for source, _ in _sources(encoder).values())
