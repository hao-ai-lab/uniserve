"""Named DiffusionGemma weight representations.

``precisions`` returns the presets ``uniserve_models.loading.load_model``
accepts by name for the dense BF16 checkpoint. ``checkpoint_precision`` is
the dense base of the calibrated ModelOpt NVFP4 checkpoint: that checkpoint
stores only the routed expert projections packed, so
``uniserve_models.loading`` overlays NVFP4 weights and their static input
scales on exactly the ``backbone.layers.{i}.moe.experts`` modules (under
both capabilities sharing the backbone) and every other module stays BF16.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from uniserve.loading import weights

from .config import Config


def precisions(config: Config) -> Mapping[str, weights.Config]:
    """Return the dense checkpoint's named numerical representations."""
    return MappingProxyType({"bf16": weights.Config()})


def checkpoint_precision(config: Config) -> weights.Config:
    """Return the BF16 base overlaid by calibrated expert projections."""
    return weights.Config()
