"""Named DiffusionGemma weight representations.

``precisions`` holds the presets ``uniserve_models.loading.load_model``
accepts by name for the dense BF16 checkpoint. ``checkpoint_precision`` is
the dense base of the calibrated ModelOpt NVFP4 checkpoint: that checkpoint
stores only the routed expert projections packed, so
``uniserve_models.loading`` overlays NVFP4 weights and their static input
scales on exactly the ``backbone.layers.{i}.moe.experts`` modules (under
both capabilities sharing the backbone) and every other module stays BF16.
"""

from __future__ import annotations

from types import MappingProxyType

from uniserve.loading import weights

precisions = MappingProxyType({"bf16": weights.Config()})

checkpoint_precision = precisions["bf16"]
