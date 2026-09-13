"""Common numerical composition and checkpoint declarations for library models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from torch import nn

from uniserve_worker.modeling.components import Call, ComponentSpec
from uniserve_worker.modeling.geometry import CacheGeometry, Shape, TensorOutputLayout
from uniserve_worker.modeling.resources import TensorNeeds
from uniserve_worker.modeling.tensors import AttentionMode, TensorViews

from ..nn.parallel import ParallelConfig

if TYPE_CHECKING:
    from ..loader.component import CheckpointComponent
    from ..modeling.context import BuildContext
    from ..modeling.image_diffusion import ImageDiffusion
    from ..modeling.inputs import ImageProcessor


class Model(nn.Module):
    """Compose numerical modules and declare their loading and tensor contracts."""

    architecture: str
    cache_geometry: CacheGeometry
    vocab_size: int
    hidden_size: int
    text_max_tokens: int
    max_vit_grid_tokens: int = 0
    text_topology: tuple[str, ...] = ("tp",)
    text_attention_mode: AttentionMode = AttentionMode.PAGED_VARLEN
    image_processor: ImageProcessor | None = None
    generation: ImageDiffusion | None = None
    media_profile: str | None = None
    # Each logical result component declares the call and maximum numerical
    # input shape whose TensorNeeds.outputs bound its complete result. Runtime
    # resolves wire names, allocation, placement and reader lifetimes.
    output_shapes: Mapping[str, tuple[Call, Shape]] = MappingProxyType({})
    num_inference_steps: int = 0

    def __init__(self, config: Any = None, context: BuildContext | None = None) -> None:
        """Retain numerical configuration for component and geometry declarations."""

        super().__init__()
        self.config = config

    @classmethod
    def components(cls, config: Any) -> tuple[ComponentSpec, ...]:
        """Declare numerical component roles before physical placement is bound."""

        raise NotImplementedError(f"{cls.__name__} does not declare numerical components")

    @classmethod
    def validate_parallel(cls, config: Any, parallel: Mapping[str, ParallelConfig]) -> None:
        """Validate logical component names before any communication resources exist.

        Concrete models add dimension and algorithm constraints. Physical rank
        membership and distribution do not enter this numerical interface.
        """

        declarations = {component.name: component for component in cls.components(config)}
        unknown = parallel.keys() - declarations.keys()
        if unknown:
            raise ValueError(f"unknown numerical components: {sorted(unknown)}")
        for name, geometry in parallel.items():
            if any(call.call is Call.TEXT for call in declarations[name].calls) and (
                geometry.sequence_parallel.kind not in {"local", "ulysses"}
            ):
                raise ValueError(
                    "paged attention sequence execution requires Ulysses head exchange"
                )

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Describe explicit numerical views and outputs for a supported call."""

        raise NotImplementedError(f"{type(self).__name__} does not declare {call.value} tensors")

    def prepare_metadata(self, call: Call, shape: Shape, *, out: TensorViews) -> None:
        """Initialize caller-owned constants, which numerical calls subsequently read.

        Models declaring constants must implement their mathematical preparation.
        The default is an empty operation only for calls without constants.
        """

        if out or self.tensor_specs(call, shape).constants:
            raise NotImplementedError(
                f"{type(self).__name__} does not prepare {call.value} constants"
            )

    def output_layout(
        self,
        entry: str,
        output_index: int,
        *,
        frames: int | None,
        units: int | None,
        prompt_tokens: int,
    ) -> TensorOutputLayout | None:
        """Describe numerical result geometry; return None when this rank has no result."""

        return TensorOutputLayout()

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare the resident components populated by checkpoint loading."""

        raise NotImplementedError(f"{type(self).__name__} does not declare checkpoint components")


__all__ = [
    "Model",
]
