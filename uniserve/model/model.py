"""Common numerical composition and checkpoint declarations for library models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torch import nn

from uniserve.distributed.parallel import ParallelConfig
from uniserve.model.components import ComponentCall

if TYPE_CHECKING:
    from uniserve.distributed.mesh import DeviceMesh
    from uniserve.loading.component import CheckpointComponent
    from uniserve.model.limits import ModelLimits
    from uniserve.nn.layer import LayerConfig


class Model(nn.Module):
    """Compose numerical modules and declare their loading and tensor contracts."""

    architecture: str
    minimum_cuda_capability: tuple[int, int] | None = None

    def __init__(
        self,
        config: Any = None,
        *,
        parallel: Mapping[str, ParallelConfig] | None = None,
        meshes: Mapping[str, DeviceMesh] | None = None,
        layers: Mapping[str, LayerConfig] | None = None,
        limits: ModelLimits | None = None,
    ) -> None:
        """Retain configuration; concrete compositions consume borrowed layer bindings.

        The common construction signature allows checkpoint loading of concrete
        models. A plain module composition needs no distributed resources.
        """

        super().__init__()
        self.config = config

    @classmethod
    def component_calls(cls, config: Any) -> tuple[ComponentCall, ...]:
        """Declare numerical roles; a plain composition has no serving entries."""

        return ()

    @classmethod
    def validate_parallel(cls, config: Any, parallel: Mapping[str, ParallelConfig]) -> None:
        """Validate logical component names before any communication resources exist.

        Concrete models add dimension and algorithm constraints. Physical rank
        membership and distribution do not enter this numerical interface.
        """

        calls = cls.component_calls(config)
        components = {call.component for call in calls}
        unknown = parallel.keys() - components
        if unknown:
            raise ValueError(f"unknown numerical components: {sorted(unknown)}")
        for path, partition in parallel.items():
            if any(call.component == path and call.method == "forward" for call in calls) and (
                partition.sequence_parallel.kind not in {"local", "ulysses"}
            ):
                raise ValueError(
                    "paged attention sequence execution requires Ulysses head exchange"
                )

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare the resident components populated by checkpoint loading."""

        raise NotImplementedError(f"{type(self).__name__} does not declare checkpoint components")


__all__ = [
    "Model",
]
