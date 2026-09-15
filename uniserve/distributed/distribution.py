"""Mathematical placement of one logical tensor on a device mesh."""

from dataclasses import dataclass

from torch.distributed.tensor import Placement, Shard

from .mesh import DeviceMesh


@dataclass(frozen=True, slots=True)
class Distribution:
    """Describe tensor placements without owning storage or communication."""

    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple) or len(
            self.placements
        ) != len(self.mesh.axes):
            raise ValueError("placements must provide one entry per mesh axis")
        if any(
            not isinstance(placement, Placement)
            for placement in self.placements
        ):
            raise TypeError(
                "tensor placements must use PyTorch Placement objects"
            )

    def shard_axes(self, dim: int) -> tuple[str, ...]:
        """Return the mesh axes partitioning the specified tensor dimension."""
        return tuple(
            axis
            for axis, placement in zip(
                self.mesh.axes, self.placements, strict=True
            )
            if isinstance(placement, Shard) and placement.dim == dim
        )
