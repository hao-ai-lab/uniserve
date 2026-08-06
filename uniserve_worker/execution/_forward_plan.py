"""Private immutable value prepared by ``ModelExecutor`` for ``ModelRunner``."""

from __future__ import annotations

from dataclasses import dataclass

from uniserve_worker.forward import ForwardContext, ForwardRow, RouteId
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.host_staging import TensorStagingSlot


@dataclass(frozen=True, slots=True)
class ForwardBinding:
    """One physical row's registered operation identity and raw-output dtype.

    ``row_id``/``slot`` come from the row; ``output_dtype`` is the raw neural
    output dtype declared by the route row ABI; the ``(session_id, epoch,
    op_id)`` triple is the registered operation identity the row lowers from and
    ``base_version`` is the point index that operation advances from, taken from
    its registered parent.
    """

    row_id: int
    slot: int
    output_dtype: str
    session_id: int
    epoch: int
    op_id: int
    base_version: int

    def __post_init__(self) -> None:
        if self.row_id < 0 or self.slot < 0 or not self.output_dtype:
            raise ValueError("forward output binding is invalid")
        if self.session_id < 1 or self.epoch < 1 or self.op_id < 1:
            raise ValueError("forward binding operation identity is invalid")
        if self.base_version < 0:
            raise ValueError("forward binding base version is invalid")


@dataclass(frozen=True, slots=True)
class GraphKey:
    model_revision: str
    spec_digest: str
    route: RouteId
    shape: tuple[int, ...]
    dtype: str
    backend: str
    topology: str

    def __post_init__(self) -> None:
        if not self.model_revision or not self.spec_digest or not self.dtype:
            raise ValueError("graph key identities and dtype must be present")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("graph key shape dimensions must be non-negative")


@dataclass(frozen=True, slots=True)
class ForwardPlan:
    """One physical route call lowered directly from registered operations."""

    route: RouteId
    rows: tuple[ForwardRow, ...]
    context: ForwardContext
    bindings: tuple[ForwardBinding, ...]
    graph_key: GraphKey
    graph_eligible: bool
    device: str
    weights: WeightSet
    staging_slot: TensorStagingSlot | None = None

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("forward plan must contain at least one row")
        if len(self.rows) != len(self.bindings):
            raise ValueError("forward plan must contain one binding per row")
        for row, binding in zip(self.rows, self.bindings, strict=True):
            if row.row_id != binding.row_id or row.output_slot != binding.slot:
                raise ValueError("forward plan bindings are not row-aligned")
        if not self.device:
            raise ValueError("forward plan device must be present")

    @property
    def operations(self) -> tuple[tuple[int, int, int], ...]:
        """The registered operation identities this call lowers from."""

        return tuple(
            (binding.session_id, binding.epoch, binding.op_id) for binding in self.bindings
        )


__all__: list[str] = []
