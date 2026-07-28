"""Private immutable value prepared by ``ModelExecutor`` for ``ModelRunner``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from uniserve_worker.forward import ForwardContext, ForwardRow, RouteId
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.host_staging import TensorStagingSlot


class OutputKind(StrEnum):
    TOKEN = "token"
    FLOW = "flow"
    ENCODE = "encode"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class OutputSlot:
    row_id: int
    slot: int
    kind: OutputKind
    dtype: str

    def __post_init__(self) -> None:
        if self.row_id < 0 or self.slot < 0 or not self.dtype:
            raise ValueError("forward output declaration is invalid")


@dataclass(frozen=True, slots=True)
class TransactionId:
    operations: tuple[tuple[int, int, int], ...]
    base_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.operations or any(
            session_id < 1 or epoch < 1 or operation_id < 1
            for session_id, epoch, operation_id in self.operations
        ):
            raise ValueError("forward transaction identity is invalid")
        if len(self.base_versions) != len(self.operations) or any(
            version < 0 for version in self.base_versions
        ):
            raise ValueError("forward transaction versions are invalid")


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
    """One physical route and its complete transaction-bounded preparation."""

    route: RouteId
    rows: tuple[ForwardRow, ...]
    context: ForwardContext
    outputs: tuple[OutputSlot, ...]
    transaction: TransactionId
    graph_key: GraphKey
    graph_eligible: bool
    device: str
    weights: WeightSet
    staging_slot: TensorStagingSlot | None = None

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("forward plan must contain at least one row")
        if len(self.rows) != len(self.outputs):
            raise ValueError("forward plan must contain one output slot per row")
        for row, output in zip(self.rows, self.outputs, strict=True):
            if row.row_id != output.row_id or row.output_slot != output.slot:
                raise ValueError("forward plan output slots are not row-aligned")
        if not self.device:
            raise ValueError("forward plan device must be present")


__all__: list[str] = []
