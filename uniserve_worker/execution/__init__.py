"""System-owned transaction, lowering, and model execution interfaces."""

from typing import TYPE_CHECKING, Any

from .engine import (
    AdminOutcome,
    EngineBackpressure,
    EngineExecutionError,
    EnginePoisoned,
    EngineState,
    ExecutionEngine,
    PayloadConflict,
    PreLaunchRejection,
    StaleStep,
    TransactionExecutor,
)
from .lowering import (
    AttentionLayerSpec,
    GraphCapacity,
    LoweredBatch,
    LoweredSegment,
    LoweringError,
    RoleSequences,
    SegmentTableArrays,
    SegmentTableError,
    lower_rows,
    select_capacity,
)
from .transaction import (
    AdapterPayload,
    AdapterRowOutcome,
    ConformanceCase,
    ConformanceManifest,
    DistributedConfigurationError,
    ManifestError,
    PreparedTransaction,
    RankDisagreement,
    RankFanOutExecutor,
    RankMember,
    ResidentAdapter,
    StandardTransactionExecutor,
    build_manifest,
    case_set_hash,
    generate_cases,
    validate_manifest,
)

if TYPE_CHECKING:
    from .runner import ModelRunner, RunnerConfig

_RUNNER_EXPORTS = frozenset({"ModelRunner", "RunnerConfig"})


def __getattr__(name: str) -> Any:
    if name not in _RUNNER_EXPORTS:
        raise AttributeError(name)
    from . import runner

    value = getattr(runner, name)
    globals()[name] = value
    return value

__all__ = [
    "AdapterPayload",
    "AdapterRowOutcome",
    "AdminOutcome",
    "AttentionLayerSpec",
    "ConformanceCase",
    "ConformanceManifest",
    "DistributedConfigurationError",
    "EngineBackpressure",
    "EngineExecutionError",
    "EnginePoisoned",
    "EngineState",
    "ExecutionEngine",
    "GraphCapacity",
    "LoweredBatch",
    "LoweredSegment",
    "LoweringError",
    "ManifestError",
    "ModelRunner",
    "PayloadConflict",
    "PreLaunchRejection",
    "PreparedTransaction",
    "RankDisagreement",
    "RankFanOutExecutor",
    "RankMember",
    "ResidentAdapter",
    "RoleSequences",
    "RunnerConfig",
    "SegmentTableArrays",
    "SegmentTableError",
    "StaleStep",
    "StandardTransactionExecutor",
    "TransactionExecutor",
    "build_manifest",
    "case_set_hash",
    "generate_cases",
    "lower_rows",
    "select_capacity",
    "validate_manifest",
]
