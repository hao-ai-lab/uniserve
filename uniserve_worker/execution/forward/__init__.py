"""Unified forward execution interfaces."""
from __future__ import annotations

from .batch import ForwardBatchBuilder
from .descriptor import ForwardModelDescriptor, ForwardModelModules, descriptor_from_model
from .executor import ForwardExecutor
from .fallback import (
    EagerFallbackReason,
    EagerFallbackRecorder,
    EagerFallbackWarning,
    ForwardGraphPolicy,
    StrictForwardGraphError,
)
from .plan import (
    CacheSpanPlan,
    ForwardModality,
    ForwardOutputKind,
    ForwardOutputSlot,
    ForwardPlan,
    ForwardPlanBuilder,
    ForwardPostprocessPolicy,
    ForwardResultProjection,
    ForwardRowPlan,
    ForwardRuntimeHandles,
    ForwardSegmentClass,
    ForwardSegmentPlan,
    ForwardShapeSummary,
    KvWritePolicy,
    TextTokenSpanPlan,
)
from .postprocess import ForwardPostprocessor
from .result import (
    CommitNeuralResult,
    DenoiseBranchKey,
    DenoiseVelocityResult,
    EncodeResult,
    ForwardGraphExecutionInfo,
    ForwardResult,
    TextLogitsResult,
)
from .runner import EagerForwardRunner, ForwardRunner

__all__ = [
    "CacheSpanPlan",
    "CommitNeuralResult",
    "DenoiseBranchKey",
    "DenoiseVelocityResult",
    "EagerFallbackReason",
    "EagerFallbackRecorder",
    "EagerFallbackWarning",
    "EagerForwardRunner",
    "EncodeResult",
    "ForwardBatchBuilder",
    "ForwardExecutor",
    "ForwardGraphExecutionInfo",
    "ForwardGraphPolicy",
    "ForwardModality",
    "ForwardModelDescriptor",
    "ForwardModelModules",
    "ForwardOutputKind",
    "ForwardOutputSlot",
    "ForwardPlan",
    "ForwardPlanBuilder",
    "ForwardPostprocessPolicy",
    "ForwardPostprocessor",
    "ForwardResult",
    "ForwardResultProjection",
    "ForwardRowPlan",
    "ForwardRunner",
    "ForwardRuntimeHandles",
    "ForwardSegmentClass",
    "ForwardSegmentPlan",
    "ForwardShapeSummary",
    "KvWritePolicy",
    "StrictForwardGraphError",
    "TextLogitsResult",
    "TextTokenSpanPlan",
    "descriptor_from_model",
]
