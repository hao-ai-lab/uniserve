"""Closed model-family execution contracts on the canonical registry path."""

from __future__ import annotations

import pytest

from uniserve_worker.contracts import OperationTag
from uniserve_worker.contracts.model_family import (
    FamilyExecutionContract,
    ModelFamilyDescriptor,
    ModelOperationSet,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.models.cache_registrations import qwen3_cache_registration
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.registry import ModelRegistry
from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

pytestmark = pytest.mark.contract

_GEOMETRY = dict(
    layer_count=2,
    query_heads=8,
    kv_heads=2,
    qk_head_dim=64,
    value_head_dim=64,
)

_MODELS = (
    Qwen3ForCausalLM,
    BagelForUnifiedGeneration,
    SenseNovaU1ForUnifiedGeneration,
)


def test_canonical_registry_resolves_family_execution_descriptors():
    registry = ModelRegistry()
    for model in _MODELS:
        registry.register(model, names=model.architectures)

    assert registry.resolve_descriptor(("Qwen3ForCausalLM",)).family == "qwen3"
    assert registry.resolve_descriptor(("bagel",)).family == "bagel"
    assert registry.resolve_descriptor(("neo_chat",)).family == "sensenova"


@pytest.mark.parametrize("model", _MODELS)
def test_advertised_operations_equal_the_family_cache_schema(model):
    descriptor = ModelFamilyDescriptor.from_model_class(model)
    registration = descriptor.build_cache_registration(**_GEOMETRY)

    assert descriptor.operation_tags == frozenset(
        region.operation_tag for region in registration.schema.regions
    )


def test_operation_vocabulary_maps_workload_shapes_to_general_operations():
    descriptor = ModelFamilyDescriptor.from_model_class(SenseNovaU1ForUnifiedGeneration)

    assert descriptor.operation_tags == frozenset(OperationTag)


def test_cache_schema_cannot_understate_the_advertised_operation_set():
    contract = FamilyExecutionContract(
        operations=ModelOperationSet.from_kinds(("prefill_und", "denoise_gen")),
        cache_registration_factory=qwen3_cache_registration,
    )

    with pytest.raises(WorkerError, match="advertises"):
        contract.build_cache_registration(family="invalid", **_GEOMETRY)
