"""Strict target model registry (Stage 6 slice, dormant)."""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.execution import OperationTag
from uniserve_worker.models.target_registry import (
    ModelRegistration,
    RegistryError,
    target_registry,
)

pytestmark = pytest.mark.contract

_GEOMETRY = dict(
    layer_count=2, query_heads=8, kv_heads=2, qk_head_dim=64, value_head_dim=64
)


def test_known_architectures_and_aliases_resolve_explicitly():
    registry = target_registry()
    assert registry.resolve(("Qwen3ForCausalLM",)).family == "qwen3"
    assert registry.resolve(("Qwen3MoeForCausalLM",)).family == "qwen3"
    assert registry.resolve(("bagel",)).family == "bagel"
    assert registry.resolve(("neo_chat", "unrelated")).family == "sensenova"


def test_unknown_and_ambiguous_architectures_fail_closed():
    registry = target_registry()
    with pytest.raises(RegistryError, match="no explicit registration"):
        registry.resolve(("LlamaForCausalLM",))
    with pytest.raises(RegistryError, match="several families"):
        registry.resolve(("Qwen3ForCausalLM", "BAGEL"))


def test_duplicate_architecture_registration_is_rejected():
    registry = target_registry()
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(
            ModelRegistration(
                family="qwen3-copy",
                architectures=("Qwen3ForCausalLM",),
                operations=frozenset({OperationTag.SEQUENCE_STEP}),
                cache_registration=registry.resolve(
                    ("Qwen3ForCausalLM",)
                ).cache_registration,
            )
        )


def test_advertised_operations_must_equal_the_lowerable_set():
    registry = target_registry()
    for architecture in ("Qwen3ForCausalLM", "BAGEL", "NEOChatModel"):
        registry.resolve((architecture,)).validate(**_GEOMETRY)
    qwen3 = registry.resolve(("Qwen3ForCausalLM",))
    overclaiming = ModelRegistration(
        family="qwen3",
        architectures=("Overclaim",),
        operations=frozenset({OperationTag.SEQUENCE_STEP, OperationTag.FLOW_STEP}),
        cache_registration=qwen3.cache_registration,
    )
    with pytest.raises(RegistryError, match="advertises"):
        overclaiming.validate(**_GEOMETRY)
