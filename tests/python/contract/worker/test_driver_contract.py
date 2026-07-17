"""Capability-schema + op-schema conformance (plan §10.5, DoD 3, 9).

The capability declaration is checked for every backend on CPU (class-level
introspection). The full caps() schema is checked against the GPU-free Stub.
"""

from __future__ import annotations

import pytest

from tests.python.fixtures.model_targets import (
    ADAPTER_MODES,
    BACKEND_CLASSES,
    CONTROL_METHODS,
    KNOWN_CONTROLS,
    KNOWN_OP_KINDS,
    KNOWN_RESOURCE_CLASSES,
    cpu_engine,
    load_backend_class,
)

pytestmark = pytest.mark.contract

ALL_BACKENDS = list(BACKEND_CLASSES)

CAPS_REQUIRED = {
    "block_size": int,
    "num_blocks": int,
    "num_layers": int,
    "scratch_capacity_tokens": int,
    "supported_ops": list,
    "max_latent_size": int,
    "latent_downsample": int,
    "bytes_per_token": int,
    "supported_controls": list,
    "adapter_mode": str,
    "execution_constraints": dict,
    "resource_classes": list,
}


def _minimal_model_config(name):
    if name == "sensenova":
        return {
            "llm_config": {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
        }
    return {}


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_capability_declaration_wellformed(name):
    cls = load_backend_class(name)
    ops = set(cls.supported_ops)
    ctrls = set(cls.supported_controls)
    assert ops, f"{name} declares no ops"
    assert ops <= KNOWN_OP_KINDS, f"{name} declares unknown ops {ops - KNOWN_OP_KINDS}"
    assert ctrls <= KNOWN_CONTROLS, f"{name} declares unknown controls {ctrls - KNOWN_CONTROLS}"
    assert cls.adapter_mode in ADAPTER_MODES


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_declared_controls_have_methods(name):
    """No control may be declared without a real method (DoD 4: no aspirational caps)."""
    cls = load_backend_class(name)
    for ctrl in cls.supported_controls:
        meth = CONTROL_METHODS[ctrl]
        assert callable(getattr(cls, meth, None)), (
            f"{name} declares control {ctrl!r} but has no {meth}() method"
        )


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_resource_classes_wellformed(name):
    cls = load_backend_class(name)
    declared = set(cls.resource_plan.classes())
    assert declared, f"{name} declares no resource classes"
    assert declared <= KNOWN_RESOURCE_CLASSES, (
        f"{name} declares unknown resource classes {declared - KNOWN_RESOURCE_CLASSES}"
    )
    # Every backend manages KV.
    assert "kv_block" in declared


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_adapter_mode_implies_lora_controls(name):
    """A non-'none' adapter mode must declare load_lora/unload_lora."""
    cls = load_backend_class(name)
    if cls.adapter_mode != "none":
        ctrls = set(cls.supported_controls)
        assert {"load_lora", "unload_lora"} <= ctrls, (
            f"{name} adapter_mode != none but does not declare load/unload_lora"
        )


def test_caps_schema_required_keys():
    caps = cpu_engine().caps().to_wire()
    for key, typ in CAPS_REQUIRED.items():
        assert key in caps, f"caps missing required key {key!r}"
        assert isinstance(caps[key], typ), (
            f"caps[{key!r}] is {type(caps[key]).__name__}, want {typ.__name__}"
        )
    ec = caps["execution_constraints"]
    assert "max_batch_ops" in ec
    # und/gen mixing is a non-negotiable invariant (§9.6); the flag is gone.
    assert "supports_mixed_op_kinds" not in ec


def test_caps_values_in_vocabulary():
    caps = cpu_engine().caps().to_wire()
    assert set(caps["supported_ops"]) <= KNOWN_OP_KINDS
    assert set(caps["supported_controls"]) <= KNOWN_CONTROLS
    assert caps["adapter_mode"] in ADAPTER_MODES


def test_caps_matches_class_declaration():
    """Instance caps() is the single source of truth's class-level declaration."""
    from uniserve_worker.server.stub import StubWorker

    caps = cpu_engine().caps()
    caps = caps.to_wire()
    assert caps["supported_ops"] == list(StubWorker.supported_ops)
    assert caps["supported_controls"] == list(StubWorker.supported_controls)
    assert caps["adapter_mode"] == StubWorker.adapter_mode


@pytest.mark.parametrize("name", ["bagel", "sensenova"])
def test_new_model_caps_match_batch_policy(name):
    cls = load_backend_class(name)
    model = cls(config=_minimal_model_config(name))
    caps = model.caps().to_wire()
    policy = model.batch_policy()
    assert caps["execution_constraints"]["max_batch_ops"] == policy.max_batch_ops
    # Mixed-mode grouping is unconditional for production models (§9.6).
    assert policy.supports_mixed_modes


@pytest.mark.parametrize("name", ["bagel", "sensenova"])
def test_new_models_declare_mixed_batch_envelopes(name):
    cls = load_backend_class(name)
    policy = cls(config=_minimal_model_config(name)).batch_policy()
    assert policy.supports_mixed_modes
    assert policy.max_batch_ops >= 8


def test_model_worker_caps_declare_mixed_batch_envelope():
    # und/gen mixing is non-negotiable (§9.6): there is no worker launch override
    # that can disable it.
    from uniserve_worker.worker.model import ModelWorker

    cls = load_backend_class("sensenova")
    caps = ModelWorker(cls(config=_minimal_model_config("sensenova"))).caps().to_wire()

    assert caps["execution_constraints"]["max_batch_ops"] >= 8
    assert "supports_mixed_op_kinds" not in caps["execution_constraints"]
