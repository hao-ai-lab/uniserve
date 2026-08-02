from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "check_generation_runtime_contract",
        ROOT / "scripts" / "check_generation_runtime_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract_module()


def test_canonical_generation_runtime_contract_is_self_consistent() -> None:
    report = contract.validate_contract()
    assert report["valid"] is True
    assert report["expanded_point_count"] == 46
    assert len(report["stochastic_profile_fingerprint"]) == 64
    assert len(report["fingerprint"]) == 64


def test_stochastic_profile_contract_binds_sampling_and_load_semantics() -> None:
    config = contract.load_config(ROOT / contract.PROFILE_PATH)
    references = contract._load_json(contract.REFERENCE_PATH)
    changed = deepcopy(config)
    changed["benchmarks"]["main"]["points"][
        "sensenova_ueval_stochastic_interleave_uniserve"
    ]["harness"]["temperature"] = 0.8
    with pytest.raises(contract.ContractError, match="request controls"):
        contract._validate_stochastic_profile(changed, references)
