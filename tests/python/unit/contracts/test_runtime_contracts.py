from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _contract_module():
    spec = importlib.util.spec_from_file_location(
        "check_runtime_contracts", ROOT / "scripts" / "check_runtime_contracts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_contracts_resolve_to_one_executable_matrix() -> None:
    report = _contract_module().validate()
    assert report["valid"] is True
    assert report["expanded_point_count"] == 46
    assert report["qualification_performance_points"] == [
        "sensenova-uniserve-t2i-c32",
        "sensenova-uniserve-i2t-c32",
        "sensenova-uniserve-interleave-c4",
        "sensenova-uniserve-stochastic-interleave-c4",
    ]
    assert len(report["fingerprint"]) == 64
