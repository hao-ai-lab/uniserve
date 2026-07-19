"""Behavioral import contracts for the execution module boundaries."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_transaction_and_lowering_interfaces_import_without_torch() -> None:
    script = """
import sys

sys.modules["torch"] = None

from uniserve_worker import execution

assert execution.ExecutionEngine
assert execution.GraphCapacity
assert execution.SegmentTableArrays
assert execution.StandardTransactionExecutor
"""
    subprocess.run([sys.executable, "-c", script], check=True)
