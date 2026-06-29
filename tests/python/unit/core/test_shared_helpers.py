"""Unit tests for the shared worker helpers introduced for INC-113/INC-114."""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.env import env_flag, flag_from_value
from uniserve_worker.foundation.sizing import ceil_div

pytestmark = pytest.mark.unit


def test_flag_from_value_allowlist_semantics():
    for token in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        assert flag_from_value(token) is True, token
    for token in ("0", "false", "no", "off"):
        assert flag_from_value(token) is False, token
    # Unset / empty -> default; unrecognized -> default (conservative).
    assert flag_from_value(None) is False
    assert flag_from_value("") is False
    assert flag_from_value("garbage") is False
    assert flag_from_value(None, default=True) is True
    assert flag_from_value("garbage", default=True) is True
    # Explicit false tokens always win over a True default.
    assert flag_from_value("0", default=True) is False


def test_env_flag_reads_environment(monkeypatch):
    monkeypatch.setenv("UNISERVE_TEST_FLAG", "on")
    assert env_flag("UNISERVE_TEST_FLAG") is True
    monkeypatch.setenv("UNISERVE_TEST_FLAG", "off")
    assert env_flag("UNISERVE_TEST_FLAG") is False
    monkeypatch.delenv("UNISERVE_TEST_FLAG", raising=False)
    assert env_flag("UNISERVE_TEST_FLAG") is False
    assert env_flag("UNISERVE_TEST_FLAG", default=True) is True


def test_ceil_div_rounds_up_and_guards_zero_divisor():
    assert ceil_div(0, 256) == 0
    assert ceil_div(1, 256) == 1
    assert ceil_div(256, 256) == 1
    assert ceil_div(257, 256) == 2
    # Divisor clamps to >= 1 instead of raising ZeroDivisionError.
    assert ceil_div(5, 0) == 5
    assert ceil_div(0, 0) == 0
