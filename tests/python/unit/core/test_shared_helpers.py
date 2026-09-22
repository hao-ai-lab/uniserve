"""Behavior tests for shared worker helpers."""

from __future__ import annotations

import pytest

from uniserve.env import flag_from_value
from uniserve.math import ceil_div

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


def test_ceil_div_rounds_up_and_rejects_nonpositive_divisors():
    assert ceil_div(0, 256) == 0
    assert ceil_div(1, 256) == 1
    assert ceil_div(256, 256) == 1
    assert ceil_div(257, 256) == 2
    for divisor in (0, -4):
        with pytest.raises(ValueError, match="positive divisor"):
            ceil_div(5, divisor)
