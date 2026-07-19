"""Public ``uniserve_worker.nn`` exports resolve through the package API."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_every_nn_public_name_resolves():
    import uniserve_worker.nn as nn

    unresolved = [name for name in nn.__all__ if not hasattr(nn, name)]
    assert unresolved == [], f"unresolvable nn exports: {unresolved}"
