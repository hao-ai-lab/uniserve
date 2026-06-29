"""The lazy ``uniserve_worker.nn`` barrel must still resolve every public name.

De-barreling the layer package (audit finding #20) replaced the eager submodule
imports with a PEP 562 ``__getattr__``; this pins that every name advertised in
``nn.__all__`` is still importable, so the lazy form cannot silently drop an
export until some downstream model fails at runtime.
"""
from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_every_nn_public_name_resolves():
    import uniserve_worker.nn as nn

    unresolved = [name for name in nn.__all__ if not hasattr(nn, name)]
    assert unresolved == [], f"nn barrel exports that no longer resolve: {unresolved}"


def test_nn_barrel_from_import_round_trips():
    nn = importlib.import_module("uniserve_worker.nn")
    for name in nn.__all__:
        obj = getattr(nn, name)
        assert obj is not None
