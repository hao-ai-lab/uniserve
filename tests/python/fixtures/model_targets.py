"""CPU model-target registry + contract vocabulary for worker contract tests."""

from __future__ import annotations

import importlib

import pytest

from uniserve_worker.contracts.op_kinds import OP_KINDS

# All model runtime classes by short name. Class-level introspection
# (supported_ops / supported_controls / adapter_mode / resource_plan) needs no
# GPU; full instantiation (model load) does and is exercised in the e2e GPU run.
BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "stub": ("uniserve_worker.server.stub", "StubWorker"),
    "sensenova": ("uniserve_worker.models.sensenova.model", "SenseNovaU1ForUnifiedGeneration"),
    "bagel": ("uniserve_worker.models.bagel", "BagelForUnifiedGeneration"),
}

# Sourced from the canonical contract vocabulary so a model declaring a real op
# (e.g. commit_writeback) is never reported as "unknown" by a stale local copy,
# and so this set can never drift from op_kinds.py.
KNOWN_OP_KINDS = frozenset(OP_KINDS)
KNOWN_CONTROLS = {
    "copy_blocks",
    "load_lora",
    "unload_lora",
    "free_encoder",
    "reset_prefix_cache",
    "sleep",
    "wake_up",
}
# Controls the worker serves against system-owned state (the AdapterStore and
# encoder residency); a model declares them as capability only and implements
# no method.
WORKER_SERVED_CONTROLS = {
    "load_lora",
    "unload_lora",
    "copy_blocks",
    "free_encoder",
    "reset_prefix_cache",
}
# control wire name -> engine method name (here they coincide).
CONTROL_METHODS = {name: name for name in KNOWN_CONTROLS - WORKER_SERVED_CONTROLS}
ADAPTER_MODES = {"none", "engine_wide", "per_request", "multi_adapter"}
KNOWN_RESOURCE_CLASSES = {"kv_block", "encoder_output", "image_latent", "scratch", "adapter"}


def load_backend_class(name: str):
    """Import a model runtime class.

    A genuinely-absent optional dependency (e.g. ``torch`` is not installed in
    a CPU-only env) is the only legitimate reason to skip. Anything else -- an
    ``ImportError``/``ModuleNotFoundError`` for a first-party ``uniserve_worker``
    module, a missing class attribute, or any error raised while executing the
    module body -- is real breakage of sensenova/bagel construction and MUST
    surface as a failure rather than be silently skipped.
    """
    mod, cls = BACKEND_CLASSES[name]
    try:
        return getattr(importlib.import_module(mod), cls)
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        # A first-party module failing to import is a real defect, not a
        # missing optional dependency; only skip when an external package is
        # genuinely absent.
        missing = (exc.name or "").split(".", 1)[0]
        if missing and missing != "uniserve_worker":
            pytest.skip(
                f"backend {name!r} optional dependency {missing!r} absent in this env: {exc}"
            )
        raise


def cpu_engine():
    """A fully constructible execution worker for runtime conformance."""
    from uniserve_worker.server.stub import StubWorker

    return StubWorker(block_size=256)
