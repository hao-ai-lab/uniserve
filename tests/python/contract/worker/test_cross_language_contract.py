"""Cross-language contract tests (Cluster J).

Several Python constants must equal a Rust / FlatBuffers counterpart that today
is held in sync only by prose comments ("MUST mirror", "INC-38"). These tests
parse the canonical Rust/FlatBuffers sources and assert equality so drift fails
the build instead of surfacing as a runtime mismatch.

The tests skip (rather than fail) when the Rust sources are absent — e.g. a
Python-only checkout/wheel — so they are meaningful in the full repo and inert
elsewhere.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from uniserve_worker.contracts.op_kinds import OP_KINDS
from uniserve_worker.runtime.image_defaults import (
    SENSENOVA_DEFAULT_HEIGHT,
    SENSENOVA_DEFAULT_WIDTH,
)

pytestmark = pytest.mark.contract


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "crates").is_dir() and (parent / "uniserve_worker").is_dir():
            return parent
    return None


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def test_op_kind_vocabulary_matches_flatbuffers_schema():
    """Python OP_KINDS == the worker.fbs OpKind enum (snake_cased) (C1/INC-38)."""
    root = _repo_root()
    if root is None:
        pytest.skip("Rust crates not present in this checkout")
    fbs = root / "crates/protocol/worker-ipc-core/schema/worker.fbs"
    if not fbs.exists():
        pytest.skip("worker.fbs not present")
    text = fbs.read_text(encoding="utf-8")
    match = re.search(r"enum\s+OpKind\s*:\s*\w+\s*\{([^}]*)\}", text)
    assert match, "could not locate OpKind enum in worker.fbs"
    members = [m.strip() for m in match.group(1).split(",") if m.strip()]
    # Strip any explicit `= N` assignments flatbuffers allows.
    members = [m.split("=")[0].strip() for m in members]
    fbs_kinds = {_camel_to_snake(m) for m in members}
    assert fbs_kinds == set(OP_KINDS), (
        f"op-kind drift: fbs={sorted(fbs_kinds)} python={sorted(OP_KINDS)}"
    )


def test_image_defaults_match_rust_image_params_default():
    """Python image fallback dims == Rust ImageParams::default() (I5/INC-54)."""
    root = _repo_root()
    if root is None:
        pytest.skip("Rust crates not present in this checkout")
    core_rs = root / "crates/foundation/core/src/lib.rs"
    if not core_rs.exists():
        pytest.skip("foundation core lib.rs not present")
    text = core_rs.read_text(encoding="utf-8")
    block = re.search(r"impl Default for ImageParams\s*\{(.*?)\n\}", text, re.DOTALL)
    assert block, "could not locate impl Default for ImageParams"
    body = block.group(1)
    h = re.search(r"height:\s*(\d+)", body)
    w = re.search(r"width:\s*(\d+)", body)
    assert h and w, "could not parse default height/width"
    assert (int(h.group(1)), int(w.group(1))) == (SENSENOVA_DEFAULT_HEIGHT, SENSENOVA_DEFAULT_WIDTH), (
        f"image default drift: rust=({h.group(1)},{w.group(1)}) "
        f"python=({SENSENOVA_DEFAULT_HEIGHT},{SENSENOVA_DEFAULT_WIDTH})"
    )
