"""Supported checkpoint metadata controls capabilities independently of directory names."""

import json
from pathlib import Path

import pytest

from uniserve_worker.models.minimax_h3.config import resolve_h3_contract

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3/fastvideo_inference.json"
    (tmp_path / "fastvideo_inference.json").write_text(source.read_text())
    return tmp_path


def test_full_vsa_checkpoint_resolves_t2va_and_inference_grid(checkpoint):
    contract = resolve_h3_contract(checkpoint)
    assert contract["tasks"] == ["t2va"]
    assert contract["attention"] == "vsa"
    assert contract["inference_grid"] == [1, 0.75, 0.5, 0.25, 0]
    assert contract["revision"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "ref2va"),
        ("attention_backend", "FLASH_ATTN"),
        ("transformer_forwards", 50),
        ("vsa_sparsity", 0.8),
        ("schema_version", "unknown"),
    ],
)
def test_incompatible_checkpoint_is_rejected(checkpoint, field, value):
    path = checkpoint / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        resolve_h3_contract(checkpoint)


def test_architecture_without_variant_metadata_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fastvideo_inference.json"):
        resolve_h3_contract(tmp_path)
