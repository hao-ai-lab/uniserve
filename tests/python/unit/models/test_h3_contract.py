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


def test_parallel_placement_preserves_global_modality_rows_with_eight_sequence_owners():
    import torch

    from uniserve_worker.models.minimax_h3.layout import H3Layout
    from uniserve_worker.models.minimax_h3.weights import validate_h3_entries
    from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
    from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig, SequenceParallel

    ranks = tuple(range(7, -1, -1))
    config = ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (8,)))
    entry = ComponentConfig(ranks, config)
    layouts = []
    for rank in ranks:
        bindings = EntryBindings(
            {"denoiser": entry},
            {"denoiser": DeviceMesh(ranks, rank, config)},
            Communicator(ranks=ranks, rank=rank),
        )
        validate_h3_entries(bindings)
        layouts.append(H3Layout.build(bindings, frames=22, text_rows=128, audio_frames=8))
    for modality in ("text_indices", "video_indices", "audio_indices"):
        original = getattr(layouts[0].packed, modality)
        reconstructed = torch.cat(
            [layout.local_indices(original) + layout.local_start for layout in layouts]
        )
        torch.testing.assert_close(reconstructed, original, rtol=0, atol=0)
