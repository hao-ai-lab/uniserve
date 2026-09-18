"""A checkpoint's architecture declares the components serving must place."""

import json

import pytest

from uniserve_models.placement import architecture_of, entries_for

pytestmark = pytest.mark.unit


def _checkpoint(directory, metadata: dict, name: str = "config.json"):
    """Write the metadata a checkpoint declares its architecture in."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(metadata))
    return directory


def test_a_decoder_only_checkpoint_declares_one_component(tmp_path):
    root = _checkpoint(tmp_path, {"architectures": ["Qwen3ForCausalLM"]})

    assert architecture_of(root) == "Qwen3ForCausalLM"
    assert entries_for(root, 4) == {
        "model": {
            "ranks": [0, 1, 2, 3],
            "parallel_config": {"tensor_parallel_size": 4},
        }
    }


def test_a_video_checkpoint_declares_the_partition_each_component_admits(
    tmp_path,
):
    root = _checkpoint(
        tmp_path / "transformer",
        {"_class_name": "MiniMaxH3Transformer3DModel"},
    ).parent
    entries = entries_for(root, 8)

    # The denoiser divides its sequence by attention head across every rank.
    assert entries["denoiser"]["parallel_config"]["sequence_parallel"] == {
        "kind": "ulysses",
        "ulysses_degree": 8,
    }
    # The text encoder divides its tensors instead.
    assert entries["text_encoder"]["parallel_config"] == {
        "tensor_parallel_size": 8
    }
    # Both decoders divide their output timeline into one media unit per rank,
    # which is the division section 5.5 requires of them.
    for name in ("video_decoder", "audio_decoder"):
        assert entries[name]["distribution"] == "temporal_units"
        assert entries[name]["units_per_rank"] == 1
        assert entries[name]["ranks"] == list(range(8))
    # The muxer assembles the artifact and divides nothing, so it is alone.
    assert entries["muxer"]["ranks"] == [0]


def test_a_declared_partition_follows_the_rank_count(tmp_path):
    root = _checkpoint(
        tmp_path / "transformer",
        {"_class_name": "MiniMaxH3Transformer3DModel"},
    ).parent

    for ranks in (1, 2, 4, 8):
        entries = entries_for(root, ranks)
        degree = entries["denoiser"]["parallel_config"]["sequence_parallel"]
        assert degree["ulysses_degree"] == ranks
        assert entries["text_encoder"]["parallel_config"] == {
            "tensor_parallel_size": ranks
        }
        assert entries["video_decoder"]["ranks"] == list(range(ranks))
        # A component placed alone stays alone however wide the instance is.
        assert entries["muxer"]["ranks"] == [0]


def test_an_unsupported_or_ambiguous_architecture_is_refused(tmp_path):
    unsupported = _checkpoint(tmp_path / "a", {"architectures": ["Other"]})
    with pytest.raises(ValueError):
        entries_for(unsupported, 1)

    with pytest.raises(ValueError):
        entries_for(
            _checkpoint(
                tmp_path / "b",
                {"architectures": ["Qwen3ForCausalLM", "Other"]},
            ),
            1,
        )

    (tmp_path / "c").mkdir(parents=True)
    with pytest.raises(ValueError):
        entries_for(tmp_path / "c", 1)


def test_a_placement_requires_a_positive_rank_count(tmp_path):
    root = _checkpoint(tmp_path, {"architectures": ["Qwen3ForCausalLM"]})

    for ranks in (0, -1):
        with pytest.raises(ValueError):
            entries_for(root, ranks)
