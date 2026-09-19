"""A model declares how each of its components divides work over ranks."""

import json

import pytest

from uniserve_models.placement import (
    Partition,
    architecture_of,
    divisions_of,
    entries_for,
)

pytestmark = pytest.mark.unit


def _checkpoint(directory, metadata: dict, name: str = "config.json"):
    """Write the root metadata a checkpoint names its architecture in."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(metadata))
    return directory


def _media_checkpoint(directory):
    """A modular media checkpoint names its pipeline, not its transformer."""
    return _checkpoint(
        directory,
        {"_class_name": "MiniMaxH3ModularPipeline"},
        name="modular_model_index.json",
    )


def test_a_decoder_only_checkpoint_divides_one_component_by_tensor(tmp_path):
    root = _checkpoint(tmp_path, {"architectures": ["Qwen3ForCausalLM"]})

    assert architecture_of(root) == "Qwen3ForCausalLM"
    assert entries_for(root, 4) == {
        "model": {
            "ranks": [0, 1, 2, 3],
            "parallel_config": {"tensor_parallel_size": 4},
        }
    }


def test_a_media_checkpoint_declares_the_division_each_component_admits(
    tmp_path,
):
    root = _media_checkpoint(tmp_path)

    assert divisions_of(architecture_of(root)) == {
        "denoiser": Partition.SEQUENCE,
        "text_encoder": Partition.TENSOR,
        "video_decoder": Partition.MEDIA_UNITS,
        "audio_decoder": Partition.MEDIA_UNITS,
        "muxer": Partition.NOTHING,
    }


def test_each_division_places_the_component_it_describes(tmp_path):
    entries = entries_for(_media_checkpoint(tmp_path), 8)

    # A sequence division exchanges attention heads across every rank.
    assert entries["denoiser"] == {
        "ranks": list(range(8)),
        "parallel_config": {
            "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 8}
        },
    }
    # A tensor division shards each tensor across every rank.
    assert entries["text_encoder"] == {
        "ranks": list(range(8)),
        "parallel_config": {"tensor_parallel_size": 8},
    }
    # A media-unit division gives every rank its own unit of the timeline.
    for name in ("video_decoder", "audio_decoder"):
        assert entries[name] == {
            "ranks": list(range(8)),
            "parallel_config": {},
            "distribution": "temporal_units",
            "units_per_rank": 1,
        }
    # A component that divides nothing is held whole by one rank.
    assert entries["muxer"] == {"ranks": [0], "parallel_config": {}}


def test_a_division_is_the_models_and_the_placement_is_the_instances(tmp_path):
    root = _media_checkpoint(tmp_path)
    divisions = divisions_of(architecture_of(root))

    for ranks in (1, 2, 4, 8):
        # The declaration does not depend on how wide the instance is.
        assert divisions_of(architecture_of(root)) == divisions
        entries = entries_for(root, ranks)
        sequence = entries["denoiser"]["parallel_config"]["sequence_parallel"]
        assert sequence["ulysses_degree"] == ranks
        assert entries["text_encoder"]["parallel_config"] == {
            "tensor_parallel_size": ranks
        }
        assert entries["video_decoder"]["ranks"] == list(range(ranks))
        # A component that divides nothing stays on one rank at any width.
        assert entries["muxer"]["ranks"] == [0]


def test_every_supported_architecture_declares_known_divisions():
    # A package added without a declaration, or naming a division shared code
    # cannot place, is otherwise found only when a rank looks for a component.
    from uniserve_models.placement import _PACKAGES

    for architecture in _PACKAGES:
        divisions = divisions_of(architecture)
        assert divisions, f"{architecture} declares no component"
        assert all(
            isinstance(partition, Partition) for partition in divisions.values()
        )


def test_an_unsupported_or_ambiguous_architecture_is_refused(tmp_path):
    with pytest.raises(ValueError):
        entries_for(
            _checkpoint(tmp_path / "a", {"architectures": ["Other"]}), 1
        )

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
