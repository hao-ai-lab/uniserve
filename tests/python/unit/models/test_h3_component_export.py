"""A component export loads its other components from its pinned base.

FastH3 OmniRef ships its DiT partition and schedulers and pins the MiniMax-H3
revision that supplies the text encoder, tokenizer, processor and VAEs. A
local copy of that base is accepted only when its Hugging Face download
records place every supplied file at the pinned revision.
"""

import json
import shutil
from pathlib import Path

import pytest

from uniserve.loading import Config as IOConfig
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import PddGrid, SparseAttention

pytestmark = pytest.mark.unit

REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
SUPPLIED = ("text_encoder", "vae", "audio_vae", "tokenizer", "processor")


def _record(root: Path, name: str, commit: str) -> None:
    # huggingface_hub's local-directory download record: commit, etag and
    # timestamp lines.
    path = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{commit}\n0123abcd\n1790914448.68\n")


@pytest.fixture
def checkpoints(tmp_path):
    """An OmniRef-like export and a local base recorded at its revision."""
    fixture = Path(__file__).parents[2] / "fixtures/models/fasth3"
    base, export = tmp_path / "base", tmp_path / "export"
    shutil.copytree(fixture, base)
    (base / "fastvideo_inference.json").unlink()
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel", {}],
            }
        )
    )
    for directory in ("tokenizer", "processor"):
        (base / directory).mkdir()
        (base / directory / "tokenizer_config.json").write_text("{}")
    for directory in SUPPLIED:
        for path in sorted((base / directory).rglob("*")):
            _record(base, path.relative_to(base).as_posix(), REVISION)

    export.mkdir()
    for directory in ("scheduler", "audio_scheduler"):
        shutil.copytree(fixture / directory, export / directory)
    (export / "transformer_ref").mkdir()
    transformer = json.loads((fixture / "transformer/config.json").read_text())
    (export / "transformer_ref/config.json").write_text(
        json.dumps({**transformer, "pdd_steps": 32})
    )
    (export / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    (export / "fastvideo_inference.json").write_text(
        json.dumps(
            {
                "schema_version": "fasth3-inference-contract-v1",
                "model_type": "ref2va",
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "conditioning": "fixed_ordered_references_target_only_flow",
                "base_model_revision": f"hf://MiniMaxAI/MiniMax-H3@{REVISION}",
                "transformer_component": "transformer_ref",
                "guidance_scale": 1.0,
                "pdd_steps": 32,
                "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "transformer_forwards": 8,
                "num_inference_steps": 8,
                "grid_max_t": 0.999,
                "video_scheduler_shift": 12.0,
                "audio_scheduler_shift": 3.0,
                "vsa_ref_policy": "p2_multi_region",
                "vsa_ref_keep_rate": 0.1,
                "vsa_sparsity": 0.9,
                "vsa_tile_size": 128,
            }
        )
    )
    return export, base


def _read(export, base):
    # Dummy reads resolve no payload; the configuration is complete.
    return models.read_config(
        export, io=IOConfig(mode="dummy"), modules=frozenset(), base=base
    )


def test_export_reads_its_other_components_from_the_base(checkpoints):
    export, base = checkpoints
    config = _read(export, base)

    assert set(config.model.denoisers) == {"reference_denoiser"}
    denoiser = config.model.denoisers["reference_denoiser"]
    assert isinstance(denoiser.schedule, PddGrid)
    assert denoiser.attention == SparseAttention(
        tile=128, sparsity=0.9, reference_keep=0.1
    )
    assert denoiser.max_sequence_rows == 131_072
    assert config.tokenizer == base / "tokenizer"
    # The identity names the export; the base is pinned by its revision.
    assert config.checkpoint_identity == models.checkpoint_identity(export)
    assert set(config.entry_points) >= {"text_encoder", "reference_denoiser"}


def test_base_recorded_at_another_revision_is_refused(checkpoints):
    export, base = checkpoints
    _record(base, "vae/config.json", "42ed227ee7df40d41602854ae760620d6eb651fe")
    with pytest.raises(ValueError, match="vae/config.json at 42ed227e"):
        _read(export, base)


def test_base_without_download_records_is_refused(checkpoints):
    export, base = checkpoints
    (base / "text_encoder" / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="no Hugging Face download record"):
        _read(export, base)


def test_checkpoint_without_a_pinned_base_takes_none(checkpoints):
    _, base = checkpoints
    with pytest.raises(ValueError, match="pins no base"):
        _read(base, base)


def test_export_without_a_local_base_reads_the_pinned_hub_revision(
    checkpoints, tmp_path, monkeypatch
):
    from huggingface_hub.errors import EntryNotFoundError

    export, base = checkpoints
    snapshot = tmp_path / "hub" / REVISION
    published = sorted(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.relative_to(base).parts[0] != ".cache"
    )

    # The Hub is an external service: it serves the base's files at the
    # pinned commit only.
    def download(*, repo_id, filename, cache_dir, revision):
        assert (repo_id, revision) == ("MiniMaxAI/MiniMax-H3", REVISION)
        if filename not in published:
            raise EntryNotFoundError(f"{filename} is not published")
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / filename, target)
        return str(target)

    def files(self, *, repo_id, revision):
        assert (repo_id, revision) == ("MiniMaxAI/MiniMax-H3", REVISION)
        return published

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", files)
    config = models.read_config(
        export, io=IOConfig(mode="dummy"), modules=frozenset()
    )

    assert set(config.model.denoisers) == {"reference_denoiser"}
    assert config.tokenizer == snapshot / "tokenizer"
