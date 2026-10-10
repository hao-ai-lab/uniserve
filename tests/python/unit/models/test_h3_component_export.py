"""A FastH3 export reads the components it omits at its pinned base revision.

FastH3 OmniRef ships its DiT partition and schedulers and pins, in
``base_model_revision``, the MiniMax-H3 revision that supplies the text
encoder, tokenizer, processor and VAEs. The loader reads every directory the
export omits from the Hub snapshot of that revision.
"""

import json
import shutil
from pathlib import Path

import pytest

from uniserve.diffusion import BlockGrid
from uniserve.loading import Config as IOConfig
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import SparseAttention

pytestmark = pytest.mark.unit

REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


@pytest.fixture
def checkpoints(tmp_path):
    """An OmniRef-like export and the files of its base revision."""
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
                "model_type": "ref2va",
                "base_model_revision": f"hf://MiniMaxAI/MiniMax-H3@{REVISION}",
                "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "grid_max_t": 0.999,
                "vsa_ref_keep_rate": 0.1,
                "vsa_sparsity": 0.9,
                "vsa_tile_size": 128,
            }
        )
    )
    return export, base


def test_export_reads_the_directories_it_omits_at_the_pinned_revision(
    checkpoints, tmp_path, monkeypatch
):
    from huggingface_hub.errors import EntryNotFoundError

    export, base = checkpoints
    snapshot = tmp_path / "hub" / REVISION
    published = sorted(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file()
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
    # Dummy reads resolve no payload; the configuration is complete.
    config = models.read_config(
        export, io=IOConfig(mode="dummy"), modules=frozenset()
    )

    assert set(config.model.denoisers) == {"transformer_ref"}
    denoiser = config.model.denoisers["transformer_ref"]
    assert isinstance(denoiser.grids["video"], BlockGrid)
    assert denoiser.attention == SparseAttention(
        tile=128, sparsity=0.9, reference_keep=0.1
    )
    # The export's own schedulers and DiT, the base's tokenizer.
    assert config.tokenizer == snapshot / "tokenizer"
    assert set(config.entry_points) >= {"text_encoder", "transformer_ref"}


def test_a_checkpoint_holding_every_component_reads_no_base(checkpoints):
    _, base = checkpoints
    config = models.read_config(
        base, io=IOConfig(mode="dummy"), modules=frozenset()
    )
    assert set(config.model.denoisers) == {"transformer"}
    assert config.tokenizer == base / "tokenizer"
