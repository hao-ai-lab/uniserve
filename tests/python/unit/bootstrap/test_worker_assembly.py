from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

import uniserve_worker.bootstrap.assembly as assembly
from uniserve_worker.bootstrap.cli import create_worker_cli_parser
from uniserve_worker.bootstrap.config import WorkerLaunchConfig
from uniserve_worker.bootstrap.plan import (
    WorkerImplementation,
    resolve_worker_plan,
)
from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.outputs import EncodeOutput
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.server.worker_kind import WorkerKind
from uniserve_worker.worker.encoder import EncoderWorker
from uniserve_worker.worker.sampler import SamplerWorker

pytestmark = pytest.mark.unit


def _parse_launch(*arguments: str) -> WorkerLaunchConfig:
    namespace = create_worker_cli_parser().parse_args(
        [
            "--service-name",
            "worker-test",
            "--ipc-payload-cap",
            "4096",
            *arguments,
        ]
    )
    return WorkerLaunchConfig.from_namespace(namespace)


def test_worker_cli_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import uniserve_worker.main; assert 'torch' not in sys.modules"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_worker_plan_resolves_each_role_to_an_executable_worker_kind():
    assert resolve_worker_plan(WorkerKind.PREFILL).implementation is WorkerImplementation.MODEL
    assert resolve_worker_plan(WorkerKind.ENCODER).implementation is WorkerImplementation.ENCODER
    assert resolve_worker_plan(WorkerKind.SAMPLER).implementation is WorkerImplementation.SAMPLER


def test_model_free_role_does_not_require_model_path():
    launch = _parse_launch(
        "--worker-kind",
        "sampler",
        "--transfer-backend",
        "shm",
    )

    assert launch.worker_kind is WorkerKind.SAMPLER
    assert launch.model is None


def test_cross_process_role_rejects_process_local_transport():
    with pytest.raises(ValueError, match="cross-process logits transport"):
        _parse_launch("--worker-kind", "sampler")


def test_mesh_parser_rejects_unknown_configuration():
    with pytest.raises(ValueError, match="unknown --mesh key"):
        _parse_launch(
            "--no-model",
            "--allow-stub",
            "--mesh",
            "mystery=value",
        )


def test_worker_contract_never_fabricates_missing_operations():
    with pytest.raises(WorkerError) as error:
        SamplerWorker(allowed_ops=frozenset({"decode_und"}))

    assert error.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_pipeline_depth_is_part_of_worker_contract():
    worker = SamplerWorker(pipeline_depth=3)

    assert worker.caps().pipeline_depth == 3
    assert worker.caps().to_wire()["pipeline_depth"] == 3


class _VisionModel(UniModel):
    num_layers = 1
    encoder_cache_budget = 4

    def encode_image(self, pixels, grid, op=None):
        return EncodeOutput(
            req_id=int(op["req_id"]),
            encoder_handle=7,
            num_tokens=1,
        )


def test_encoder_assembly_executes_the_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
):
    launch = _parse_launch(
        "--worker-kind",
        "encoder",
        "--model",
        "/model",
    )
    monkeypatch.setattr(
        "uniserve_worker.backends.attention.init_attention_backends",
        lambda: (),
    )
    monkeypatch.setattr(
        "uniserve_worker.server.distributed.build_device_mesh",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "uniserve_worker.nn.mesh.set_current_mesh",
        lambda _mesh: None,
    )
    monkeypatch.setattr(
        "uniserve_worker.foundation.runtime_config.set_execution_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "uniserve_worker.bootstrap.model_loader.load_worker_model",
        lambda _request: SimpleNamespace(
            model=_VisionModel(),
            descriptor=None,
        ),
    )

    worker = assembly.assemble_worker(launch)

    assert isinstance(worker, EncoderWorker)
    assert worker.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 3}],
            "ops": [{"req_id": 3, "kind": "vit_encode", "mm_hash": 11}],
        }
    ) == {
        "step_id": 1,
        "per_seq": [{"req_id": 3, "encoder_handle": 7, "num_tokens": 1}],
    }
