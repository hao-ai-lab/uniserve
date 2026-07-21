"""Resource-plane conformance (plan §10.5, DoD 6): drop-cleanup + residency.

Drop-cleanup is the worker half of the resource-consistency contract: after a
request is dropped, the worker retains no per-request state. Run against the
GPU-free Stub (the reference) and the worker-side ResourceRuntime.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.server.stub import StubUniModel, StubWorker
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.integration


def _run_request(engine, rid):
    engine.execute(
        seal_batch(
            rid,
            [
                {
                    "req_id": rid,
                    "kind": "prefill_und",
                    "modality": "und",
                    "pos_range": [0, 3],
                    "token_ids": [1, 2, 3],
                }
            ],
            new_reqs=[{"req_id": rid, "image": {"steps": 2}}],
        )
    )


def test_drop_request_leaves_no_resident_state():
    engine = StubWorker(block_size=256)
    _run_request(engine, 1)
    _run_request(engine, 2)
    sessions = engine.model_executor.sessions
    assert 1 in sessions and 2 in sessions
    engine.drop_request(1)
    assert 1 not in sessions, "dropped request must leave no worker-resident state"
    # request 2 untouched
    assert 2 in sessions
    assert sessions.get(2).kv_length("stub_emitted") == 1


def test_model_executor_default_resource_runtime_enforces_model_totals():
    from uniserve_worker.contracts.model_protocols import UniModel
    from uniserve_worker.contracts.resource_plan import ResourcePlan
    from uniserve_worker.execution import ExecutorConfig, ModelExecutor

    class TinyBlockModel(UniModel):
        resource_plan = ResourcePlan(kv_block="per_block")
        num_blocks = 1

        def forward(self, batch):  # pragma: no cover - admission fails first.
            raise AssertionError("unreachable")

    runner = ModelExecutor(TinyBlockModel(), config=ExecutorConfig(simulation=True))
    with pytest.raises(WorkerError) as exc:
        runner.execute(
            seal_batch(
                1,
                [{"req_id": 1, "kind": "prefill_und", "token_ids": [1], "pos_range": [0, 1]}],
                new_reqs=[{"req_id": 1, "block_ids": [1, 2]}],
            )
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert runner.resource_runtime.used("kv_block") == 0
    assert 1 not in runner.sessions


def test_model_executor_resource_runtime_tracks_blocks_latents_and_drop():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.execute(
        seal_batch(
            1,
            [
                {
                    "req_id": 7,
                    "kind": "prefill_und",
                    "pos_range": [0, 2],
                    "token_ids": [10, 11],
                    "new_block_ids": [2, 3],
                }
            ],
            new_reqs=[
                {
                    "req_id": 7,
                    "block_ids": [1, 2],
                    "image": {"height": 32, "width": 48, "steps": 1},
                }
            ],
        )
    )
    rt = worker.model_executor.resource_runtime
    assert rt.used("kv_block") == 3

    worker.execute(
        seal_batch(
            2,
            [
                {
                    "req_id": 7,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 2, 3],
                }
            ],
            base_version=1,
        )
    )
    assert rt.used("image_latent") == 6
    assert rt.used("scratch") == 1

    worker.execute(
        seal_batch(
            3,
            [{"req_id": 7, "kind": "commit_gen"}],
            base_version=2,
        )
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0
    assert rt.used("kv_block") == 3

    worker.drop_request(7)
    assert rt.total_active() == 0


def test_model_executor_denoise_scratch_is_one_live_lease_per_cfg_branch():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.execute(
        seal_batch(
            1,
            [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 2, 2],
                    "cfg": {
                        "branch_count": 3,
                        "text_scale": 1.0,
                        "img_scale": 1.0,
                        "renorm_type": "none",
                        "renorm_min": 0.0,
                        "interval": [0.0, 1.0],
                    },
                }
            ],
            new_reqs=[{"req_id": 17, "image": {"height": 32, "width": 32, "steps": 2}}],
        )
    )
    rt = worker.model_executor.resource_runtime
    assert rt.used("image_latent") == 4
    assert rt.used("scratch") == 3

    worker.execute(
        seal_batch(
            2,
            [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 1,
                    "latent_shape": [1, 2, 2],
                    "cfg": {
                        "branch_count": 3,
                        "text_scale": 1.0,
                        "img_scale": 1.0,
                        "renorm_type": "none",
                        "renorm_min": 0.0,
                        "interval": [0.0, 1.0],
                    },
                }
            ],
            base_version=1,
        )
    )
    assert rt.used("image_latent") == 4
    assert rt.used("scratch") == 3

    worker.execute(
        seal_batch(
            3,
            [{"req_id": 17, "kind": "commit_gen"}],
            base_version=2,
        )
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0

    worker.execute(
        seal_batch(
            4,
            [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 3, 2],
                }
            ],
            base_version=3,
        )
    )
    assert rt.used("image_latent") == 6
    assert rt.used("scratch") == 1


def test_model_executor_rejects_kv_blocks_beyond_declared_capacity():
    worker = ModelWorker(StubUniModel(), block_size=256, kv_token_capacity=512)
    with pytest.raises(WorkerError) as exc:
        worker.execute(
            seal_batch(
                1,
                [
                    {
                        "req_id": 9,
                        "kind": "prefill_und",
                        "pos_range": [0, 1],
                        "token_ids": [10],
                    }
                ],
                new_reqs=[{"req_id": 9, "block_ids": [1, 2, 3]}],
            )
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert worker.model_executor.resource_runtime.used("kv_block") == 0
    assert 9 not in worker.model_executor.sessions


def test_model_executor_rolls_back_denoise_latent_if_scratch_admission_fails():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.model_executor.resource_runtime.totals["image_latent"] = 64
    worker.model_executor.resource_runtime.totals["scratch"] = 0

    worker.execute(
        seal_batch(
            1,
            [
                {
                    "req_id": 11,
                    "kind": "prefill_und",
                    "pos_range": [0, 1],
                    "token_ids": [10],
                    "new_block_ids": [1],
                }
            ],
            new_reqs=[{"req_id": 11, "image": {"steps": 1}}],
        )
    )

    worker.model_executor.resource_runtime.totals["scratch"] = 1
    worker.model_executor.resource_runtime.acquire("scratch", req_id=99, units=1)
    with pytest.raises(WorkerError) as exc:
        worker.execute(
            seal_batch(
                2,
                [
                    {
                        "req_id": 11,
                        "kind": "denoise_gen",
                        "timestep_idx": 0,
                        "latent_shape": [8],
                    }
                ],
                base_version=1,
            )
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert worker.model_executor.resource_runtime.used("image_latent") == 0
    assert worker.model_executor.sessions.get(11).residency.image_latent_active is False


def test_model_executor_uses_resource_plan_latent_downsample_for_image_accounting():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    from uniserve_worker.contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan

    worker.model.resource_plan = ResourcePlan(
        kv_block="per_block",
        image_latent=LatentTokens(downsample=32),
        scratch=PerBranch(),
    )
    worker.model_executor.resource_plan = worker.model.resource_plan
    worker.model_executor.resource_runtime.totals["image_latent"] = 4096

    worker.execute(
        seal_batch(
            1,
            [
                {
                    "req_id": 21,
                    "kind": "prefill_und",
                    "pos_range": [0, 1],
                    "token_ids": [10],
                    "new_block_ids": [1],
                }
            ],
            new_reqs=[
                {
                    "req_id": 21,
                    "image": {"height": 1152, "width": 2048, "steps": 2},
                }
            ],
        )
    )

    result = worker.execute(
        seal_batch(
            2,
            [
                {
                    "req_id": 21,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                }
            ],
            base_version=1,
        )
    )

    [row] = result["per_seq"]
    assert row["req_id"] == 21
    assert row["denoise_done"] is False
    assert row["num_steps_done"] == 1
    assert worker.model_executor.resource_runtime.used("image_latent") == 2304
