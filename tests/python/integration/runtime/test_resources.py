"""Resource-plane conformance (plan §10.5, DoD 6): drop-cleanup + residency.

Drop-cleanup is the worker half of the resource-consistency contract: after a
request is dropped, the worker retains no per-request state. Run against the
GPU-free Stub (the reference) and the worker-side ResourceRuntime.
"""

from __future__ import annotations

import pytest

from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.server.stub import StubUniModel, StubWorker
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.integration


def _run_request(engine, rid):
    engine.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": rid, "image": {"steps": 2}}],
            "ops": [
                {
                    "req_id": rid,
                    "kind": "prefill_und",
                    "modality": "und",
                    "pos_range": [0, 3],
                    "token_ids": [1, 2, 3],
                }
            ],
        }
    )


def test_drop_request_leaves_no_resident_state():
    engine = StubWorker(block_size=256)
    _run_request(engine, 1)
    _run_request(engine, 2)
    # the stub keeps per-request bookkeeping in reqs/emitted/steps
    assert 1 in engine.reqs or 1 in engine.emitted
    engine.drop_request(1)
    for table in (engine.reqs, engine.emitted, engine.steps):
        assert 1 not in table, "dropped request must leave no worker-resident state"
    # request 2 untouched
    assert 2 in engine.emitted


def test_model_runner_default_resource_runtime_enforces_model_totals():
    from uniserve_worker.contracts.model_protocols import ModelHooks
    from uniserve_worker.contracts.resource_plan import ResourcePlan
    from uniserve_worker.execution.runner import ModelRunner, RunnerConfig

    class TinyBlockModel(ModelHooks):
        resource_plan = ResourcePlan(kv_block="per_block")
        num_blocks = 1
        whole_batch_forward = True

        def forward(self, batch):  # pragma: no cover - admission fails first.
            raise AssertionError("unreachable")

    runner = ModelRunner(TinyBlockModel(), config=RunnerConfig(simulation=True))
    with pytest.raises(WorkerError) as exc:
        runner.execute(
            {
                "step_id": 1,
                "new_reqs": [{"req_id": 1, "block_ids": [1, 2]}],
                "ops": [
                    {"req_id": 1, "kind": "prefill_und", "token_ids": [1], "pos_range": [0, 1]}
                ],
            }
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert runner.resource_runtime.used("kv_block") == 0
    assert 1 not in runner.request_states


def test_model_runner_resource_runtime_tracks_blocks_latents_and_drop():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.execute(
        {
            "step_id": 1,
            "new_reqs": [
                {"req_id": 7, "block_ids": [1, 2], "image": {"height": 32, "width": 48, "steps": 1}}
            ],
            "ops": [
                {
                    "req_id": 7,
                    "kind": "prefill_und",
                    "pos_range": [0, 2],
                    "token_ids": [10, 11],
                    "new_block_ids": [2, 3],
                }
            ],
        }
    )
    rt = worker.model_runner.resource_runtime
    assert rt.used("kv_block") == 3

    worker.execute(
        {
            "step_id": 2,
            "new_reqs": [],
            "ops": [
                {
                    "req_id": 7,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 2, 3],
                }
            ],
        }
    )
    assert rt.used("image_latent") == 6
    assert rt.used("scratch") == 1

    worker.execute(
        {
            "step_id": 3,
            "new_reqs": [],
            "ops": [{"req_id": 7, "kind": "commit_gen"}],
        }
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0
    assert rt.used("kv_block") == 3

    worker.drop_request(7)
    assert rt.total_active() == 0


def test_model_runner_denoise_scratch_is_one_live_lease_per_cfg_branch():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 17, "image": {"height": 32, "width": 32, "steps": 2}}],
            "ops": [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 2, 2],
                    "cfg": {"branch_count": 3},
                }
            ],
        }
    )
    rt = worker.model_runner.resource_runtime
    assert rt.used("image_latent") == 4
    assert rt.used("scratch") == 3

    worker.execute(
        {
            "step_id": 2,
            "new_reqs": [],
            "ops": [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 1,
                    "latent_shape": [1, 2, 2],
                    "cfg": {"branch_count": 3},
                }
            ],
        }
    )
    assert rt.used("image_latent") == 4
    assert rt.used("scratch") == 3

    worker.execute(
        {
            "step_id": 3,
            "new_reqs": [],
            "ops": [{"req_id": 17, "kind": "commit_gen"}],
        }
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0

    worker.execute(
        {
            "step_id": 4,
            "new_reqs": [],
            "ops": [
                {
                    "req_id": 17,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                    "latent_shape": [1, 3, 2],
                }
            ],
        }
    )
    assert rt.used("image_latent") == 6
    assert rt.used("scratch") == 1


def test_model_runner_rejects_kv_blocks_beyond_declared_capacity():
    worker = ModelWorker(StubUniModel(), block_size=256, kv_token_capacity=512)
    with pytest.raises(WorkerError) as exc:
        worker.execute(
            {
                "step_id": 1,
                "new_reqs": [{"req_id": 9, "block_ids": [1, 2, 3]}],
                "ops": [
                    {
                        "req_id": 9,
                        "kind": "prefill_und",
                        "pos_range": [0, 1],
                        "token_ids": [10],
                    }
                ],
            }
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert worker.model_runner.resource_runtime.used("kv_block") == 0
    assert 9 not in worker.model_runner.request_states


def test_model_runner_rolls_back_denoise_latent_if_scratch_admission_fails():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    worker.model_runner.resource_runtime.totals["image_latent"] = 64
    worker.model_runner.resource_runtime.totals["scratch"] = 0

    worker.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 11, "image": {"steps": 1}}],
            "ops": [
                {
                    "req_id": 11,
                    "kind": "prefill_und",
                    "pos_range": [0, 1],
                    "token_ids": [10],
                    "new_block_ids": [1],
                }
            ],
        }
    )

    worker.model_runner.resource_runtime.totals["scratch"] = 1
    worker.model_runner.resource_runtime.acquire("scratch", req_id=99, units=1)
    with pytest.raises(WorkerError) as exc:
        worker.execute(
            {
                "step_id": 2,
                "new_reqs": [],
                "ops": [
                    {
                        "req_id": 11,
                        "kind": "denoise_gen",
                        "timestep_idx": 0,
                        "latent_shape": [8],
                    }
                ],
            }
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert worker.model_runner.resource_runtime.used("image_latent") == 0
    assert worker.model_runner.request_states.get(11).residency.image_latent_active is False


def test_model_runner_uses_resource_plan_latent_downsample_for_image_accounting():
    worker = ModelWorker(StubUniModel(), block_size=256, simulation=True)
    from uniserve_worker.contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan

    worker.model.resource_plan = ResourcePlan(
        kv_block="per_block",
        image_latent=LatentTokens(downsample=32),
        scratch=PerBranch(),
    )
    worker.model_runner.resource_plan = worker.model.resource_plan
    worker.model_runner.resource_runtime.totals["image_latent"] = 4096

    worker.execute(
        {
            "step_id": 1,
            "new_reqs": [
                {
                    "req_id": 21,
                    "image": {"height": 1152, "width": 2048, "steps": 2},
                }
            ],
            "ops": [
                {
                    "req_id": 21,
                    "kind": "prefill_und",
                    "pos_range": [0, 1],
                    "token_ids": [10],
                    "new_block_ids": [1],
                }
            ],
        }
    )

    result = worker.execute(
        {
            "step_id": 2,
            "new_reqs": [],
            "ops": [
                {
                    "req_id": 21,
                    "kind": "denoise_gen",
                    "timestep_idx": 0,
                }
            ],
        }
    )

    assert result["per_seq"] == [{"req_id": 21, "denoise_done": False, "num_steps_done": 1}]
    assert worker.model_runner.resource_runtime.used("image_latent") == 2304
