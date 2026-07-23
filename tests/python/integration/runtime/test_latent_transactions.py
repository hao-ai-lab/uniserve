"""Latent transaction conformance for executor-driven denoise steps.

A failed denoise step must leave every touched request's committed flow latent and schedule cursor unchanged, whether the failure lands in a sibling operation of the same step or midway through a multi-step denoise burst, and retrying the same operations must reproduce the uninterrupted trajectory bit for bit. The success path commits the accepted update tensor itself into the system latent store without an extra copy.

The initial latent noise is counter-based: it is a pure function of the request's session seed and the denoise operation's stable ``op_id`` (see ``flow_noise_seed``), so it is invariant to batch position and reproduces exactly on retry with no generator-state snapshot in the step transaction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.forward_context import get_forward_context
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution import ExecutorConfig, ModelExecutor
from uniserve_worker.execution.flow import GenState, GuidePlan, PreparedFlowStep
from uniserve_worker.nn.diffusion import FlowMatchSchedule, ScheduleDirection
from uniserve_worker.nn.diffusion.noise import init_latent
from uniserve_worker.runtime.request_state import flow_noise_seed
from uniserve_worker.runtime.residency import KvCacheSpec, ResidencyManager
from uniserve_worker.runtime.resources import ResourceRuntime

pytestmark = pytest.mark.integration

_STEPS = 4
_LATENT_SHAPE = (4, 2)


class HandleLatentFlowDriver:
    """Executor-held flow driver whose generation state leases its latent from the LatentStore.

    The velocity is a deterministic function of the committed latent, so the final latent observably depends on every accepted step; per-(request, step) failure sets inject faults after a sibling's update was accepted and midway through a burst.
    """

    def __init__(self, residency: ResidencyManager) -> None:
        self.residency = residency
        self.fail_predict: set[tuple[int, int]] = set()
        self.fail_accept: set[tuple[int, int]] = set()
        self.accepted: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Initial noise recorded per op_id, so a test can compare the noise a
        # given operation produced across batch positions and retries.
        self.initial_noise: dict[int, torch.Tensor] = {}

    def prepare_flow_step(self, req_id: int, state: Any, op: Any) -> PreparedFlowStep:
        req_id = int(req_id)
        latent_view = get_forward_context().latent_view
        assert latent_view is not None
        generation_state = latent_view.state(req_id)
        step_index = int(op.get("timestep_idx", state.schedule_cursor) or 0)
        if step_index == 0 and generation_state is None:
            generation_state = GenState(
                latent_pool=self.residency.latent,
                latent_handle=req_id,
                vae_pos_ids=torch.zeros(_LATENT_SHAPE[0], dtype=torch.long),
                num_vae=_LATENT_SHAPE[0],
                H=8,
                W=8,
                schedule=FlowMatchSchedule(
                    num_steps=_STEPS, shift=1.0, direction=ScheduleDirection.DESCENDING
                ),
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
                cfg_renorm_type="global",
                cfg_renorm_min=0.0,
                cfg_interval=(0.0, 1.0),
                cond_pos=0,
            )
            # Counter-based initial noise: seeded from the session seed and the
            # operation's stable op_id, exactly as the production flow drivers.
            op_id = int(op.get("op_id") or 0)
            noise = init_latent(
                _LATENT_SHAPE,
                rng=torch.Generator(device="cpu").manual_seed(
                    flow_noise_seed(int(getattr(state, "seed", 0) or 0), op_id)
                ),
                device="cpu",
                dtype=torch.float32,
            )
            self.initial_noise[op_id] = noise.clone()
            generation_state.x_t = noise
            latent_view.set_state(req_id, generation_state)
        assert generation_state is not None
        t, t_next = generation_state.schedule.pair(step_index, device="cpu")
        return PreparedFlowStep(
            req_id=req_id,
            state=state,
            op=op,
            latent=generation_state.x_t,
            t=t,
            t_next=t_next,
            step_index=step_index,
            total_steps=_STEPS,
            guide=GuidePlan.resolve(
                recipe="additive_deltas",
                text_scale=1.0,
                img_scale=1.0,
                interval=(0.0, 1.0),
                renorm="global",
                renorm_min=0.0,
                t=t,
                branch_count=1,
            ),
            extra={"gs": generation_state},
        )

    def predict_flow_velocity_batch(self, steps: Any, branches_by_step: Any, **_: Any) -> None:
        return None

    def predict_flow_velocity(self, step: PreparedFlowStep, branch: Any) -> torch.Tensor:
        if (int(step.req_id), int(step.step_index)) in self.fail_predict:
            raise RuntimeError("injected failure in velocity prediction")
        return step.latent * 0.5

    def apply_flow_update(self, step: PreparedFlowStep, latent: torch.Tensor) -> None:
        if (int(step.req_id), int(step.step_index)) in self.fail_accept:
            raise RuntimeError("injected failure before this row's update was accepted")
        generation_state = step.extra["gs"]
        generation_state.x_t = latent.to(
            dtype=generation_state.x_t.dtype, device=generation_state.x_t.device
        )
        self.accepted.append((latent, generation_state.x_t))


class HandleLatentDenoiseCPUModel(UniModel):
    """Denoise model exposing only a declared flow-driver surface to the executor."""

    resource_classes = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("denoise_gen",)
    adapter_mode = "none"
    device = "cpu"
    num_layers = 1
    num_blocks = 8
    block_size = 16
    eos_id = 2
    img_start_id = 3

    def __init__(self, residency: ResidencyManager, flow_driver: HandleLatentFlowDriver) -> None:
        self.residency = residency
        self.segment_executor = SimpleNamespace(
            release_staging=lambda cache: None,
            flow_execution=flow_driver,
        )


def _fresh() -> tuple[HandleLatentFlowDriver, ModelExecutor]:
    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": 8})
    residency = ResidencyManager.build(
        KvCacheSpec(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32),
        num_blocks=8,
        block_size=16,
        device="cpu",
        ledger=ledger,
    )
    driver = HandleLatentFlowDriver(residency)
    model = HandleLatentDenoiseCPUModel(residency, driver)
    executor = ModelExecutor(
        model,
        config=ExecutorConfig(simulation=True),
        resource_runtime=ledger,
        residency=residency,
    )
    assert executor.latent_store is not None
    return driver, executor


def _denoise_batch(step_id: int, req_ids: list[int], **kwargs: Any) -> dict[str, Any]:
    return seal_batch(
        step_id,
        [{"req_id": req_id, "kind": "denoise_gen"} for req_id in req_ids],
        **kwargs,
    )


def _burst_batch(
    step_id: int, req_id: int, *, step_count: int, **kwargs: Any
) -> dict[str, Any]:
    return seal_batch(
        step_id,
        [{"req_id": req_id, "kind": "denoise_gen", "denoise_step_count": step_count}],
        **kwargs,
    )


def _new_reqs(req_ids: list[int], *, seed: int = 0) -> list[dict[str, Any]]:
    return [{"req_id": req_id, "block_ids": [], "seed": seed} for req_id in req_ids]


def _denoise_batch_opids(
    step_id: int, specs: list[tuple[int, int]], **kwargs: Any
) -> dict[str, Any]:
    """A denoise batch whose rows carry explicit, scheduler-stable op ids.

    ``specs`` pairs each request id with the op id the scheduler assigns its
    denoise operation; pinning the op id lets a test hold an operation's
    identity fixed while its batch position, siblings, or step id vary.
    """
    return seal_batch(
        step_id,
        [{"req_id": req_id, "kind": "denoise_gen", "op_id": op_id} for req_id, op_id in specs],
        **kwargs,
    )


def _seeded_new_reqs(pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [{"req_id": req_id, "block_ids": [], "seed": seed} for req_id, seed in pairs]


def test_failed_sibling_denoise_step_restores_latent_and_schedule_cursor():
    driver, executor = _fresh()
    _control_driver, control = _fresh()

    executor.execute(_denoise_batch(1, [5, 6], new_reqs=_new_reqs([5, 6])))
    control.execute(_denoise_batch(1, [5, 6], new_reqs=_new_reqs([5, 6])))
    committed_5 = executor.latent_store.get(5)
    committed_6 = executor.latent_store.get(6)
    state_5 = executor.latent_store.state(5)
    snapshot_5 = committed_5.clone()
    snapshot_6 = committed_6.clone()

    # Request 5's second update is accepted before request 6's acceptance fails,
    # so the rollback must undo an already-applied sibling latent update.
    driver.fail_accept.add((6, 1))
    with pytest.raises(RuntimeError, match="injected failure"):
        executor.execute(_denoise_batch(2, [5, 6], base_version=1))
    driver.fail_accept.clear()

    assert executor.latent_store.get(5) is committed_5
    assert executor.latent_store.get(6) is committed_6
    assert executor.latent_store.state(5) is state_5
    assert torch.equal(committed_5, snapshot_5)
    assert torch.equal(committed_6, snapshot_6)
    assert executor.sessions.get(5).schedule_cursor == 1
    assert executor.sessions.get(6).schedule_cursor == 1
    assert executor.sessions.get(5).version == 1
    assert executor.sessions.get(6).version == 1

    # The retried step and the remaining schedule reproduce the uninterrupted
    # trajectory bit for bit.
    for step_id in (2, 3, 4):
        retried = executor.execute(_denoise_batch(step_id, [5, 6], base_version=step_id - 1))
        expected = control.execute(_denoise_batch(step_id, [5, 6], base_version=step_id - 1))
        assert [row["num_steps_done"] for row in retried["per_seq"]] == [
            row["num_steps_done"] for row in expected["per_seq"]
        ]
    for req_id in (5, 6):
        assert torch.equal(executor.latent_store.get(req_id), control.latent_store.get(req_id))


def test_mid_burst_failure_restores_pre_burst_latent_and_retry_is_bit_identical():
    driver, executor = _fresh()
    _control_driver, control = _fresh()

    executor.execute(_denoise_batch(1, [7], new_reqs=_new_reqs([7])))
    control.execute(_denoise_batch(1, [7], new_reqs=_new_reqs([7])))
    committed = executor.latent_store.get(7)
    snapshot = committed.clone()

    # The burst covers the remaining three schedule steps and fails on its
    # second one, after the first in-burst update was already accepted.
    driver.fail_predict.add((7, 2))
    with pytest.raises(RuntimeError, match="injected failure"):
        executor.execute(_burst_batch(2, 7, step_count=3, base_version=1))
    driver.fail_predict.clear()

    assert executor.latent_store.get(7) is committed
    assert torch.equal(committed, snapshot)
    assert executor.sessions.get(7).schedule_cursor == 1
    assert executor.sessions.get(7).version == 1

    retried = executor.execute(_burst_batch(3, 7, step_count=3, base_version=1))
    expected = control.execute(_burst_batch(2, 7, step_count=3, base_version=1))
    assert retried["per_seq"][0]["denoise_done"] is True
    assert retried["per_seq"][0]["num_steps_done"] == expected["per_seq"][0]["num_steps_done"]
    assert torch.equal(executor.latent_store.get(7), control.latent_store.get(7))


def test_denoise_success_path_commits_the_accepted_tensor_without_copies():
    driver, executor = _fresh()

    executor.execute(_denoise_batch(1, [9], new_reqs=_new_reqs([9])))
    first = executor.latent_store.get(9)
    accepted, stored = driver.accepted[-1]
    assert stored is accepted
    assert first is stored
    first_snapshot = first.clone()

    executor.execute(_denoise_batch(2, [9], base_version=1))
    second = executor.latent_store.get(9)
    accepted, stored = driver.accepted[-1]
    assert stored is accepted
    assert second is stored
    # Each accepted step replaces the buffer entry with a fresh tensor and
    # leaves the prior step's tensor storage untouched.
    assert second is not first
    assert torch.equal(first, first_snapshot)


def test_failed_noise_init_step_retries_to_identical_noise_without_rng_snapshot():
    driver, executor = _fresh()
    control_driver, control = _fresh()
    # The scheduler-stable op id of this generation's first denoise step; it is
    # the sole coordinate (with the session seed) of the initial noise.
    op_id = (7 << 40) + 3

    control.execute(_denoise_batch_opids(1, [(8, op_id)], new_reqs=_new_reqs([8], seed=321)))
    clean_noise = control_driver.initial_noise[op_id].clone()

    # The first denoise step samples its initial noise, then fails and rolls the
    # freshly admitted request all the way back. The step transaction keeps no
    # generator-state snapshot.
    driver.fail_predict.add((8, 0))
    with pytest.raises(RuntimeError, match="injected failure"):
        executor.execute(_denoise_batch_opids(1, [(8, op_id)], new_reqs=_new_reqs([8], seed=321)))
    failed_attempt_noise = driver.initial_noise[op_id].clone()
    driver.fail_predict.clear()
    assert 8 not in executor.sessions

    # Re-admitting and retrying the same operation reproduces the noise the
    # failed attempt sampled — bit for bit — and matches an uninterrupted run,
    # purely from the counter coordinates.
    executor.execute(_denoise_batch_opids(2, [(8, op_id)], new_reqs=_new_reqs([8], seed=321)))
    assert torch.equal(driver.initial_noise[op_id], failed_attempt_noise)
    assert torch.equal(driver.initial_noise[op_id], clean_noise)


def test_initial_noise_is_invariant_to_batch_position_and_across_runs():
    op_a = 0xA1
    solo_driver, solo = _fresh()
    solo.execute(_denoise_batch_opids(1, [(3, op_a)], new_reqs=_new_reqs([3], seed=99)))

    # The same operation (req 3, op id ``op_a``, seed 99) now runs as the second
    # row of a two-request batch, behind a sibling with a different op id.
    batched_driver, batched = _fresh()
    batched.execute(
        _denoise_batch_opids(
            1,
            [(4, 0xB2), (3, op_a)],
            new_reqs=_seeded_new_reqs([(4, 7), (3, 99)]),
        )
    )

    # Batch position, batch composition, and a separate executor run leave the
    # operation's initial noise unchanged.
    assert torch.equal(solo_driver.initial_noise[op_a], batched_driver.initial_noise[op_a])
    # A distinct op id in the same batch draws independently.
    assert not torch.equal(batched_driver.initial_noise[op_a], batched_driver.initial_noise[0xB2])


def test_distinct_ops_in_one_session_seed_get_distinct_initial_noise():
    driver, executor = _fresh()
    # The SenseNova travel workload's four images share one session seed and are
    # separated only by their distinct op ids; the four initial noises differ.
    op_ids = [0x100, 0x200, 0x300, 0x400]
    executor.execute(
        _denoise_batch_opids(
            1,
            [(60 + index, op_id) for index, op_id in enumerate(op_ids)],
            new_reqs=_seeded_new_reqs([(60 + index, 4242) for index in range(len(op_ids))]),
        )
    )

    noises = [driver.initial_noise[op_id] for op_id in op_ids]
    for i in range(len(noises)):
        for j in range(i + 1, len(noises)):
            assert not torch.equal(noises[i], noises[j])
