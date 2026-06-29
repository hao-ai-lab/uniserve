"""Resource-plane conformance (plan §10.5, DoD 6): drop-cleanup + residency.

Drop-cleanup is the worker half of the resource-consistency contract: after a
request is dropped, the driver retains no per-request state. Run against the
GPU-free Stub (the reference) and the worker-side ResourceRuntime.
"""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache
from uniserve_worker.runtime.resources import VALID_CLASSES, ResourceRuntime
from uniserve_worker.server.runner_driver import RunnerDriver
from uniserve_worker.server.stub import StubEngine, StubUniModel

pytestmark = pytest.mark.integration


def _run_request(engine, rid):
    engine.execute({
        "step_id": 1,
        "new_reqs": [{"req_id": rid, "image": {"steps": 2}}],
        "ops": [{"req_id": rid, "kind": "prefill_und", "modality": "und",
                 "pos_range": [0, 3], "token_ids": [1, 2, 3]}],
    })


def test_drop_request_leaves_no_resident_state():
    engine = StubEngine(block_size=256)
    _run_request(engine, 1)
    _run_request(engine, 2)
    # the stub keeps per-request bookkeeping in reqs/emitted/steps
    assert 1 in engine.reqs or 1 in engine.emitted
    engine.drop_request(1)
    for table in (engine.reqs, engine.emitted, engine.steps):
        assert 1 not in table, "dropped request must leave no worker-resident state"
    # request 2 untouched
    assert 2 in engine.emitted


def test_resource_runtime_residency_and_release():
    rt = ResourceRuntime(["kv_block", "image_latent", "scratch"],
                         totals={"kv_block": 1000, "image_latent": 128, "scratch": 16})
    rt.acquire("kv_block", req_id=1, units=256)
    rt.acquire("image_latent", req_id=1, units=64)
    rt.acquire("kv_block", req_id=2, units=128)
    assert rt.used("kv_block") == 384
    assert rt.total_active() == 448
    # commit releases just the image latents
    assert rt.release_class("image_latent", 1) == 64
    assert rt.total_active() == 384
    # finishing request 1 releases the rest
    freed = rt.release_request(1)
    assert freed == 256
    assert rt.used("kv_block") == 128  # only request 2 remains
    rt.release_request(2)
    assert rt.total_active() == 0


def test_resource_runtime_pressure_snapshot():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 1000})
    rt.acquire("kv_block", req_id=1, units=300)
    (kv,) = [p for p in rt.pressure() if p["class"] == "kv_block"]
    assert kv["total"] == 1000 and kv["used"] == 300 and kv["free"] == 700


def test_resource_runtime_enforces_declared_totals():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 4})
    rt.acquire("kv_block", req_id=1, units=3)
    with pytest.raises(WorkerError) as exc:
        rt.acquire("kv_block", req_id=2, units=2)
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert rt.used("kv_block") == 3


def test_resource_runtime_rejects_unknown_class():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 1})
    with pytest.raises(WorkerError) as exc:
        rt.acquire("encoder_output", req_id=1, units=1)
    assert exc.value.code == ErrorCode.CAPABILITY_MISMATCH
    assert set(VALID_CLASSES) >= {"kv_block", "scratch", "image_latent"}


def test_model_runner_default_resource_runtime_enforces_model_totals():
    from uniserve_worker.contracts.model_protocols import ModelHooks
    from uniserve_worker.contracts.resource_plan import ResourcePlan
    from uniserve_worker.execution.runner import ModelRunner

    class TinyBlockModel(ModelHooks):
        resource_plan = ResourcePlan(kv_block="per_block")
        num_blocks = 1
        whole_batch_forward = True

        def forward(self, batch):  # pragma: no cover - admission fails first.
            raise AssertionError("unreachable")

    runner = ModelRunner(TinyBlockModel())
    with pytest.raises(WorkerError) as exc:
        runner.execute(
            {
                "step_id": 1,
                "new_reqs": [{"req_id": 1, "block_ids": [1, 2]}],
                "ops": [{"req_id": 1, "kind": "prefill_und", "token_ids": [1], "pos_range": [0, 1]}],
            }
        )
    assert exc.value.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert runner.resource_runtime.used("kv_block") == 0
    assert 1 not in runner.request_states


def test_model_runner_resource_runtime_tracks_blocks_latents_and_drop():
    driver = RunnerDriver(StubUniModel(), block_size=256)
    driver.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 7, "block_ids": [1, 2], "image": {"steps": 1}}],
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
    rt = driver.runner.resource_runtime
    assert rt.used("kv_block") == 3

    driver.execute(
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

    driver.execute(
        {
            "step_id": 3,
            "new_reqs": [],
            "ops": [{"req_id": 7, "kind": "commit_gen"}],
        }
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0
    assert rt.used("kv_block") == 3

    driver.drop_request(7)
    assert rt.total_active() == 0


def test_model_runner_denoise_scratch_is_one_live_lease_per_cfg_branch():
    driver = RunnerDriver(StubUniModel(), block_size=256)
    driver.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 17, "image": {"steps": 2}}],
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
    rt = driver.runner.resource_runtime
    assert rt.used("image_latent") == 4
    assert rt.used("scratch") == 3

    driver.execute(
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

    driver.execute(
        {
            "step_id": 3,
            "new_reqs": [],
            "ops": [{"req_id": 17, "kind": "commit_gen"}],
        }
    )
    assert rt.used("image_latent") == 0
    assert rt.used("scratch") == 0

    driver.execute(
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
    driver = RunnerDriver(StubUniModel(), block_size=256, kv_token_capacity=512)
    with pytest.raises(WorkerError) as exc:
        driver.execute(
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
    assert driver.runner.resource_runtime.used("kv_block") == 0
    assert 9 not in driver.runner.request_states


def test_model_runner_rolls_back_denoise_latent_if_scratch_admission_fails():
    driver = RunnerDriver(StubUniModel(), block_size=256)
    driver.runner.resource_runtime.totals["image_latent"] = 64
    driver.runner.resource_runtime.totals["scratch"] = 0

    driver.execute(
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

    driver.runner.resource_runtime.totals["scratch"] = 1
    driver.runner.resource_runtime.acquire("scratch", req_id=99, units=1)
    with pytest.raises(WorkerError) as exc:
        driver.execute(
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
    assert driver.runner.resource_runtime.used("image_latent") == 0
    assert driver.runner.request_states.get(11).residency.image_latent_active is False


def test_model_runner_uses_resource_plan_latent_downsample_for_image_accounting():
    driver = RunnerDriver(StubUniModel(), block_size=256)
    from uniserve_worker.contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan

    driver.model.resource_plan = ResourcePlan(
        kv_block="per_block",
        image_latent=LatentTokens(downsample=32),
        scratch=PerBranch(),
    )
    driver.runner.resource_plan = driver.model.resource_plan
    driver.runner.resource_runtime.totals["image_latent"] = 4096

    driver.execute(
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

    result = driver.execute(
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
    assert driver.runner.resource_runtime.used("image_latent") == 2304


def test_paged_kv_pool_rejects_views_without_enough_host_blocks():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
    )
    with pytest.raises(WorkerError) as exc:
        pool.view([0], base_len=5)
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR

    with pytest.raises(WorkerError) as exc:
        pool.view([2], base_len=0)
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_paged_kv_pool_rejects_construction_with_nonpositive_dimensions():
    for bad in (
        {"num_layers": 0},
        {"num_blocks": 0},
        {"block_size": 0},
    ):
        kwargs = {
            "num_layers": 1,
            "num_blocks": 2,
            "block_size": 4,
            "num_kv_heads": 1,
            "head_dim": 2,
            "device": "cpu",
        }
        kwargs.update(bad)
        with pytest.raises(WorkerError) as exc:
            PagedKVPool(**kwargs)
        assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_paged_text_cache_overflows_when_no_allocator_is_wired():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    # Two blocks => capacity of 8 tokens; a transient request for more than that
    # has no allocator to grow into, so it must fail fast instead of writing past
    # the resident blocks.
    cache = PagedTextCache(pool, [0, 1], num_layers=1)
    with pytest.raises(WorkerError) as exc:
        cache.request_cache_for_transient(layer_idx=0, n_tokens=9)
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    # The failed capacity check must not have advanced the logical length.
    assert cache.get_seq_length() == 0


def test_paged_text_cache_grows_blocks_through_allocator_on_overflow():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    free = [1, 2, 3]
    allocated: list[int] = []

    def allocate_blocks(n):
        taken = free[:n]
        del free[:n]
        allocated.extend(taken)
        return taken

    # Starts with a single block (capacity 4) but is allowed to grow on demand.
    cache = PagedTextCache(pool, [0], num_layers=1, allocate_blocks=allocate_blocks)
    cache.request_cache_for_transient(layer_idx=0, n_tokens=6)
    # ceil((6 - 4) / 4) == 1 extra block pulled from the free list.
    assert allocated == [1]
    assert cache.block_ids == [0, 1]
    # The transient request stages tokens without advancing the persistent length.
    assert cache.get_seq_length() == 0


def test_paged_text_cache_rejects_short_allocator_return():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )

    def under_provision(n):  # noqa: ARG001 - intentionally ignores the request
        return []

    # An allocator that hands back fewer blocks than requested would otherwise
    # leave the cache silently under-provisioned; ensure_capacity must surface it.
    cache = PagedTextCache(pool, [0], num_layers=1, allocate_blocks=under_provision)
    with pytest.raises(WorkerError) as exc:
        cache.request_cache_for_transient(layer_idx=0, n_tokens=6)
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert cache.block_ids == [0]


def test_transformers_cache_adapter_stages_transient_tokens_in_paged_tail():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    cache = PagedTextCache(pool, [0, 1], num_layers=1)
    k_prefix = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
    v_prefix = k_prefix + 100
    cache.update(k_prefix, v_prefix, layer_idx=0)
    assert cache.get_seq_length() == 3

    view = cache.request_cache_for_transient(layer_idx=0, n_tokens=2)
    k_tail = torch.full((2, 2, 2), 7.0)
    v_tail = torch.full((2, 2, 2), 9.0)
    view.append(0, k_tail, v_tail)

    assert cache.get_seq_length() == 3
    staged_k, staged_v = pool.read(0, [0, 1], start=3, length=2)
    torch.testing.assert_close(staged_k, k_tail)
    torch.testing.assert_close(staged_v, v_tail)


def test_batched_paged_text_cache_tracks_per_row_lengths_and_appends():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    row0 = PagedTextCache(pool, [0, 1], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [2, 3], num_layers=1, length=5)
    batched = BatchedPagedTextCache([row0, row1])

    view = batched.request_cache_for_transient(layer_idx=0, n_tokens=2)

    assert view.block_table().tolist() == [[0, 1], [2, 3]]
    assert view.cache_seqlens().tolist() == [3, 5]

    k = torch.arange(8, dtype=torch.float32).view(2, 2, 1, 2)
    v = k + 100
    view.append(0, k, v)

    torch.testing.assert_close(pool.k[0, 0, 3], k[0, 0])
    torch.testing.assert_close(pool.k[0, 1, 0], k[0, 1])
    torch.testing.assert_close(pool.v[0, 3, 1], v[1, 0])
    torch.testing.assert_close(pool.v[0, 3, 2], v[1, 1])


def test_sensenova_denoise_cache_stages_persistent_kv_into_scratch_pool():
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    wrapper = SenseNovaU1ForUnifiedGeneration(config={}, device="cpu", block_size=4)
    wrapper.num_layers = 1
    wrapper.block_size = 4
    source_pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    wrapper.scratch_pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    wrapper._scratch_free = [0, 1]
    source = PagedTextCache(source_pool, [0], num_layers=1)
    k = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)
    v = k + 100
    source.update(k, v, layer_idx=0)

    staged = wrapper._denoise_cache(source)

    assert staged.pool is wrapper.scratch_pool
    assert staged.block_ids == [0]
    assert source.block_ids == [0]
    staged_k, staged_v = wrapper.scratch_pool.read(0, staged.block_ids, start=0, length=3)
    torch.testing.assert_close(staged_k, k[0].transpose(0, 1).contiguous())
    torch.testing.assert_close(staged_v, v[0].transpose(0, 1).contiguous())
    assert wrapper._scratch_free == [1]


def test_sensenova_denoise_cache_uses_generation_scratch_pool_when_split(monkeypatch):
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
    from uniserve_worker.nn.decoder import Modality
    from uniserve_worker.nn.mesh import LocalP2PTransport

    wrapper = SenseNovaU1ForUnifiedGeneration(config={}, device="cpu", block_size=4)
    wrapper.num_layers = 1
    wrapper.block_size = 4
    wrapper.gen_device = "cpu"  # gen_device is derived from device; pin it for the split path
    # The denoise snapshot now branches on the tower axis: install a two-coordinate
    # tower (both on cpu here) so _denoise_cache routes through the gen-tower KV
    # residency via reshard_kv_snapshot. The cpu->cpu copy + no-op barriers make
    # this the framework path without a GPU.
    wrapper._tower_coords = {Modality.TEXT: 0, Modality.GEN: 1}
    wrapper._tower_transport = LocalP2PTransport(
        axis="tower",
        devices=(torch.device("cpu"), torch.device("cpu")),
        _coord=0,
    )
    source_pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    wrapper.gen_scratch_pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    wrapper._gen_scratch_free = [0, 1]
    source = PagedTextCache(source_pool, [1], num_layers=1)
    k = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2)
    v = -k
    source.update(k, v, layer_idx=0)

    staged = wrapper._denoise_cache(source)

    assert staged.pool is wrapper.gen_scratch_pool
    assert staged.block_ids == [0]
    staged_k, staged_v = wrapper.gen_scratch_pool.read(0, staged.block_ids, start=0, length=4)
    torch.testing.assert_close(staged_k, k[0].transpose(0, 1).contiguous())
    torch.testing.assert_close(staged_v, v[0].transpose(0, 1).contiguous())
    assert wrapper._gen_scratch_free == [1]
