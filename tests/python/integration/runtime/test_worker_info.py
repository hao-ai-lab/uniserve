"""Worker startup information across the Python IPC boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_worker.bootstrap.worker_info_builder import build_worker_info
from uniserve_worker.config import LaneConfig, WorkerConfig
from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
from uniserve_worker.modeling.model import Model
from uniserve_worker.models.stub import StubModel
from uniserve_worker.protocol.batch import COMPUTATIONS, ForwardMode
from uniserve_worker.worker import Worker

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backends", (("local",), ("local", "shm")))
def test_worker_info_reports_schedulable_work_and_bounds(backends: tuple[str, ...]) -> None:
    worker = execution_worker(transfer_backends=backends)
    try:
        info = worker.info.to_mapping()
    finally:
        worker.close()

    assert info["device"] == "cpu"
    assert tuple(info["transfer_backends"]) == backends
    assert set(info["supported_ops"]) == {
        "prefill",
        "decode",
        "verify",
        "vision_encoding",
        "latent_encoding",
        "latent_preparation",
        "denoising",
        "image_decoding",
        "tensor",
        "kv_publish",
        "kv_install",
    }
    assert info["kv_cache"]["num_layers"] > 0
    assert info["kv_cache"]["num_kv_heads"] > 0
    assert info["kv_cache"]["head_dim"] > 0
    assert info["max_batch_ops"] > 0
    assert info["max_unresolved_ops"] > 0
    assert info["request_slots"] > 0
    assert info["model_name"]
    assert len(info["configuration_id"]) == 64


def test_action_model_reports_zero_kv_geometry() -> None:
    class ActionModel(Model):
        architecture = "ActionModel"

        @classmethod
        def components(cls, config):
            return (ComponentSpec("decoder", (CallSpec(Call.DECODE_VIDEO),)),)

        generation = None

    worker_config = WorkerConfig(
        device="cpu",
        rank=0,
        world_size=1,
        block_size=64,
        kv_token_capacity=None,
        attention_backend=None,
        model_dtype="bfloat16",
        kv_cache_dtype=None,
        kv_memory_fraction=0.9,
        max_batch_operations=2,
        max_batch_tokens=2,
        max_request_pool_size=2,
        generation_device=None,
    )

    info = build_worker_info(ActionModel(), worker_config)

    assert info.uses_kv is False
    assert info.kv_cache is None
    assert info.request_slots == 2
    assert info.max_batch_ops == 2


def test_loaded_worker_identity_distinguishes_incarnations_in_one_process() -> None:
    snapshots = []
    for grant in (None, 1 << 40):
        worker = execution_worker(
            worker_id="encoder-0",
            execution=WorkerConfig(
                graph_policy="off",
                prefill_cuda_graph=False,
                flow_graph_batch_sizes=(1,),
                flow_graph_shapes=((16, 16),),
                pool_memory_bytes=grant,
            ),
        )
        try:
            snapshots.append(worker.info.to_mapping())
        finally:
            worker.close()
    first, second = snapshots
    assert first["endpoint"]["worker_id"] == second["endpoint"]["worker_id"] == "encoder-0"
    assert first["endpoint"]["rank"] == second["endpoint"]["rank"] == 0
    assert first["world_size"] == second["world_size"] == 1
    assert first["endpoint"]["node"] == second["endpoint"]["node"]
    assert first["endpoint"]["address_space"] == second["endpoint"]["address_space"]
    assert first["endpoint"]["incarnation"] != second["endpoint"]["incarnation"]
    assert first["configuration_id"] == second["configuration_id"]


@pytest.mark.parametrize("with_lane_limits", (False, True))
def test_worker_info_reports_limits_safe_for_all_bound_lanes(with_lane_limits) -> None:
    config = replace(
        stub_worker_config(16, max_batch_tokens=256),
        max_batch_operations=4,
        lanes=(
            LaneConfig(
                "decode", 64, (ForwardMode.DECODE, ForwardMode.VERIFY), max_batch_operations=2
            ),
            LaneConfig(
                "compute",
                64,
                tuple(
                    kind
                    for kind in COMPUTATIONS
                    if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                ),
                max_batch_tokens=128,
            ),
        )
        if with_lane_limits
        else (),
    )

    # The scheduler receives one shared bound even when lanes constrain different
    # dimensions. Unspecified lane limits inherit the configured model capacity.
    info = build_worker_info(StubModel(), config)

    assert info.max_batch_ops == (2 if with_lane_limits else 4)
    assert info.max_batch_tokens == (128 if with_lane_limits else 256)


def test_worker_identity_and_capabilities_reflect_enabled_operations() -> None:
    identities = []
    config = replace(stub_worker_config(16, max_batch_tokens=256), graph_policy="off")
    for allowed in (
        frozenset({ForwardMode.PREFILL}),
        frozenset({ForwardMode.PREFILL, ForwardMode.DECODE}),
    ):
        with Worker(
            StubModel(),
            worker_config=config,
            sampling_group=None,
            tokenizer=None,
            allowed_work_variants=allowed,
            pipeline_depth=1,
            completion_payload_bytes=65536,
        ) as worker:
            info = worker.info.to_mapping()
            assert set(info["supported_ops"]) == {code.value for code in allowed}
            assert worker.supports_run_kind(ForwardMode.PREFILL)
            assert worker.supports_run_kind(ForwardMode.DECODE) == (ForwardMode.DECODE in allowed)
            identities.append(info["configuration_id"])

    # Same model and geometry, but different executable work: callers must not
    # mistake these workers for the same resolved configuration.
    assert identities[0] != identities[1]
