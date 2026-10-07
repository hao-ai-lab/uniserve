"""Worker startup information across the Python IPC boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_models.stub import Model, image_processor
from uniserve_worker.bootstrap.report import build_worker_layout
from uniserve_worker.config.execution import LaneConfig, WorkerConfig
from uniserve_worker.protocol.call import CALL_KINDS, ForwardMode
from uniserve_worker.worker import Worker

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backends", (("local",), ("local", "shm")))
def test_worker_info_reports_schedulable_work_and_bounds(
    backends: tuple[str, ...],
) -> None:
    worker = execution_worker(transfer_backends=backends)
    try:
        info = worker.info.to_mapping()
    finally:
        worker.close()

    assert info["device"] == "cpu"
    assert tuple(info["transfer_backends"]) == backends
    assert set(info["supported_calls"]) == {
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
    assert info["kv_cache"]["num_units"] > 1
    assert info["kv_cache"]["unit_bytes"] > 0
    assert all(
        group["layer_ids"]
        and group["num_kv_heads"] > 0
        and group["head_dim"] > 0
        for group in info["kv_cache"]["groups"]
    )
    assert info["max_batch_calls"] > 0
    assert info["max_unresolved_calls"] > 0
    assert info["request_slots"] > 0
    assert info["model_name"]


def test_loaded_worker_identity_distinguishes_incarnations_in_one_process() -> (
    None
):
    snapshots = []
    for grant in (None, 1 << 40):
        worker = execution_worker(
            worker_id="encoder-0",
            execution=WorkerConfig(
                graph_policy="off",
                prefill_cuda_graph=False,
                flow_graph_batch_sizes=(1,),
                flow_graph_shapes=((16, 16),),
                pool_storage_bytes=grant,
            ),
        )
        try:
            snapshots.append(worker.info.to_mapping())
        finally:
            worker.close()
    first, second = snapshots
    assert (
        first["endpoint"]["worker_id"]
        == second["endpoint"]["worker_id"]
        == "encoder-0"
    )
    assert first["endpoint"]["rank"] == second["endpoint"]["rank"] == 0
    assert first["world_size"] == second["world_size"] == 1
    assert first["endpoint"]["node"] == second["endpoint"]["node"]
    assert (
        first["endpoint"]["address_space"]
        == second["endpoint"]["address_space"]
    )
    assert first["endpoint"]["incarnation"] != second["endpoint"]["incarnation"]


@pytest.mark.parametrize("with_lane_limits", (False, True))
def test_worker_info_reports_limits_safe_for_all_bound_lanes(
    with_lane_limits,
) -> None:
    config = replace(
        stub_worker_config(16, max_batch_tokens=256),
        max_batch_calls=4,
        lanes=(
            LaneConfig(
                "decode",
                64,
                (ForwardMode.DECODE, ForwardMode.VERIFY),
                max_batch_calls=2,
            ),
            LaneConfig(
                "compute",
                64,
                tuple(
                    kind
                    for kind in CALL_KINDS
                    if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                ),
                max_batch_tokens=128,
            ),
        )
        if with_lane_limits
        else (),
    )

    # The scheduler receives one shared bound even when lanes constrain
    # different dimensions. Unspecified lane limits inherit the configured
    # model capacity.
    info = build_worker_layout(
        Model(), config, image_processor=image_processor()
    ).info

    assert info.max_batch_calls == (2 if with_lane_limits else 4)
    assert info.max_batch_tokens == (128 if with_lane_limits else 256)


@pytest.mark.parametrize("attention_backend", ("torch", "auto"))
def test_worker_capabilities_reflect_enabled_calls(attention_backend) -> None:
    config = replace(
        stub_worker_config(16, max_batch_tokens=256),
        graph_policy="off",
        attention_backend=attention_backend,
    )
    for allowed in (
        frozenset({ForwardMode.PREFILL}),
        frozenset({ForwardMode.PREFILL, ForwardMode.DECODE}),
    ):
        with Worker(
            Model(),
            image_processor=image_processor(),
            worker_config=config,
            sampling_group=None,
            tokenizer=None,
            allowed_calls=allowed,
            queue_depth=1,
            completion_payload_bytes=65536,
        ) as worker:
            info = worker.info.to_mapping()
            assert set(info["supported_calls"]) == {
                code.value for code in allowed
            }
            assert worker.supports_computation(ForwardMode.PREFILL)
            assert worker.supports_computation(ForwardMode.DECODE) == (
                ForwardMode.DECODE in allowed
            )
