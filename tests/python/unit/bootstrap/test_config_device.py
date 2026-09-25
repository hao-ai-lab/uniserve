"""Device and lane configuration for worker launches."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args
from uniserve_worker.protocol.call import ForwardMode, MediaCall

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mesh", ["tower=gen:cuda:1", "tower=text:cuda;gen:cuda:1"]
)
def test_flow_component_device_is_resolved_with_the_rank_device(mesh, tmp_path):
    config = worker_args(
        tmp_path,
        max_batch_tokens=8192,
        device="cuda:0",
        mesh=mesh,
    )
    assert config.execution.device == "cuda:0"
    assert config.execution.generation_device == "cuda:1"


@pytest.mark.parametrize(
    "mesh", ["tower=text:cuda:2;gen:cuda:1", "tower=gen:cuda"]
)
def test_expert_devices_reject_inconsistent_rank_or_repeated_devices(
    mesh, tmp_path
):
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            max_batch_tokens=8192,
            device="cuda:0",
            mesh=mesh,
        )


def test_quantization_config_defaults_to_model_policy(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        max_batch_tokens=8192,
    )

    assert config.model is not None
    assert config.model.quantization_config == {}


def test_engine_batch_capacity_reaches_worker_resources(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        device="cpu",
        max_batch_calls=128,
        max_batch_tokens=16384,
    )

    assert config.execution.max_batch_calls == 128
    assert config.execution.max_batch_tokens == 16384


def test_quantization_config_reaches_model_launch_config(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        max_batch_tokens=8192,
        quantization_config={"mode": "balanced"},
    )

    assert config.model is not None
    assert config.model.quantization_config == {"mode": "balanced"}


def test_component_quantization_config_reaches_model_launch_config(
    tmp_path,
) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        max_batch_tokens=8192,
        quantization_config={
            "mode": "performance",
            "components": {
                "transformer.attention": "nvfp4",
                "transformer.mlp": "fp8",
                "text_encoder": "bf16",
                "video_vae": "bf16",
            },
        },
    )

    assert config.model is not None
    assert config.model.quantization_config["mode"] == "performance"
    assert config.model.quantization_config["components"] == {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "bf16",
    }


def test_execution_lanes_are_typed_and_domain_disjoint(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        max_batch_tokens=8192,
        lane=[
            '{"lane_id":"decode","sm_budget":64,"domains":["decode"]}',
            '{"lane_id":"compute","sm_budget":88,"domains":["prefill","flow"]}',
        ],
    )

    assert tuple(
        (lane.lane_id, lane.sm_budget) for lane in config.execution.lanes
    ) == (
        ("decode", 64),
        ("compute", 88),
    )
    assert config.execution.lanes[0].call_kinds == (
        ForwardMode.DECODE,
        ForwardMode.VERIFY,
    )


@pytest.mark.parametrize(
    "capacity", ['"kv_capacity_tokens":4096', '"latent_capacity_units":8']
)
def test_execution_lanes_reject_a_pool_capacity(capacity, tmp_path) -> None:
    # A lane partitions compute only; its calls share the worker-wide KV and
    # latent pools, so a lane-local capacity is not a lane field.
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            max_batch_tokens=8192,
            lane=[
                '{"lane_id":"decode","sm_budget":64,"domains":["decode"],'
                + capacity
                + "}"
            ],
        )


def test_execution_lanes_reject_duplicate_domain_bindings(tmp_path) -> None:
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            max_batch_tokens=8192,
            lane=[
                '{"lane_id":"a","sm_budget":64,"domains":["decode"]}',
                '{"lane_id":"b","sm_budget":64,"domains":["decode"]}',
            ],
        )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("ar_decode", {ForwardMode.DECODE}),
        (
            "diffusion_decode",
            {MediaCall.VIDEO_DECODING, MediaCall.AUDIO_DECODING},
        ),
        (
            "media_append",
            {MediaCall.VIDEO_ENCODING, MediaCall.AUDIO_ENCODING},
        ),
        (
            "diffusion_finalize",
            {MediaCall.IMAGE_DECODING, MediaCall.MUXING},
        ),
    ],
)
def test_launch_capabilities_select_concrete_computations(
    selector, expected, tmp_path
):
    config = worker_args(
        tmp_path,
        max_batch_tokens=8192,
        supported_calls=selector,
    )
    assert config.supported_calls == expected
