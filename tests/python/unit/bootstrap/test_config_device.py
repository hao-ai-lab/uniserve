"""Device and lane configuration for worker launches."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args
from uniserve_worker.protocol.call import CALL_KINDS, ForwardMode, MediaCall

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


def test_storage_fraction_may_grant_the_whole_device(tmp_path) -> None:
    # The fraction is a share of each device's total storage that the grant
    # also bounds by the storage free at startup, so one is a valid share.
    config = worker_args(tmp_path, kv_memory_fraction=1.0)

    assert config.execution.kv_storage_fraction == 1.0


@pytest.mark.parametrize(
    "fraction", [0.0, -0.5, 1.5, float("nan"), float("inf")]
)
def test_storage_fraction_outside_the_unit_interval_is_refused(
    fraction, tmp_path
) -> None:
    with pytest.raises(SystemExit):
        worker_args(tmp_path, kv_memory_fraction=fraction)


@pytest.mark.parametrize(
    ("device", "mesh", "option"),
    [
        ("gpu0", None, "--device"),
        ("cuda:x", None, "--device"),
        ("cuda:0", "tower=text:cpu:x;gen:cuda:1", "--mesh tower text"),
        ("cuda:0", "tower=gen:bogus", "--mesh tower gen"),
    ],
)
def test_a_malformed_device_string_is_a_usage_error(
    device, mesh, option, tmp_path, capsys
):
    # argparse reports a usage error with exit status 2.
    with pytest.raises(SystemExit) as exit_info:
        worker_args(tmp_path, device=device, mesh=mesh)

    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err


def test_quantization_config_defaults_to_model_policy(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        max_batch_tokens=8192,
    )

    assert config.model is not None
    assert config.model.quantization_config == {}


def test_engine_capacities_reach_worker_resources(tmp_path) -> None:
    config = worker_args(
        tmp_path,
        ipc_payload_cap=65536,
        model="model",
        device="cpu",
        max_batch_calls=128,
        max_batch_tokens=16384,
        max_request_pool_size=257,
    )

    assert config.execution.max_batch_calls == 128
    assert config.execution.max_batch_tokens == 16384
    assert config.execution.max_request_pool_size == 257


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
    # The decode domain holds the calls that advance admitted requests
    # without growing their prompt.
    assert config.execution.lanes[0].call_kinds == (
        ForwardMode.DECODE,
        ForwardMode.VERIFY,
        ForwardMode.TOKEN_DENOISING,
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


def test_a_launch_without_a_selector_serves_every_call_kind(tmp_path):
    # The launching side narrows capabilities only when it has a reason to,
    # so a worker it does not narrow accepts every call kind it implements.
    config = worker_args(tmp_path, max_batch_tokens=8192)
    assert config.supported_calls == frozenset(CALL_KINDS)
