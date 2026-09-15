"""Device and lane configuration for worker launches."""

from __future__ import annotations

import pytest

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.protocol.operation import ForwardMode, PipelineStage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mesh", ["tower=gen:cuda:1", "tower=text:cuda;gen:cuda:1"]
)
def test_flow_component_device_is_resolved_with_the_rank_device(mesh):
    config = parse_worker_args(
        [
            "--service-name",
            "component-devices",
            "--ipc-payload-cap",
            "65536",
            "--max-batch-tokens",
            "8192",
            "--model",
            "model",
            "--device",
            "cuda:0",
            "--mesh",
            mesh,
        ]
    )
    assert config.execution.device == "cuda:0"
    assert config.execution.generation_device == "cuda:1"


@pytest.mark.parametrize(
    "mesh", ["tower=text:cuda:2;gen:cuda:1", "tower=gen:cuda"]
)
def test_expert_devices_reject_inconsistent_rank_or_repeated_devices(mesh):
    with pytest.raises(SystemExit):
        parse_worker_args(
            [
                "--service-name",
                "component-devices",
                "--ipc-payload-cap",
                "65536",
                "--max-batch-tokens",
                "8192",
                "--model",
                "model",
                "--device",
                "cuda:0",
                "--mesh",
                mesh,
            ]
        )


@pytest.mark.parametrize(
    "allocator", ("expandable_segments:True", "backend:cudaMallocAsync")
)
def test_cuda_ipc_publication_rejects_nonexportable_allocations(
    monkeypatch, allocator: str
) -> None:
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", allocator)
    with pytest.raises(SystemExit):
        parse_worker_args(
            [
                "--service-name",
                "cuda-publication",
                "--ipc-payload-cap",
                "65536",
                "--model",
                "model",
                "--max-batch-tokens",
                "8192",
                "--device",
                "cuda:0",
                "--transfer-backends",
                "shm,cuda_ipc",
                "--publish-backends",
                "shm,cuda_ipc",
            ]
        )


def test_quantization_config_defaults_to_model_policy() -> None:
    config = parse_worker_args(
        [
            "--service-name",
            "precision-default-test",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--max-batch-tokens",
            "8192",
        ]
    )

    assert config.model is not None
    assert config.model.quantization_config == {}


def test_engine_batch_capacity_reaches_worker_resources() -> None:
    config = parse_worker_args(
        [
            "--service-name",
            "capacity-test",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--device",
            "cpu",
            "--max-batch-operations",
            "128",
            "--max-batch-tokens",
            "16384",
        ]
    )

    assert config.execution.max_batch_operations == 128
    assert config.execution.max_batch_tokens == 16384


def test_quantization_config_reaches_model_launch_config() -> None:
    config = parse_worker_args(
        [
            "--service-name",
            "precision-test",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--max-batch-tokens",
            "8192",
            "--quantization-config",
            '{"mode":"balanced"}',
        ]
    )

    assert config.model is not None
    assert config.model.quantization_config == {"mode": "balanced"}


def test_component_quantization_config_reaches_model_launch_config() -> None:
    config = parse_worker_args(
        [
            "--service-name",
            "precision-test",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--max-batch-tokens",
            "8192",
            "--quantization-config",
            '{"mode":"performance","components":{"transformer.attention":"nvfp4","transformer.mlp":"fp8","text_encoder":"bf16","video_vae":"bf16"}}',
        ]
    )

    assert config.model is not None
    assert config.model.quantization_config["mode"] == "performance"
    assert config.model.quantization_config["components"] == {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "bf16",
    }


def test_execution_lanes_are_typed_and_domain_disjoint() -> None:
    config = parse_worker_args(
        [
            "--service-name",
            "lane-test",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--max-batch-tokens",
            "8192",
            "--lane",
            '{"lane_id":"decode","sm_budget":64,"domains":["decode"]}',
            "--lane",
            '{"lane_id":"compute","sm_budget":88,"domains":["prefill","flow"]}',
        ]
    )

    assert tuple(
        (lane.lane_id, lane.sm_budget) for lane in config.execution.lanes
    ) == (
        ("decode", 64),
        ("compute", 88),
    )
    assert config.execution.lanes[0].computations == (
        ForwardMode.DECODE,
        ForwardMode.VERIFY,
    )


def test_execution_lanes_reject_duplicate_domain_bindings() -> None:
    with pytest.raises(SystemExit):
        parse_worker_args(
            [
                "--service-name",
                "lane-test",
                "--ipc-payload-cap",
                "65536",
                "--model",
                "model",
                "--max-batch-tokens",
                "8192",
                "--lane",
                '{"lane_id":"a","sm_budget":64,"domains":["decode"]}',
                "--lane",
                '{"lane_id":"b","sm_budget":64,"domains":["decode"]}',
            ]
        )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("ar_decode", {ForwardMode.DECODE}),
        (
            "diffusion_decode",
            {PipelineStage.VIDEO_DECODING, PipelineStage.AUDIO_DECODING},
        ),
        (
            "media_append",
            {PipelineStage.VIDEO_ENCODING, PipelineStage.AUDIO_ENCODING},
        ),
        (
            "diffusion_finalize",
            {PipelineStage.IMAGE_DECODING, PipelineStage.MUXING},
        ),
    ],
)
def test_launch_capabilities_select_concrete_computations(selector, expected):
    config = parse_worker_args(
        [
            "--service-name",
            "capabilities",
            "--ipc-payload-cap",
            "65536",
            "--max-batch-tokens",
            "8192",
            "--model",
            "model",
            "--supported-ops",
            selector,
        ]
    )
    assert config.supported_ops == expected
