"""Device normalization for single-GPU (tp=1) worker launches."""

from __future__ import annotations

import pytest

from uniserve_worker.batch import Domain
from uniserve_worker.bootstrap.cli import parse_worker_launch
from uniserve_worker.bootstrap.config import _normalize_device

pytestmark = pytest.mark.unit


def test_unindexed_cuda_is_pinned_to_concrete_index() -> None:
    # The frontend launches tp=1 workers with ``--device cuda`` while tensors
    # materialize on ``cuda:0``; the unindexed form must be pinned so the
    # per-forward device-equality check does not reject every output.
    assert _normalize_device("cuda") == "cuda:0"


def test_indexed_cuda_is_preserved() -> None:
    assert _normalize_device("cuda:0") == "cuda:0"
    assert _normalize_device("cuda:1") == "cuda:1"


def test_cpu_is_preserved() -> None:
    assert _normalize_device("cpu") == "cpu"


def test_engine_batch_token_budget_reaches_worker_resources() -> None:
    config = parse_worker_launch(
        [
            "--service-name",
            "capacity-contract",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--device",
            "cpu",
            "--max-batch-tokens",
            "16384",
        ]
    )

    assert config.resources.max_batch_tokens == 16384


def test_execution_lanes_are_typed_and_domain_disjoint() -> None:
    config = parse_worker_launch(
        [
            "--service-name",
            "lane-contract",
            "--ipc-payload-cap",
            "65536",
            "--model",
            "model",
            "--max-batch-tokens",
            "8192",
            "--lane",
            '{"lane_id":"und","sm_budget":64,"domains":["und"]}',
            "--lane",
            '{"lane_id":"gen","sm_budget":88,"domains":["gen"]}',
        ]
    )

    assert tuple((lane.lane_id, lane.sm_budget) for lane in config.execution.lanes) == (
        ("und", 64),
        ("gen", 88),
    )
    assert config.execution.lanes[0].domains == (Domain.UND,)


def test_execution_lanes_reject_duplicate_domain_bindings() -> None:
    with pytest.raises(SystemExit):
        parse_worker_launch(
            [
                "--service-name",
                "lane-contract",
                "--ipc-payload-cap",
                "65536",
                "--model",
                "model",
                "--max-batch-tokens",
                "8192",
                "--lane",
                '{"lane_id":"a","sm_budget":64,"domains":["und"]}',
                "--lane",
                '{"lane_id":"b","sm_budget":64,"domains":["und"]}',
            ]
        )
