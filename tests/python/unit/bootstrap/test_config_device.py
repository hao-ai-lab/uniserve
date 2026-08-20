"""Device and lane configuration for worker launches."""

from __future__ import annotations

import pytest

from uniserve_worker.batch import Domain
from uniserve_worker.bootstrap.cli import parse_worker_launch

pytestmark = pytest.mark.unit


def test_engine_batch_capacity_reaches_worker_resources() -> None:
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
            "--max-batch-operations",
            "128",
            "--max-batch-tokens",
            "16384",
        ]
    )

    assert config.resources.max_batch_operations == 128
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
            '{"lane_id":"decode","sm_budget":64,"domains":["decode"]}',
            "--lane",
            '{"lane_id":"compute","sm_budget":88,"domains":["prefill","flow"]}',
        ]
    )

    assert tuple((lane.lane_id, lane.sm_budget) for lane in config.execution.lanes) == (
        ("decode", 64),
        ("compute", 88),
    )
    assert config.execution.lanes[0].domains == (Domain.DECODE,)


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
                '{"lane_id":"a","sm_budget":64,"domains":["decode"]}',
                "--lane",
                '{"lane_id":"b","sm_budget":64,"domains":["decode"]}',
            ]
        )
