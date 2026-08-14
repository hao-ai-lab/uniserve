from __future__ import annotations

import pytest

from uniserve_worker.batch import Domain
from uniserve_worker.foundation.runtime_config import LaneConfig
from uniserve_worker.worker.model import _has_decode_flow_partition

pytestmark = pytest.mark.unit


def test_mixed_graphs_require_decode_and_flow_on_one_physical_partition() -> None:
    split = (
        LaneConfig("decode", 64, (Domain.DECODE,)),
        LaneConfig("compute", 88, (Domain.PREFILL, Domain.FLOW)),
    )
    colocated = (
        LaneConfig("mixed", 88, (Domain.DECODE, Domain.FLOW)),
        LaneConfig("prefill", 64, (Domain.PREFILL,)),
    )

    assert _has_decode_flow_partition(())
    assert not _has_decode_flow_partition(split)
    assert _has_decode_flow_partition(colocated)
