"""Forward observations remain independent across serialization and delivery."""

import pickle
from types import MappingProxyType

import pytest

from uniserve_worker.protocol.output import BatchOutput, ForwardStats

pytestmark = pytest.mark.unit


def test_forward_statistics_preserve_snapshots_and_serialized_totals():
    counts = {"decode": 2}
    stats = ForwardStats(
        mode_counts=MappingProxyType(counts),
        mode_tokens={"decode": 5},
        component_us={"forward": 13},
        cuda_graph_replays=2,
        cuda_graph_unpadded_tokens=5,
        cuda_graph_padded_tokens=3,
    )
    counts["decode"] = 99

    report = BatchOutput(7, forward_stats=stats)
    for restored in (
        BatchOutput.from_mapping(report.to_mapping()),
        pickle.loads(pickle.dumps(report)),
    ):
        assert restored == report

    # Neither the caller's dictionary nor a serialized record can modify a
    # completed invocation's counters while another consumer reports them.
    serialized = stats.to_mapping()
    serialized["mode_counts"]["decode"] = 4
    assert stats.mode_counts == {"decode": 2}
    with pytest.raises(TypeError):
        stats.mode_counts["decode"] = 4
