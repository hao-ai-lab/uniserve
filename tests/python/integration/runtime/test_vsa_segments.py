"""Segment-wise VSA reproduces its dense equations over 128-row tiles.

Dense tiles attend every live tile; every video query tile attends the live
dense tiles and, of each segment, the key tiles of highest pooled score that
the segment keeps; empty tiles neither attend nor are attended; and each row
adds its gate times the softmax over live tiles of their mean values.
"""

import pytest
import torch

from tests.python.fixtures.vsa import (
    SEGMENT_TILE,
    SEGMENT_VALID,
    segment_reference,
    segment_tables,
)
from uniserve.nn.attention import vsa
from uniserve.runtime import ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_segment_attention_matches_its_dense_equations():
    torch.manual_seed(2024)
    # Interleaved projections exercise the strided rows a merged projection
    # produces.
    projected = torch.randn(
        len(SEGMENT_VALID) * SEGMENT_TILE,
        4,
        4,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v, gate = projected.unbind(2)
    module = vsa.SegmentAttention(
        vsa.BlockAttention(128**-0.5, tile_size=SEGMENT_TILE)
    )
    segments = vsa.Segments(SEGMENT_TILE, q.shape[0], *segment_tables(q.device))

    with ExecutionContext(module) as context:
        context.prepare(None)
        with context.activate():
            actual = module(q, k, v, gate, segments)

    expected, attending = segment_reference(q, k, v, gate)
    # The established BF16 attention tolerance of the VSA tests covers the
    # kernel's softmax and MMA rounding and the BF16 compression terms.
    torch.testing.assert_close(
        actual[attending], expected[attending], rtol=2e-2, atol=2e-2
    )
