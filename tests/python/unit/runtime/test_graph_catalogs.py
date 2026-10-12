"""Default capture catalogs cover every batch a worker admits."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve_worker.model_executor.graph_inputs import (
    decode_captures,
    prefill_captures,
)

pytestmark = pytest.mark.unit

# One unit per cache page and a pool far larger than any admitted batch.
POOL = {"row_units": 1, "num_units": 1 << 20}


@pytest.mark.parametrize("max_rows", [1, 5, 31, 100, 128, 129, 300, 1000])
def test_default_decode_sizes_hold_every_admitted_batch(max_rows):
    sizes = decode_captures(cpu_worker_config(), max_rows=max_rows, **POOL)

    assert sizes[-1] == max_rows
    for rows in range(1, max_rows + 1):
        bucket = next(size for size in sizes if size >= rows)
        if rows >= 128:
            # Padding stays within a sixteenth of the bucket.
            assert bucket - rows <= bucket // 16


def test_default_decode_sizes_follow_the_pool():
    config = cpu_worker_config()

    # A pool of 41 units holds 40 single-unit rows beside the sentinel.
    sizes = decode_captures(config, max_rows=128, row_units=1, num_units=41)

    assert sizes[-1] == 40


def test_named_decode_sizes_are_the_stated_catalog():
    config = replace(cpu_worker_config(), decode_graph_batch_sizes=(1, 2, 4))

    assert decode_captures(config, max_rows=128, **POOL) == (1, 2, 4)
    assert decode_captures(config, max_rows=3, **POOL) == (1, 2)


def test_graph_policy_off_captures_nothing():
    config = replace(cpu_worker_config(), graph_policy="off")

    assert decode_captures(config, max_rows=128, **POOL) == ()
    assert (
        prefill_captures(
            config,
            max_rows=128,
            max_tokens=4096,
            image_builder=False,
            feature_injection=False,
            device_causality=True,
        )
        == ()
    )


@pytest.mark.parametrize("max_rows", [1, 7, 31, 40, 128, 500])
def test_prefill_buckets_hold_every_admitted_batch(max_rows):
    max_tokens = 1024
    shapes = prefill_captures(
        cpu_worker_config(),
        max_rows=max_rows,
        max_tokens=max_tokens,
        image_builder=False,
        feature_injection=False,
        device_causality=True,
    )

    # A bucket serves a batch when it has a row beyond the batch's, for the
    # padding sequence, and at least its tokens.
    for rows in {1, max_rows // 2 or 1, max_rows}:
        for tokens in (rows, rows + 1, 3 * rows, max_tokens):
            if tokens > max_tokens:
                continue
            assert any(
                shape.row_bucket > rows and shape.token_bucket >= tokens
                for shape in shapes
            ), (rows, tokens)
