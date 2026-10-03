"""A hybrid model's cache groups, unit geometry and table widths."""

import pytest

from tests.python.fixtures.hybrid import hybrid_model
from uniserve_worker.bootstrap.cache import (
    cache_info,
    group_layers,
    plan_cache,
    resident_width,
    table_widths,
)
from uniserve_worker.bootstrap.capacity import derive_runtime_kv_capacity
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.worker_info import KvGroupKind

pytestmark = pytest.mark.unit

WINDOWED = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
FULL = (5, 11)


def _config(**options):
    return WorkerConfig(
        device="cpu", block_size=16, max_sequence_tokens=256, **options
    )


def test_hybrid_layers_form_one_group_per_window_and_page_shape():
    model = hybrid_model()
    info = cache_info(model, _config(), num_units=20)

    windowed, full = info.groups
    # The windowed rows are the widest, so their pages hold ``block_size``
    # tokens; a full-attention page holds twice as many in one unit. Twelve
    # layers give two columns: five units per windowed page.
    assert windowed.kind is KvGroupKind.SLIDING_WINDOW
    assert (windowed.window, windowed.sink) == (8, 0)
    assert (windowed.page_tokens, windowed.units_per_page) == (16, 5)
    assert windowed.layer_ids == WINDOWED
    assert (windowed.num_kv_heads, windowed.head_dim) == (4, 4)
    assert full.kind is KvGroupKind.FULL
    assert (full.page_tokens, full.units_per_page) == (32, 1)
    assert full.layer_ids == FULL
    assert (full.num_kv_heads, full.head_dim) == (1, 8)
    assert (info.num_units, info.dtype) == (20, "float32")
    # Two columns of key and value planes, each 16 windowed rows of 16 FP32
    # elements with one initialization flag.
    assert info.unit_bytes == 2 * 2 * (16 * 16 * 4 + 1)

    planes = plan_cache(model, _config())
    assert group_layers(model, planes) == (WINDOWED, FULL)


def test_hybrid_tables_bound_windowed_widths_by_history_and_queries():
    planes = plan_cache(hybrid_model(), _config())
    # A windowed table stages the pages eight history tokens and sixteen
    # queries intersect, at most ceil(24 / 16) + 1; a full table spans the
    # longest sequence. The group's five unit positions are five tables.
    assert table_widths(
        planes, max_sequence_tokens=256, max_query_tokens=16
    ) == (3, 3, 3, 3, 3, 8)
    # A slot's installed windowed table may still hold every page.
    assert resident_width(planes, max_sequence_tokens=256) == 16


def test_hybrid_capacity_counts_units_of_every_group():
    planes = plan_cache(hybrid_model(), _config())
    capacity = derive_runtime_kv_capacity(
        pages=tuple(
            (group.page_tokens, group.units_per_page) for group in planes.groups
        ),
        kv_token_capacity=256,
        unit_bytes=planes.unit_bytes,
    )
    # 16 windowed pages of five units and 8 full pages of one unit.
    assert capacity.num_units == 16 * 5 + 8
