from __future__ import annotations

import pytest
import torch

from uniserve_worker.runtime.product_store import (
    ProductRecord,
    ProductStore,
    VisionFeatureProduct,
)

pytestmark = pytest.mark.unit


def _vision_record(handle: int, session_id: int) -> ProductRecord:
    return ProductRecord(
        handle=handle,
        session_id=session_id,
        payload=VisionFeatureProduct(
            features=torch.full((2, 3), float(handle)),
            grid=torch.tensor([[1, 2]], dtype=torch.long),
            height=16,
            width=32,
            source_base64=f"image-{handle}",
        ),
        content_hash=handle,
    )


def _commit(store: ProductStore, record: ProductRecord) -> None:
    transaction = store.begin_step({record.session_id})
    transaction.stage(record)
    transaction.prepare()
    transaction.publish()
    transaction.finalize()


def test_encoder_product_lifetime_is_owned_by_explicit_handle_release() -> None:
    store = ProductStore(encoder_cache_budget=1)
    cached = _vision_record(11, 1)
    _commit(store, cached)

    store.drop(1)

    assert store.require(11) == cached

    store.release((11,))
    replacement = _vision_record(12, 2)
    _commit(store, replacement)

    assert store.require(12) == replacement
    assert store.encoder_output_count() == 1
