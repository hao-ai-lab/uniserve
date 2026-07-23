from dataclasses import replace

import pytest

from uniserve_worker.batch import (
    FlowOperation,
    Guidance,
    KvLeaseDelta,
    OperationEnvelope,
    PublishedKv,
    PublishedProduct,
    SequenceMode,
    SequenceOperation,
    TokenPolicy,
)
from uniserve_worker.foundation.errors import WorkerError


def _envelope(operation):
    return OperationEnvelope.create(
        session_id=7,
        epoch=3,
        op_id=11,
        base_version=5,
        admission_digest="a" * 64,
        model_spec_digest="b" * 64,
        weight_digest="c" * 64,
        operation=operation,
    )


def test_product_runtime_locator_does_not_change_operation_identity():
    operation = SequenceOperation(
        mode=SequenceMode.SAMPLE,
        lease=KvLeaseDelta(),
        position=(9, 10),
        policy=TokenPolicy(),
        input=PublishedProduct(handle=19, locator="runtime-a"),
    )

    first = _envelope(operation)
    second = _envelope(replace(operation, input=replace(operation.input, locator="runtime-b")))

    assert first.digest == second.digest


def test_kv_runtime_locators_do_not_change_operation_identity():
    conditioning = PublishedKv(
        handle=7,
        locators=("runtime-a", "runtime-b"),
        source_version=5,
        kv_tokens=64,
        block_ids=(2,),
        group_id=0,
        position=64,
    )
    operation = FlowOperation(
        latent_handle=31,
        position=64,
        start_step=0,
        step_count=1,
        conditioning_position=64,
        conditioning=conditioning,
        guidance=Guidance(
            branch_count=1,
            text_scale=1.0,
            image_scale=1.0,
            renorm_type="none",
            renorm_min=0.0,
            interval=(0.0, 1.0),
        ),
        image_prompt="",
    )

    first = _envelope(operation)
    second = _envelope(
        replace(
            operation,
            conditioning=replace(conditioning, locators=("runtime-c", "runtime-d")),
        )
    )

    assert first.digest == second.digest


def test_kv_publication_selects_tensor_parallel_rank_locators():
    publication = PublishedKv(
        handle=7,
        locators=("rank-0-key", "rank-0-value", "rank-1-key", "rank-1-value"),
        source_version=5,
        kv_tokens=64,
        block_ids=(2,),
        group_id=0,
        position=64,
    )

    selected = publication.for_tensor_rank(1, 2)

    assert selected.locators == ("rank-1-key", "rank-1-value")


def test_kv_publication_rejects_incomplete_tensor_parallel_locator_groups():
    publication = PublishedKv(
        handle=7,
        locators=("rank-0-key", "rank-0-value", "rank-1-key"),
        source_version=5,
        kv_tokens=64,
        block_ids=(2,),
        group_id=0,
        position=64,
    )

    with pytest.raises(WorkerError, match="do not divide"):
        publication.for_tensor_rank(1, 2)
