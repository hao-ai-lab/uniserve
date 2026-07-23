from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from uniserve_worker.batch import (
    Admission,
    ExecutionResult,
    KvLeaseDelta,
    OperationEnvelope,
    OperationResult,
    PublishedProduct,
    SequenceAdmission,
    SequenceDelta,
    SequenceEffect,
    SequenceMode,
    SequenceOperation,
    TokenInput,
    TokenPolicy,
    TokenSource,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.kv_store import KvStore
from uniserve_worker.runtime.latent_store import LatentStore
from uniserve_worker.runtime.product_store import ProductStore
from uniserve_worker.runtime.replay import ReplayRecord, ReplayStore
from uniserve_worker.runtime.request_session import SessionStore
from uniserve_worker.runtime.snapshot_store import SnapshotProvider
from uniserve_worker.runtime.transfer import LocalTransport, Locator, fetch_locator


@dataclass(frozen=True)
class _CommittedPublication:
    provider: SnapshotProvider
    transport: LocalTransport
    result: ExecutionResult
    value: torch.Tensor


def _committed_publication(root: Path, session_id: int = 7) -> _CommittedPublication:
    sessions = SessionStore()
    kv = KvStore()
    latents = LatentStore(capacity_tokens=16)
    products = ProductStore(encoder_cache_budget=4)
    replay = ReplayStore()
    transport = LocalTransport()
    admission = Admission.create(session_id, sequence=SequenceAdmission())
    session = sessions.admit(admission, epoch=3)
    kv.admit(admission)
    operation = OperationEnvelope.create(
        session_id=session_id,
        epoch=3,
        op_id=11,
        base_version=0,
        admission_digest=admission.digest,
        model_spec_digest="1" * 64,
        weight_digest="2" * 64,
        operation=SequenceOperation(
            mode=SequenceMode.DECODE,
            lease=KvLeaseDelta(),
            position=(0, 1),
            policy=TokenPolicy(),
            input=TokenInput(
                token_ids=(3,),
                source=TokenSource.WIRE,
                draft_token_ids=(),
                burst_tokens=1,
                stop_token_ids=(),
                stop_terminal=False,
                return_all_logits=False,
            ),
        ),
    )
    value = torch.tensor([[1.25, -3.5]], dtype=torch.float32)
    locator = transport.publish(value).to_wire_json()
    operation_result = OperationResult.for_operation(
        operation,
        SequenceDelta(
            SequenceEffect(published_logits=PublishedProduct(handle=19, locator=locator))
        ),
    )
    result = ExecutionResult(step_id=23, operations=(operation_result,))
    session.version = operation_result.result_version
    session.last_op_id = operation.op_id
    session.last_digest = operation.digest
    session.last_step_id = result.step_id
    replay.restore_records(
        {session_id},
        (
            ReplayRecord(
                session_id=session_id,
                epoch=operation.epoch,
                op_id=operation.op_id,
                digest=operation.digest,
                step_id=result.step_id,
                result=operation_result,
            ),
        ),
    )
    provider = SnapshotProvider(
        root,
        model_spec_digest="1" * 64,
        weight_digest="2" * 64,
        topology={"tp_size": 1, "tp_rank": 0},
        device="cpu",
        sessions=sessions,
        kv=kv,
        latents=latents,
        products=products,
        replay=replay,
        adapters=None,
        transport=transport,
    )
    return _CommittedPublication(provider, transport, result, value)


def _published_locator(result: ExecutionResult) -> Locator:
    delta = result.operations[0].delta
    assert isinstance(delta, SequenceDelta)
    assert delta.effect.published_logits is not None
    return Locator.from_wire_json(delta.effect.published_logits.locator)


def test_execution_snapshot_keeps_published_tensor_reachable_after_transport_loss(tmp_path):
    committed = _committed_publication(tmp_path)

    durable_result = committed.provider.snapshot_execution({7}, committed.result)
    locator = _published_locator(durable_result)

    assert set(locator.meta["durable_snapshot"]) == {
        "format_version",
        "root",
        "object",
        "tensor",
    }
    restored = fetch_locator(LocalTransport(), locator)
    torch.testing.assert_close(restored, committed.value, rtol=0, atol=0)


def test_durable_locator_rejects_modified_snapshot_bytes(tmp_path):
    committed = _committed_publication(tmp_path)
    durable_result = committed.provider.snapshot_execution({7}, committed.result)
    locator = _published_locator(durable_result)
    descriptor = locator.meta["durable_snapshot"]
    tensor_path = (
        Path(descriptor["root"])
        / "objects"
        / descriptor["object"]
        / "tensors.safetensors"
    )
    with tensor_path.open("r+b") as target:
        target.seek(-1, 2)
        original = target.read(1)
        target.seek(-1, 2)
        target.write(bytes((original[0] ^ 0xFF,)))

    with pytest.raises(WorkerError, match="content verification"):
        fetch_locator(LocalTransport(), locator)


def test_restored_replay_republishes_runtime_assets_without_persisting_locators(tmp_path):
    committed = _committed_publication(tmp_path)
    durable_result = committed.provider.snapshot_execution({7}, committed.result)
    old_locator = _published_locator(durable_result)

    sessions = SessionStore()
    kv = KvStore()
    replay = ReplayStore()
    transport = LocalTransport()
    restored = SnapshotProvider(
        tmp_path,
        model_spec_digest="1" * 64,
        weight_digest="2" * 64,
        topology={"tp_size": 1, "tp_rank": 0},
        device="cpu",
        sessions=sessions,
        kv=kv,
        latents=LatentStore(capacity_tokens=16),
        products=ProductStore(encoder_cache_budget=4),
        replay=replay,
        adapters=None,
        transport=transport,
    )

    references = restored.restore_latest()
    records = replay.snapshot_records({7})

    assert len(references) == 1
    assert sessions.get(7).version == 1
    assert len(records) == 1
    replayed_locator = _published_locator(
        ExecutionResult(step_id=records[0].step_id, operations=(records[0].result,))
    )
    assert replayed_locator.session == transport.session()
    assert "durable_snapshot" not in replayed_locator.meta
    torch.testing.assert_close(fetch_locator(transport, replayed_locator), committed.value)
    torch.testing.assert_close(fetch_locator(LocalTransport(), old_locator), committed.value)
