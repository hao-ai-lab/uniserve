"""Shared transaction and replay boundary for worker operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..contracts.batches import Batch
from ..contracts.operation import OperationEnvelope
from ..contracts.outputs import FinalizableSeqResult
from ..foundation.errors import invalid_descriptor
from ..foundation.wire import wire_int
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore, TransactionalStore
from ..runtime.resources import ResourceRuntime

__all__ = ["OperationExecutor"]


class _VersionedSeqResult:
    """Attach operation identity to a result finalized at response time."""

    def __init__(self, source: FinalizableSeqResult, operation: OperationEnvelope) -> None:
        self.source = source
        self.req_id = operation.session_id
        self.op_id = operation.op_id
        self.epoch = operation.epoch
        self.base_version = operation.base_version
        self.result_version = operation.base_version + 1

    def finalize(self) -> dict[str, Any]:
        result = dict(self.source.finalize())
        _stamp_result(result, self)
        return result

    def ready(self) -> bool:
        ready = getattr(self.source, "ready", None)
        return bool(ready()) if callable(ready) else True

    def cuda_ready_group_key(self) -> int:
        group_key = getattr(self.source, "cuda_ready_group_key", None)
        return int(group_key()) if callable(group_key) else id(self.source)

    def cuda_ready_elapsed_us(self) -> int | None:
        elapsed = getattr(self.source, "cuda_ready_elapsed_us", None)
        return elapsed() if callable(elapsed) else None

    def __deepcopy__(self, memo: dict[int, Any]) -> "_VersionedSeqResult":
        del memo
        return self


def _stamp_result(result: dict[str, Any], operation: Any) -> None:
    result["op_id"] = int(operation.op_id)
    result["epoch"] = int(operation.epoch)
    result["base_version"] = int(operation.base_version)
    result["result_version"] = int(operation.base_version) + 1


class OperationExecutor:
    """Validate, transact, execute, and replay one typed operation batch."""

    def __init__(
        self,
        sessions: SessionStore,
        effect: Callable[..., dict[str, Any]],
        *,
        admit: Callable[[Batch], None],
        resources: ResourceRuntime | None = None,
        stores: Sequence[TransactionalStore] = (),
        replay: ReplayStore | None = None,
    ) -> None:
        self.sessions = sessions
        self.effect = effect
        self.admit = admit
        self.resources = resources or ResourceRuntime((), totals={})
        self.stores = tuple(stores)
        self.replay = replay or ReplayStore()

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        parsed = Batch.from_wire(batch)
        replay = self.replay.lookup(parsed.ops, step_id=parsed.step_id)
        if replay is not None:
            return replay
        new_request_ids = {int(item["req_id"]) for item in parsed.new_reqs}
        self.sessions.validate_operations(parsed.ops, new_request_ids)
        transaction = self.sessions.begin_step(
            parsed.step_id,
            parsed.ops,
            self.resources,
            self.stores,
        )
        try:
            self.admit(parsed)
            response = self.effect(
                parsed,
                defer_text_cpu_results=defer_text_cpu_results,
            )
            response = self._validate_and_stamp(parsed, response)
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise
        self.replay.commit(parsed.ops, response)
        return response

    @staticmethod
    def _validate_and_stamp(parsed: Batch, response: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise invalid_descriptor("operation executor must return a result map")
        result = dict(response)
        step_id = wire_int(result.get("step_id"), "operation result.step_id", minimum=0)
        if step_id != parsed.step_id:
            raise invalid_descriptor(
                f"operation result step {step_id} does not match batch step {parsed.step_id}"
            )
        per_seq = result.get("per_seq")
        if not isinstance(per_seq, list):
            raise invalid_descriptor("operation result.per_seq must be a list")
        if len(per_seq) != len(parsed.ops):
            raise invalid_descriptor(
                f"operation result contains {len(per_seq)} rows for {len(parsed.ops)} operations"
            )
        stamped: list[Any] = []
        for index, (operation, sequence_result) in enumerate(zip(parsed.ops, per_seq, strict=True)):
            if isinstance(sequence_result, Mapping):
                output = dict(sequence_result)
                request_id = wire_int(
                    output.get("req_id"),
                    f"operation result.per_seq[{index}].req_id",
                    minimum=0,
                )
                if request_id != operation.session_id:
                    raise invalid_descriptor(
                        f"operation result row {index} belongs to session {request_id}, "
                        f"expected {operation.session_id}"
                    )
                _stamp_result(output, operation)
                stamped.append(output)
                continue
            if isinstance(sequence_result, FinalizableSeqResult):
                request_id = wire_int(
                    getattr(sequence_result, "req_id", None),
                    f"operation result.per_seq[{index}].req_id",
                    minimum=0,
                )
                if request_id != operation.session_id:
                    raise invalid_descriptor(
                        f"operation result row {index} belongs to session {request_id}, "
                        f"expected {operation.session_id}"
                    )
                stamped.append(_VersionedSeqResult(sequence_result, operation))
                continue
            raise invalid_descriptor(
                f"operation result.per_seq[{index}] must be a map or finalizable result"
            )
        result["per_seq"] = stamped
        return result
