"""Distributed transaction coordination: one logical engine per rank group.

Dormant Stage 9 deliverable from ``specs/unified_forward_execution.md``. The
control plane observes one logical engine; ranks share exact logical state
(the single session registry in `ExecutionEngine`) while each rank owns its
local executor and physical residency. The protocol laws implemented here:

* every rank verifies one configuration fingerprint before the group exists —
  a mismatch is a startup failure, not a runtime discovery;
* prepare is an all-rank agreement: any rank's pre-launch failure aborts
  every other rank's reservation and surfaces as a typed noncommitted error
  (the step may retry);
* after launch, the transaction is group-atomic: a rank failure or any
  disagreement in finalized results poisons the entire logical engine epoch,
  aborting still-provisional ownership on every rank;
* one designated result rank produces the durable result; other ranks must
  agree exactly (logical results are rank-consistent by law).

The fan-out composes with the existing single-engine machine: it *is* a
`TransactionExecutor`, so the exactly-once window, acknowledgement eviction,
and poisoning semantics come from `ExecutionEngine` unchanged. Real
multi-process transports (NCCL agreement, rank-sharded pools) bind behind
the same seam at cutover; nothing routes production traffic here.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.execution import ExecuteBatch, RowResult, SessionDelta
from ..runtime.immutable_session import RequestSession
from .engine import EngineExecutionError, PreLaunchRejection
from .transaction import PreparedTransaction, StandardTransactionExecutor

__all__ = [
    "DistributedConfigurationError",
    "RankDisagreement",
    "RankFanOutExecutor",
    "RankMember",
]


class DistributedConfigurationError(RuntimeError):
    """Ranks disagree about static configuration before readiness."""


class RankDisagreement(EngineExecutionError):
    """Ranks produced conflicting logical results after launch."""


@dataclass(frozen=True, slots=True)
class RankMember:
    """One rank's local execution half plus its static identity."""

    rank: int
    configuration_fingerprint: str
    executor: StandardTransactionExecutor


@dataclass(slots=True)
class _GroupPrepared:
    prepared: tuple[PreparedTransaction, ...]


class RankFanOutExecutor:
    """All-rank prepare/launch agreement behind one TransactionExecutor."""

    def __init__(self, members: tuple[RankMember, ...], *, result_rank: int = 0) -> None:
        if not members:
            raise DistributedConfigurationError("a rank group needs members")
        ranks = [member.rank for member in members]
        if sorted(ranks) != list(range(len(members))):
            raise DistributedConfigurationError(
                f"rank identities must be dense from zero; got {sorted(ranks)}"
            )
        fingerprints = {member.configuration_fingerprint for member in members}
        if len(fingerprints) != 1:
            raise DistributedConfigurationError(
                "all ranks must share one configuration fingerprint; got "
                f"{sorted(fingerprints)}"
            )
        if result_rank not in ranks:
            raise DistributedConfigurationError(
                f"designated result rank {result_rank} is not a member"
            )
        self._members = tuple(sorted(members, key=lambda member: member.rank))
        self._result_rank = result_rank

    # ------------------------------------------------------------------ #

    def prepare(
        self,
        batch: ExecuteBatch,
        sessions: tuple[RequestSession, ...],
    ) -> _GroupPrepared:
        """All-rank prepare agreement; any failure leaves nothing reserved."""

        prepared: list[PreparedTransaction] = []
        try:
            for member in self._members:
                prepared.append(member.executor.prepare(batch, sessions))
        except Exception as error:
            for transaction in prepared:
                transaction.reservation.abort()
            if isinstance(error, EngineExecutionError):
                raise
            raise PreLaunchRejection(str(error)) from error
        return _GroupPrepared(prepared=tuple(prepared))

    def launch(
        self,
        prepared: _GroupPrepared,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]:
        """Group-atomic launch: rank failure or disagreement poisons upstream."""

        outcomes: list[tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]] = []
        try:
            for member, transaction in zip(self._members, prepared.prepared):
                outcomes.append(member.executor.launch(transaction))
        except Exception:
            # The failing rank aborted its own reservation; abort the rest
            # (committed ranks cannot be uncommitted — interpretation of the
            # group state is unsafe, which is exactly why the engine poisons).
            for member, transaction in zip(self._members, prepared.prepared):
                transaction.reservation.abort()
            raise
        designated = outcomes[self._result_rank]
        for member, outcome in zip(self._members, outcomes):
            if outcome != designated:
                raise RankDisagreement(
                    f"rank {member.rank} finalized conflicting logical results"
                )
        return designated
