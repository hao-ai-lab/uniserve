"""Digest-bound replay records for scheduler execution batches."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from ..foundation.errors import invalid_descriptor

__all__ = ["ReplayStore", "batch_digest"]


def batch_digest(batch: Mapping[str, Any]) -> str:
    """Return the canonical digest of one descriptor-only wire batch."""
    try:
        payload = json.dumps(
            batch,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise invalid_descriptor("execute batch contains a non-canonical wire value") from error
    return hashlib.sha256(payload).hexdigest()


class ReplayStore:
    """Bounded committed-result store keyed by scheduler step identity."""

    def __init__(self, capacity: int = 1024) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[int, tuple[str, dict[str, Any]]] = OrderedDict()
        self._latest_committed_step = -1

    def lookup(self, step_id: int, digest: str) -> dict[str, Any] | None:
        step = int(step_id)
        record = self._records.get(step)
        if record is not None:
            recorded_digest, result = record
            if recorded_digest != digest:
                raise invalid_descriptor(
                    f"execute step {step} conflicts with its committed batch digest"
                )
            self._records.move_to_end(step)
            return copy.deepcopy(result)
        if step <= self._latest_committed_step:
            raise invalid_descriptor(f"execute step {step} is stale and has no replay record")
        return None

    def commit(self, step_id: int, digest: str, result: dict[str, Any]) -> None:
        step = int(step_id)
        self._records[step] = (str(digest), copy.deepcopy(result))
        self._records.move_to_end(step)
        self._latest_committed_step = max(self._latest_committed_step, step)
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
