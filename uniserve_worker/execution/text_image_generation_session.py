"""Text-image generation serving-session orchestration."""
from __future__ import annotations

from typing import Any

__all__ = [
    "TextImageGenerationSession",
]


class TextImageGenerationSession:
    """Owns request lifecycle state for text-image generation flows."""

    def __init__(self, owner: Any, req_id: int | None = None) -> None:
        self.owner = owner
        self.req_id = None if req_id is None else int(req_id)
        self.records: dict[int, dict[str, Any]] = {}
        self.generated: dict[int, Any] = {}

    def begin_request(
        self,
        req_id: int,
        *,
        sampling: dict[str, Any] | None = None,
        image: dict[str, Any] | None = None,
        neg_token_ids: list[int] | None = None,
        lora_id: Any = None,
    ) -> dict[str, Any]:
        record = {
            "sampling": dict(sampling or {}),
            "image": dict(image or {}),
            "neg_token_ids": list(neg_token_ids or []),
            "lora_id": lora_id,
            "dims": None,
        }
        self.records[int(req_id)] = record
        self.release_generated_state(req_id)
        return record

    def record(self, req_id: int) -> dict[str, Any]:
        return self.records.setdefault(int(req_id), {})

    def generation_state(self, req_id: int) -> Any | None:
        return self.generated.get(int(req_id))

    def set_generation_state(self, req_id: int, value: Any) -> None:
        self.generated[int(req_id)] = value

    def release_generated_state(self, req_id: int) -> Any | None:
        state = self.generated.pop(int(req_id), None)
        if state is not None:
            self.release_paged_branches(state)
        return state

    def release_request(self, req_id: int, *, request_state: Any | None = None) -> None:
        target = int(req_id)
        self.release_generated_state(target)
        self.records.pop(target, None)
        if request_state is not None:
            request_state.kv_lengths.pop("default", None)

    def release_paged_branches(self, state: Any) -> None:
        branches = getattr(state, "paged_branches", None)
        if branches is None:
            return
        branches.release(self.owner.residency)
        state.paged_branches = None

    def start(self, op: dict[str, Any]) -> Any:
        state = getattr(self.owner, "_state", None)
        if callable(state):
            return state(int(op["req_id"]))
        return None

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        encode = getattr(self.owner, "encode_image", None)
        if callable(encode):
            return encode(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose image encoding")

    def prepare_denoise(self, state: Any, op: dict[str, Any]) -> Any:
        prepare = getattr(self.owner, "prepare_denoise", None)
        if callable(prepare):
            return prepare(state, op)
        raise RuntimeError("generation session owner does not expose denoise preparation")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        predict = getattr(self.owner, "predict_velocity", None)
        if callable(predict):
            return predict(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose denoise prediction")

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        decode = getattr(self.owner, "decode_image", None)
        if callable(decode):
            return decode(*args, **kwargs)
        raise RuntimeError("generation session owner does not expose image commit")

    def release(self, req_id: int | None = None) -> None:
        target = self.req_id if req_id is None else int(req_id)
        if target is not None:
            self.release_request(target)
