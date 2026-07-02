from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..spec import BenchmarkSpec

RequestKind = Literal["openai_chat", "native_generate", "images_generations"]


@dataclass(frozen=True)
class TaskRequest:
    endpoint: str
    payload: dict[str, Any]
    # Wire shape, dispatched by ``core.client.send_request``.
    kind: RequestKind


class BenchmarkTask:
    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        raise NotImplementedError
