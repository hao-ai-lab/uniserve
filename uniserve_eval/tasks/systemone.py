"""Defines decision-readout benchmark behavior for System One servers.

A decision readout reads a probability distribution over each question's
candidates from one state; it generates no tokens. The canonical endpoint is
TypeSafe System One (`POST /v1/systemone`, OpenAPI 0.2.0), which carries one
state per request. DJev's `POST /api/evaluate` serves the same readout
semantics under its NanoJev-compatible schema and is the reference system,
so a point may address it instead: each request then carries one state in a
`states` list, with the corpus spellings of the question type and images.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ..types import (
    DJEV_EVALUATE,
    SYSTEMONE,
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)
from .base import BenchmarkTask, ImageCountRule


class SystemOneTask(BenchmarkTask):
    """Builds decision readouts and validates their answered questions."""

    name: ClassVar[TaskName] = TaskName.SYSTEMONE
    allowed_endpoints: ClassVar[tuple[str, ...]] = (SYSTEMONE, DJEV_EVALUATE)
    default_endpoint: ClassVar[str] = SYSTEMONE
    default_stream: ClassVar[bool] = False
    accepts_image: ClassVar[bool] = False
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        """Build one single-state readout request for the point's endpoint.

        System One requests carry `model`, `state`, `questions`, and, when
        the row has images, the `x_images` extension. DJev requests carry
        the same state with `noul` questions spelled `boolean` and the
        images as `images`; DJev serves one model and takes no model name.

        Raises:
            ValueError: If the row has no questions.
        """
        if not example.questions:
            raise ValueError(f"readout row {example.id!r} has no questions")

        if self.point.endpoint == DJEV_EVALUATE:
            state: dict[str, Any] = {
                "id": example.id,
                "state": example.state,
                "questions": {
                    question_id: _djev_question(question)
                    for question_id, question in example.questions.items()
                },
            }
            if example.images:
                state["images"] = list(example.images)
            return TaskRequest(DJEV_EVALUATE, {"states": [state]}, stream=False)

        payload: dict[str, Any] = {
            "model": self.point.model,
            "state": example.state,
            "questions": example.questions,
        }
        if example.images:
            payload["x_images"] = list(example.images)
        return TaskRequest(SYSTEMONE, payload, stream=False)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Require answered questions and server-reported prompt lengths.

        The transport classifies a response that leaves a requested question
        unanswered as a failure, so these checks cover what a successful
        response must also carry: at least one answered question, and the
        prompt length the server encoded rather than a client estimate.
        """
        return ValidationResult(
            checks={
                "answered_questions": bool(records)
                and all(record.decision_questions > 0 for record in records),
                "server_prompt_tokens": bool(records)
                and all(
                    record.prompt_len_source == "server_usage"
                    for record in records
                ),
            }
        )


def _djev_question(question: dict[str, Any]) -> dict[str, Any]:
    """Spell a System One question in DJev's schema.

    DJev names the official `noul` type `boolean`; every other field is
    shared.
    """
    if question.get("type") != "noul":
        return dict(question)
    return {**question, "type": "boolean"}
