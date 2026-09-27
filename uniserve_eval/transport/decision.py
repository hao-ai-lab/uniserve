"""Executes decision-readout requests and records their answers.

A readout answers with one JSON body after the server has read every
question, so the request's latency closes when the complete body arrives and
there is no token timing. System One (`/v1/systemone`) answers one state
with `{model, answers, usage}`; DJev (`/api/evaluate`) answers each state of
its `states` list with the state's `answers` and `prompt_tokens`.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..types import DJEV_EVALUATE, RequestRecord


async def send_decision(
    client: httpx.AsyncClient,
    url: str,
    endpoint: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    """POST one readout and fold its answers and usage into `record`.

    The record succeeds only when every requested question of every state
    has an answer object. Its prompt length is the server's count of encoded
    prompt tokens (`usage.input_tokens`, or DJev's per-state
    `prompt_tokens`); System One also reports `usage.output_tokens`, which a
    readout defines as zero.

    Failure classifiers: `transport_status_<code>` for an error status or a
    body that is not JSON, `invalid_decision_response` for a body without
    the endpoint's answer structure, and `incomplete_answers` when a
    requested question has no answer.
    """
    response = await client.post(url, json=payload)
    record.note_http(response.status_code)
    record.close_now()
    if response.status_code >= 400:
        record.mark_failure(
            f"transport_status_{response.status_code}", response.text[:500]
        )
        return
    try:
        data = response.json()
    except Exception:
        record.mark_failure(
            f"transport_status_{response.status_code}", response.text[:500]
        )
        return

    if endpoint == DJEV_EVALUATE:
        requested = [
            (state.get("id"), list(state.get("questions", {})))
            for state in payload.get("states", [])
        ]
        answered = _djev_answers(data, record)
    else:
        requested = [(None, list(payload.get("questions", {})))]
        answered = _systemone_answers(data, record)
    if answered is None:
        record.mark_failure("invalid_decision_response", response.text[:500])
        return

    # Answers are matched to the requested questions of each state; a state
    # or question the response omits fails the request.
    by_state: dict[str, Any] = {}
    questions = 0
    for state_id, question_ids in requested:
        state_answers = answered.get(state_id)
        if not isinstance(state_answers, dict) or not all(
            isinstance(state_answers.get(question_id), dict)
            for question_id in question_ids
        ):
            record.mark_failure("incomplete_answers", response.text[:500])
            return
        questions += len(question_ids)
        by_state[str(state_id)] = state_answers

    # A single-state readout keeps its answers keyed by question id, the
    # System One shape, whichever endpoint served it.
    record.answers = (
        next(iter(by_state.values())) if len(by_state) == 1 else by_state
    )
    record.decision_states = len(requested)
    record.decision_questions = questions
    record.mark_success()


def _systemone_answers(
    data: Any, record: RequestRecord
) -> dict[Any, Any] | None:
    """Return a System One response's answers keyed under a single state.

    The response's usage becomes the record's authoritative token counts.
    """
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        return None
    usage = data.get("usage")
    if isinstance(usage, dict):
        if _is_count(usage.get("input_tokens")):
            record.prompt_len = int(usage["input_tokens"])
            record.prompt_len_source = "server_usage"
        if _is_count(usage.get("output_tokens")):
            record.output_len = int(usage["output_tokens"])
            record.output_len_source = "server_usage"
    return {None: data["answers"]}


def _djev_answers(data: Any, record: RequestRecord) -> dict[Any, Any] | None:
    """Return a DJev response's answers keyed by state id.

    DJev reports each state's encoded prompt length as `prompt_tokens`;
    their sum becomes the record's authoritative prompt length.
    """
    if not isinstance(data, dict) or not isinstance(data.get("states"), list):
        return None
    answers: dict[Any, Any] = {}
    prompt_tokens: list[int] = []
    for state in data["states"]:
        if not isinstance(state, dict) or not isinstance(
            state.get("answers"), dict
        ):
            return None
        answers[state.get("id")] = state["answers"]
        if _is_count(state.get("prompt_tokens")):
            prompt_tokens.append(int(state["prompt_tokens"]))
    if prompt_tokens and len(prompt_tokens) == len(data["states"]):
        record.prompt_len = sum(prompt_tokens)
        record.prompt_len_source = "server_usage"
    return answers


def _is_count(value: Any) -> bool:
    """Recognize a non-negative integer count, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
