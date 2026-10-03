"""Loads decision-readout rows in the System One request schema.

Each non-blank line of the JSON Lines file at `dataset_path` is one decision
row: `id`, `state`, `questions`, and optionally `images`. Rows in the
NanoJev and DJev corpus format load unchanged: their `boolean` question type
is the corpus spelling of the official `noul` type, and every field outside
the request contract (gold labels, teacher outputs, metadata) is dropped, so
only the state, the questions, and the images can reach the server.

An optional non-empty ``session_id`` on every row declares a chronological
closed-loop trace. Such rows retain file order instead of being shuffled;
the load driver sends each session's next row after its previous response.
Session identifiers remain client-side metadata.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

# Question types as the corpus spells them, mapped to the official schema.
_QUESTION_TYPES = {
    "noul": "noul",
    "boolean": "noul",
    "choice": "choice",
    "score": "score",
}

# The request fields of one question; anything else in a corpus question is
# evaluation metadata.
_QUESTION_FIELDS = ("type", "instructions", "criteria")


class SystemOneDataset(Dataset):
    """Adapts decision rows to seeded System One readout examples."""

    name: ClassVar[str] = "systemone"
    requires_path: ClassVar[bool] = True

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Return `num_prompts` rows in a seeded order.

        Independent rows are shuffled with `random.Random(point.load.seed)`;
        session traces keep file order. The first `point.load.num_prompts`
        are taken, so a fixed file and
        seed select the same rows in the same order, and consecutive rows
        mix the file's families instead of following its grouping. Question
        order within a row is the file's, which fixes the prompt and canvas
        order of a joint readout.

        Raises:
            ValueError: If a row is not an object, lacks `id`, `state`, or a
                non-empty `questions` object, repeats an id, has a question
                of another type, or has `images` that is not a non-empty
                list of strings.
            OSError: If the file cannot be read.
        """
        point = self.point
        if not point.dataset_path:
            raise ValueError("dataset 'systemone' requires dataset_path")

        rows: list[Example] = []
        seen: set[str] = set()
        path = Path(point.dataset_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = _decision_row(json.loads(line), line_no)
                if row.id in seen:
                    raise ValueError(
                        f"systemone row {line_no} repeats id {row.id!r}"
                    )
                seen.add(row.id)
                rows.append(row)

        # Session traces preserve each client's decision order. Shuffling
        # them would change both closed-loop arrivals and prefix reuse.
        if any(row.session_id is not None for row in rows):
            if any(row.session_id is None for row in rows):
                raise ValueError(
                    "session traces require session_id on every row"
                )
        else:
            random.Random(point.load.seed).shuffle(rows)
        return rows[: point.load.num_prompts]


def _decision_row(raw: Any, line_no: int) -> Example:
    """Normalize one corpus row into a System One readout example."""
    if not isinstance(raw, dict):
        raise ValueError(f"systemone row {line_no} must be an object")
    for key in ("id", "state", "questions"):
        if key not in raw:
            raise ValueError(f"systemone row {line_no} requires {key}")

    questions = raw["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            f"systemone row {line_no} requires a non-empty questions object"
        )
    normalized: dict[str, Any] = {}
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(
                f"systemone row {line_no} question {question_id!r} "
                f"must be an object"
            )
        declared = question.get("type")
        kind = (
            _QUESTION_TYPES.get(declared) if isinstance(declared, str) else None
        )
        if kind is None:
            raise ValueError(
                f"systemone row {line_no} question {question_id!r} has "
                f"unsupported type {declared!r}"
            )
        request = {
            key: question[key] for key in _QUESTION_FIELDS if key in question
        }
        request["type"] = kind
        normalized[str(question_id)] = request

    images = raw.get("images")
    if images is not None and (
        not isinstance(images, list)
        or not images
        or not all(isinstance(image, str) and image for image in images)
    ):
        raise ValueError(
            f"systemone row {line_no} images must be a non-empty string list"
        )

    session_id = raw.get("session_id")
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id
    ):
        raise ValueError(
            f"systemone row {line_no} session_id must be a non-empty string"
        )

    return Example(
        id=str(raw["id"]),
        prompt="",
        state=raw["state"],
        questions=normalized,
        images=list(images) if images is not None else None,
        session_id=session_id,
    )
