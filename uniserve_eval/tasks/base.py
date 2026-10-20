"""Defines shared request construction and observable output validation.

A task adapter maps a normalized dataset `Example` to an HTTP request and
judges the resulting `RequestRecord`s. Profile parsing in `config` consults
the adapter's class attributes and `check_*` class methods before a
`BenchmarkPoint` exists; `run_point` then binds an instance to the point to
build requests, inspect each response after measurement, and validate the
measured records.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..types import (
    BenchmarkPoint,
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
    VideoConfig,
)


class BenchmarkTask:
    """Defines the endpoint, request, and validation contract for a task.

    `config` reads these attributes and runs the `check_*` class methods
    while parsing a profile; request building does not recheck them.

    Attributes:
        name: The task identifier; `TASKS` registers each adapter under
            this identifier's value.
        allowed_endpoints: Endpoints `check_endpoint` accepts.
        default_endpoint: Endpoint used when a point declares none.
    """

    name: ClassVar[TaskName]
    allowed_endpoints: ClassVar[tuple[str, ...]]
    default_endpoint: ClassVar[str]

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the task adapter to a resolved benchmark point."""
        self.point = point

    def build_request(self, example: Example) -> TaskRequest:
        """Construct the endpoint request for one normalized example.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def inspect_output(self, record: RequestRecord) -> None:
        """Decode and classify one response after its load slot is released.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def validate(self, records: Sequence[RequestRecord]) -> ValidationResult:
        """Combine request-level checks with task-specific output checks.

        `all_requests_succeeded` fails on an empty record list.

        Raises:
            ValueError: If `validate_output` returns a check named
                `all_requests_succeeded`.
        """
        common = ValidationResult(
            checks={
                "all_requests_succeeded": bool(records)
                and all(record.success for record in records),
            },
            statistics={
                "request_count": len(records),
                "successful_requests": sum(
                    record.success for record in records
                ),
            },
        )
        return common.merged(self.validate_output(records))

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Validate output properties specific to the task.

        Subclasses may override this; the default requires only that records
        exist.
        """
        return ValidationResult(checks={"observable_output": bool(records)})

    @classmethod
    def check_endpoint(cls, endpoint: str | None, context: str) -> str:
        """Resolve and validate an endpoint supported by the task."""
        chosen = endpoint if endpoint is not None else cls.default_endpoint
        if not isinstance(chosen, str) or chosen not in cls.allowed_endpoints:
            allowed = ", ".join(cls.allowed_endpoints)
            raise ValueError(f"{context}.endpoint must be one of: {allowed}")
        return chosen

    @classmethod
    def check_video(cls, video: VideoConfig, context: str) -> None:
        """Validate video settings against the task's request contract.

        Subclasses may override this; the default accepts any settings.
        """
