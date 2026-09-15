"""Defines the common dataset loading and configuration contract."""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import BenchmarkPoint, Example


class Dataset:
    """Loads normalized examples for one benchmark point."""

    name: ClassVar[str]
    requires_tokenizer: ClassVar[bool] = False
    requires_path: ClassVar[bool] = False

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the adapter to its resolved benchmark point."""
        self.point = point

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Load normalized rows in deterministic benchmark order."""
        raise NotImplementedError

    @classmethod
    def check_point(
        cls, *, tokenizer: Any, dataset_path: Any, context: str
    ) -> None:
        """Validate path and tokenizer requirements for the adapter."""
        if tokenizer is not None and not isinstance(tokenizer, str):
            raise ValueError(f"{context}.tokenizer must be a string")
        if cls.requires_tokenizer and not tokenizer:
            raise ValueError(
                f"{context}.tokenizer is required for dataset {cls.name}"
            )
        if dataset_path is not None and not isinstance(dataset_path, str):
            raise ValueError(f"{context}.dataset_path must be a string")
        if cls.requires_path and not dataset_path:
            raise ValueError(
                f"{context}.dataset_path is required for dataset {cls.name}"
            )
