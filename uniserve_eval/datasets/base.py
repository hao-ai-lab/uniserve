"""Defines the common dataset loading and configuration contract."""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import BenchmarkPoint, Example


class Dataset:
    """Loads normalized examples for one benchmark point.

    Subclasses are registered by configuration name in `datasets.DATASETS`
    and implement `load`.

    Attributes:
        name: Configuration name, used in validation messages.
        requires_tokenizer: Whether `load` needs a tokenizer. `check_point`
            then requires the profile to set `tokenizer`, while
            `datasets.load_examples` falls back to the point's `model`.
        requires_path: Whether the point must set `dataset_path`.
    """

    name: ClassVar[str]
    requires_tokenizer: ClassVar[bool] = False
    requires_path: ClassVar[bool] = False

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the adapter to its resolved benchmark point."""
        self.point = point

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Load normalized rows in deterministic benchmark order.

        `datasets.load_examples` requires the result to contain exactly
        `point.load.num_prompts` rows and passes a tokenizer only when
        `requires_tokenizer` is set.
        """
        raise NotImplementedError

    @classmethod
    def check_point(
        cls, *, tokenizer: Any, dataset_path: Any, context: str
    ) -> None:
        """Validate path and tokenizer requirements for the adapter.

        `config.load_config` calls this with the raw, environment-expanded
        profile values before the `BenchmarkPoint` exists. An empty string
        counts as unset for a required value.

        Raises:
            ValueError: If a value is present but not a string, or a
                required value is missing or empty.
        """
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
