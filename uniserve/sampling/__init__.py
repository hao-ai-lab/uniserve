"""Sampling options and numerical token selection."""

from uniserve_worker._uniserve_ipc import SamplingParams as SamplingParams

from .categorical import sample_categorical
from .greedy import greedy
from .top_k import sample_top_k

__all__ = ["SamplingParams", "greedy", "sample_categorical", "sample_top_k"]
