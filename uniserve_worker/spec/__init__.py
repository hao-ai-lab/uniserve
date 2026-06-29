"""Speculative decoding helpers shared by model implementations."""

from .sampler import SpeculativeSampleResult, speculative_sample_target_only

__all__ = ["SpeculativeSampleResult", "speculative_sample_target_only"]
