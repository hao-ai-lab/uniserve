"""Normalization kernels with residual and modulation epilogues.

Statistics and affine expressions accumulate in FP32 and round once, when
stored into caller-supplied outputs or operands updated in place.

- ``rms``: row-wise RMS normalization and residual-add RMS normalization.
- ``residual``: weighted RMS normalization and channel-scaled residual
  updates, alone or followed by RMS or layer normalization; the normalizing
  launches optionally record per-row magnitude partials.
- ``modulation``: RMS normalization with row-indexed shift and scale
  modulation, and row-indexed gated residuals.
- ``sandwich``: sandwich normalization of a residual stream (normalized
  updates added to the residual, scaled, and further normalizations of the
  result) in one launch, bit-identical to ``rms`` launches and tensor
  operations.
- ``frame``: per-frame group normalization, SiLU and causal padding of video
  frames in two passes, bit-identical to PyTorch's normalization, SiLU and
  padding.

Each module pairs an ``unsupported`` eligibility check with its launches;
``uniserve.nn.functional`` validates inputs, allocates outputs, raises on
CUDA with the check's reason, and evaluates the same formulas with tensor
operations off CUDA.
"""
