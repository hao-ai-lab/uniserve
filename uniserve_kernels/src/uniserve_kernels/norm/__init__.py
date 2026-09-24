"""Normalization kernels with residual and modulation epilogues.

Statistics and affine expressions accumulate in FP32 and round once, when
stored into caller-supplied outputs or operands updated in place.

- ``rms``: row-wise RMS normalization and residual-add RMS normalization.
- ``residual``: weighted RMS normalization and channel-scaled residual
  updates, alone or followed by RMS or layer normalization; the normalizing
  launches optionally record per-row magnitude partials.
- ``modulation``: RMS normalization with row-indexed shift and scale
  modulation, and row-indexed gated residuals.

Each module pairs a ``can_run`` eligibility check with its launches;
``uniserve.nn.functional`` validates inputs, allocates outputs and evaluates
the same formulas with tensor operations when a check fails.
"""
