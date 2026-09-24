"""UniServe device kernels.

Modules and subpackages group kernels by domain: norm, activation,
reduction, rope, patch, quantization, cache, attention and peer_storage.
Launchers consume device tensors. Many write caller-supplied outputs and pair
with a separate eligibility check such as ``activation.can_run``; callers
such as the numerical entry points in ``uniserve.nn.functional`` run that
check, select the kernel and define the portable formula, while provider
adapters in ``uniserve.runtime.backends.attention`` bind the attention
kernels. Kernels know nothing about workers, requests, lanes, models or
runtime owners.
"""

from __future__ import annotations
