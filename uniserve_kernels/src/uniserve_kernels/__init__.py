"""UniServe device kernels.

Modules and subpackages group kernels by domain: norm, activation,
reduction, rope, routing, patch, frame, quantization, cache, attention,
diffusion and peer_storage.
Launchers consume device tensors. Many write caller-supplied outputs and pair
with a separate eligibility check such as ``activation.unsupported``, which
returns the first unmet condition as a reason. Callers such as the numerical
entry points in ``uniserve.nn.functional`` run that check, raise with its
reason on CUDA, launch the kernel and define the portable formula for other
devices, while provider adapters in ``uniserve.runtime.backends.attention``
bind the attention kernels. Kernels know nothing about workers, requests,
lanes, models or runtime owners. The package build compiles the C++ and CUDA
extensions (``setup.py``).
"""

from __future__ import annotations
