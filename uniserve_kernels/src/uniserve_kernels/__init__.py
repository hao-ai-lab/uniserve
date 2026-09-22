"""UniServe device kernels.

Subpackages group kernels by mathematical domain. Every launcher consumes
device tensors and caller-supplied outputs; numerical entry points in
``uniserve.nn.functional`` select a kernel and define the portable formula.
Kernels know nothing about workers, requests, lanes, models or runtime owners.
"""

from __future__ import annotations
