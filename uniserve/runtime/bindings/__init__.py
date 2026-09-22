"""Per-call-site preparation of numerical operators within an execution context.

Each binding owns the prepared operators, plans and borrowed backing of one
numerical layer. :class:`uniserve.runtime.ExecutionContext` creates bindings
for the layers it prepares, publishes them to the layers' call scope and
releases them.
"""

import torch


def capturing(device: torch.device) -> bool:
    """Report whether ``device``'s current stream is capturing a graph."""
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()
