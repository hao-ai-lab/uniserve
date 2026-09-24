"""Attention kernels: video sparse attention, visible-end masks and merges.

The package also builds paged-attention length columns and visible-end
prefix block sparsity. Provider adapters in
``uniserve.runtime.backends.attention`` bind these kernels, and third-party
attention libraries, to execution resources; the VSA layer in
``uniserve.nn.attention.vsa`` uses the tile and row helpers directly.
"""
