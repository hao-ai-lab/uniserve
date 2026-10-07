"""Attention kernels: video sparse attention, visible-end masks and merges.

The package also builds paged-attention length columns and visible-end
prefix block sparsity, and provides ``prefix_block``: SM100 attention of
query blocks over themselves and a window of their paged prefix. Provider
adapters in ``uniserve.runtime.backends.attention`` bind these kernels, and
third-party attention libraries, to execution resources; the VSA layer in
``uniserve.nn.attention.vsa`` uses the tile and row helpers directly.
"""
