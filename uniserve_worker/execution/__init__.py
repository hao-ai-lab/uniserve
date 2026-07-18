"""Model-backed execution: one engine, one graph runtime.

Per ``specs/unified_forward_execution.md`` completion criterion 1, this
package contains exactly the transactional :mod:`~.engine` (the sole
data-plane seam) and the consolidated :mod:`~.cuda_graph` runtime.
"""
