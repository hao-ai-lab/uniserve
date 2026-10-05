"""Native tensor coverage planning and reserved physical read submission.

The planner prefers bound process-local replicas, then other locations in
publication order. It checks the complete destination and requested coverage
before taking read credits. Submission reserves the whole fan-out; a backend
or retain callback failure cancels submitted reads while their physical
accesses remain owned by the transfer tickets.

Numerical backends supply tensor views and copies. Rust owns region selection,
read admission and cancellation through the rank's shared transfer capacity.
"""

from uniserve_worker._uniserve_ipc import fetch_tensor

__all__ = ["fetch_tensor"]
