"""Bounded native host execution for media encoding and cache transfer.

Reserve capacity before producing task inputs. A task owns its result and input
lease; cancellation releases that lease only after its physical producer stops.
"""

from uniserve_worker._uniserve_ipc import HostLane, HostTask

__all__ = ["HostLane", "HostTask"]
