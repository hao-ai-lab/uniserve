"""Shared native export selection and retirement for all storage owners.

`export_tensor` exposes one tensor or ordered first-axis spans through each
configured mechanism that reaches its consumers. Device products use local
or CUDA VMM storage; host products use local, shared-memory or rank-channel
transport. A device payload that does not fit its VMM pool uses the available
host mechanisms. Partial failure revokes accepted locations while their
producer accesses and readers continue to govern physical retirement.
"""

from uniserve_worker._uniserve_ipc import export_tensor as export_tensor
from uniserve_worker._uniserve_ipc import release_exports as release_exports
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.interface import Transport

ExportLocations = tuple[tuple[Transport, Locator], ...]
