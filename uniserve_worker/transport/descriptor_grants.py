"""Native allocation descriptor grants for consumers on the same host.

Registration retains its own descriptor until revocation. Consumers receive
an owned descriptor over an abstract Unix socket and close it after import.
Closing the service joins its native thread and releases the socket name.
"""

from uniserve_worker._uniserve_ipc import (
    DescriptorGrants as DescriptorGrants,
)
from uniserve_worker._uniserve_ipc import (
    fetch_descriptor as fetch,
)

__all__ = ["DescriptorGrants", "fetch"]
