"""Transport locations retained and revoked by their native storage owners."""

from uniserve_worker._uniserve_ipc import (
    release_exports as release_exports,
)
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.interface import Transport

ExportLocations = tuple[tuple[Transport, Locator], ...]
