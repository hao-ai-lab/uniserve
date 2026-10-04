"""Native transport reads with distinct readiness and physical retirement.

A ready ticket exposes views and a fence its consumer stream can wait on.
Copy reads retire after their backend drains device access; borrowed views
retire after every consuming stream completes. Cancellation revokes access
without releasing storage still in use.
"""

from uniserve_worker._uniserve_ipc import TransferTicket as TransferTicket
