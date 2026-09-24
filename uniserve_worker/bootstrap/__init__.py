"""Worker startup, role selection, process arguments, and IPC composition.

``launch`` opens the rank's IPC endpoint and reports it to the head before
building the ``Worker``. Sizing modules run during that build: ``capacity``
fits fixed storage into the device grant, ``cache`` describes the K/V pool,
and ``report`` assembles the ``WorkerInfo`` the rank advertises.
"""
