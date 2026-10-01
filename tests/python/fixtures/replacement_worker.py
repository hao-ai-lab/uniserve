"""An external rank whose replacement advertises a different capability.

Run by path with the worker CLI arguments. ``UNISERVE_TEST_REPLACEMENT_DIR``
names a directory shared with the test: each rank writes its process id to
``<rank>.pid`` there, and once ``replacement.json`` exists there, the rank's
worker info carries the fields that file holds in place of its own.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.worker import Worker


def main() -> None:
    directory = Path(os.environ["UNISERVE_TEST_REPLACEMENT_DIR"])
    config = parse_worker_args()
    (directory / f"{config.execution.rank}.pid").write_text(str(os.getpid()))

    name = endpoint_name(config)
    with WorkerIpcEndpoint(
        name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
        transport=config.ipc.channel_transport,
    ) as endpoint:
        register_endpoint(config, endpoint.endpoint(name))

        class Rank(Worker):
            @property
            def info(self):
                value = super().info
                change = directory / "replacement.json"
                return (
                    replace(value, **json.loads(change.read_text()))
                    if change.exists()
                    else value
                )

        with Rank.from_config(config) as worker:
            worker.bind(endpoint)
            worker.run()


if __name__ == "__main__":
    main()
