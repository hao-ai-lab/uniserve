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

        with Worker.from_config(config) as worker:
            while True:
                request = endpoint.recv()
                kind = request["kind"]
                message_id = request["message_id"]
                if kind == "info":
                    info = worker.info
                    change = directory / "replacement.json"
                    if change.exists():
                        info = replace(info, **json.loads(change.read_text()))
                    endpoint.respond(
                        {
                            "kind": "info",
                            "message_id": message_id,
                            "info": info.to_mapping(),
                        }
                    )
                elif kind == "close":
                    endpoint.respond({"kind": "ok", "message_id": message_id})
                    break
                else:
                    raise ValueError(f"unsupported worker request {kind}")


if __name__ == "__main__":
    main()
