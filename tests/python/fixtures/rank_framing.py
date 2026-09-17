"""A real CPU Worker behind deterministic IPC result fragmentation."""

import os
from collections import deque
from pathlib import Path

from tests.python.fixtures.depth_one import finalized_report
from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.media.storage import publish_media_bytes
from uniserve_worker.protocol.output import MediaOutput, PosixShmArtifact
from uniserve_worker.worker import Worker


def main():
    config = parse_worker_args()
    service = endpoint_name(config)
    with WorkerIpcEndpoint(
        service,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
    ) as endpoint:
        register_endpoint(config, service)
        with Worker.from_config(config) as worker:
            pending = {}
            worker.warmup()
            while True:
                request = endpoint.recv()
                kind = request["kind"]
                call_id = request["call_id"]
                if kind == "info":
                    endpoint.respond(
                        {
                            "kind": "info",
                            "call_id": call_id,
                            "info": worker.info.to_mapping(),
                        }
                    )
                    continue
                if kind == "close":
                    endpoint.respond({"kind": "ok", "call_id": call_id})
                    break
                if kind == "submit":
                    report = finalized_report(
                        worker, worker.submit(request["run"])
                    ).to_mapping()
                    media_case = os.environ.get("UNISERVE_TEST_MEDIA_RESPONSE")
                    media_rank = 1 if media_case == "rank-output" else 0
                    if (
                        media_case
                        and config.execution.rank == media_rank
                        and report["completions"]
                    ):
                        completion = report["completions"][0]
                        payload = b"generated media content"
                        name = publish_media_bytes(payload)
                        Path(os.environ["UNISERVE_TEST_MEDIA_NAME"]).write_text(
                            name
                        )
                        completion["media_output"] = MediaOutput(
                            handle=PosixShmArtifact(name),
                            bytes=len(payload)
                            + (1 if media_case == "short-storage" else 0),
                        ).to_mapping()
                        if media_case == "unknown-operation":
                            completion["op_id"] = {
                                **completion["op_id"],
                                "request_index": 1000,
                            }
                    fragments = deque()
                    if (
                        config.execution.rank == 0
                        and len(report["completions"]) > 1
                    ):
                        for index, completion in enumerate(
                            report["completions"]
                        ):
                            products = [
                                product
                                for product in report["products"]
                                if product["product"]["request_key"]
                                == completion["request_key"]
                                and product["product"]["producer_op_id"]
                                == completion["op_id"]
                            ]
                            fragments.append(
                                {
                                    **report,
                                    "completions": [completion],
                                    "products": products,
                                    "done": report["done"]
                                    and index == len(report["completions"]) - 1,
                                    "forward_stats": report["forward_stats"]
                                    if index == 0
                                    else None,
                                    "worker_exec_us": report["worker_exec_us"]
                                    if index == 0
                                    else None,
                                }
                            )
                    else:
                        fragments.append(report)
                    run_id = report["run_id"]
                    pending[run_id] = fragments
                elif kind == "poll":
                    run_id = request["run_id"]
                else:
                    raise ValueError(f"unsupported framing request {kind}")
                report = pending[run_id].popleft()
                endpoint.respond(
                    {"kind": "result", "call_id": call_id, "result": report}
                )
                if not pending[run_id]:
                    del pending[run_id]


if __name__ == "__main__":
    main()
