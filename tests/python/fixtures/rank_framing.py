"""A real CPU Worker behind deterministic IPC result fragmentation."""

from collections import deque

from uniserve_worker.bootstrap.cli import create_worker_cli_parser
from uniserve_worker.bootstrap.config import WorkerProcessArgs
from uniserve_worker.bootstrap.ipc import WorkerIpcEndpoint
from uniserve_worker.execution.output import finalize_run_result
from uniserve_worker.worker import Worker


def main():
    config = WorkerProcessArgs.from_namespace(create_worker_cli_parser().parse_args())
    endpoint = WorkerIpcEndpoint(
        config.ipc.service_name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.max_inflight,
    )
    worker = Worker.from_config(config)
    pending = {}
    try:
        worker.warmup()
        while True:
            request = endpoint.recv()
            kind = request["kind"]
            call_id = request["call_id"]
            if kind == "info":
                endpoint.respond(
                    {"kind": "info", "call_id": call_id, "info": worker.info.to_mapping()}
                )
                continue
            if kind == "close":
                endpoint.respond({"kind": "ok", "call_id": call_id})
                break
            if kind == "submit":
                report = finalize_run_result(worker.execute(request["run"])).to_mapping()
                fragments = deque()
                if config.execution.rank == 0 and len(report["completions"]) > 1:
                    for index, completion in enumerate(report["completions"]):
                        products = [
                            product
                            for product in report["products"]
                            if product["product"]["request_key"] == completion["request_key"]
                            and product["product"]["producer_op_id"] == completion["op_id"]
                        ]
                        fragments.append(
                            {
                                **report,
                                "completions": [completion],
                                "products": products,
                                "done": report["done"] and index == len(report["completions"]) - 1,
                                "forward_stats": report["forward_stats"] if index == 0 else None,
                                "worker_exec_us": report["worker_exec_us"] if index == 0 else None,
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
            endpoint.respond({"kind": "result", "call_id": call_id, "result": report})
            if not pending[run_id]:
                del pending[run_id]
    finally:
        worker.close()


if __name__ == "__main__":
    main()
