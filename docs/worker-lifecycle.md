# Worker startup and service lifecycle

The caller owns the endpoint and Worker in one resource scope:

```python
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.worker import Worker

service = endpoint_name(config)
with WorkerIpcEndpoint(
    service,
    max_payload=config.ipc.max_payload_bytes,
    max_inflight=config.ipc.queue_depth,
) as endpoint:
    register_endpoint(config, service)
    with Worker.from_config(config) as worker:
        worker.bind(endpoint)
        worker.run()
```

The rank names its own endpoint and reports it to the head's registration address, which the launch descriptor carries; the head binds the rank's channel from that report. Reporting happens once the endpoint exists and before model loading, so the head establishes the bounded connection during startup. Worker resources close before the endpoint, and a failed model construction still closes it. The production entry is `uniserve_worker.bootstrap.launch.run_worker`.

## Construction and ownership

`Worker.from_config(config)` resolves the checkpoint through `uniserve_models`, loads its typed numerical composition through `uniserve.loading`, and constructs execution resources, physical storage, lanes and data-plane transports. Tokenizer, image processing and flow prompt assets remain with the worker's input consumers. It performs no numerical warmup or graph capture. Construction owns rollback until it successfully returns a Worker; failed construction releases acquired resources while preserving the original exception.

`Worker(model, ...)` accepts an already-loaded model and constructs the same execution resources without requiring IPC. ModelRunner resolves attention selection and binds persistent model inputs and workspaces before Worker measures the remaining memory grant. Request capacity is then resolved against that grant and shared by Worker and ModelRunner; sizing before those allocations would overestimate available storage.

`build_worker_layout()` combines model geometry, component bindings, enabled calls, and lane admission limits into the final `WorkerInfo`. Resource allocation, warmup, and the IPC handshake consume that same report. The report advertises the encoder entry count and the maximum bytes per feature independently of total buffer-pool capacity; the scheduler bounds feature reservations by both limits. Admission limits do not reduce the storage reservations required by warmup and graph capture. The configuration identity includes the final advertised capabilities and limits, while excluding the endpoint incarnation and transient free-memory measurement.

Distributed initialization selects the local device and establishes the process world before loading rank-local weights. `bootstrap.distributed.initialize_entries()` constructs ModelEntry bindings and component meshes, including local meshes for temporal decoders. ProcessGroups owns the process groups created during bootstrap and releases component groups before its default world; a process world supplied by the embedding caller is borrowed. `load_worker_model()` loads weights against these bindings, after which Worker selects attention providers and allocates execution resources. Triton selects its bundled assembler for the target GPU architecture; Worker does not modify compiler paths or probe system toolkit locations.

`from_config()` explicitly owns distributed initialization during construction and transfers that ownership to the returned Worker. Failed construction closes the acquired environment and preserves the original exception. A directly used `ProcessGroups` context closes on both normal and exceptional exit. The outer Worker context owns the complete lifetime of the returned object.

Worker context exit releases execution resources, pending results, graphs, model references, data-plane transports, and owned distributed resources. Cleanup attempts every release. If the context body raised, cleanup errors become notes on that original exception. If the body succeeded, a cleanup failure propagates normally. `close()` is idempotent and remains available to callers managing lifetimes explicitly with `try/finally`. A closed Worker cannot enter another context, bind, run, warm up, or execute.

The IPC endpoint remains caller-owned throughout. Worker removes completion notifications and its borrowed endpoint reference after asynchronous producers stop; Worker never closes the endpoint. Keep the endpoint open until the Worker context exits or `close()` finishes. Native endpoint `close()` is idempotent and rejects concurrent closure during an endpoint call.

## Binding and execution

`Worker.bind(endpoint)` borrows an open endpoint, initializes service queues and in-flight result state, registers completion notifications, and returns the same Worker. Binding the same or a different endpoint again raises without replacing the binding. A closed endpoint is rejected. Binding does not own resource cleanup.

`Worker.run()` requires a binding and permits one blocking invocation. It prepares fixed modules, performs numerical warmup and graph capture, exercises startup scenarios, and checks readiness before receiving any request, including `Info`. It then drives admission, dependencies, execution progress and response delivery in the calling process and thread. It does not create a service thread or child process. Existing execution resources may perform asynchronous CPU, transfer, and device work.

`run()` returning or raising does not close Worker. The surrounding resource scope owns cleanup, including failures from binding, warmup, service execution, or caller code. Let startup and runtime exceptions leave that scope; partially initialized or failed execution is not a retry contract. A second `run()` invocation is rejected even before scope exit.

Direct consumers use `Worker.submit(batch)` without binding IPC. It returns the BatchState owned by Worker. The caller drives `Worker.advance()` to progress dependencies, CPU work and retirement, then calls `Worker.poll(state)` to consume the batch's BatchOutput. A `None` result means the result is not ready. Continue advancing until it arrives; consuming that state again is invalid. The caller must preserve request and collective submission order and keep the Worker scope alive throughout this process.

Explicit `warmup()` is optional for direct execution; full graph policy captures required configurations on first use before the catalog is sealed. Successful warmup is retained for the Worker's lifetime and reused by a subsequent `run()`; warmup failures leave cleanup to the owner. Warmup prepares numerical providers and graph captures, exercises synthetic requests, retires startup collective identities, and checks device memory grants before marking readiness. Successful scenarios retire their synthetic requests through the ordinary execution protocol. A failed scenario propagates its error immediately and leaves resource release to the enclosing Worker scope.

## Protocol and shutdown

Each submission batch is submitted once and answered once. A batch carries one computation through one entry, so the rank executes it as a single homogeneous group and `Submit` returns that group's one result. Delivery and safe resource retirement release its state. Completed results cannot be collected again. Duplicate submissions return `InvalidDescriptor` before applying admissions, commands, or computation.

IPC version 61 requires a matching Rust frontend and Python native extension. Version mismatch is rejected at the transport boundary; rebuild both artifacts together. Request identity is `(engine_id, request_id, request_epoch)`; retained buffers keep the exact originating identity across request-slot reuse. Each computation is identified by its `(batch_id, request_index)`, preserved when a batch is split across the ranks of its component. Every call also states the coordinates its call assumes — logical position, KV visible and computed lengths, and flow step — and a call whose coordinates contradict the request's recorded progress is refused.

`batch_id` values strictly increase in each Worker's actual transport submission order. Worker stores one high-water mark for its lifetime; no completed-batch history is retained. The Rust executor allocates a batch identity when it assembles the batch, and dispatches it to the ranks of the one component it addresses. Batches may become ready out of order, and each response retains its own `batch_id`. ID gaps are valid. Counter exhaustion is an error. Waiting timeouts continue waiting on the original transport request; they never submit it again. Rank death uses the existing scoped failure and replacement handling, retiring affected work rather than resubmitting its batch.

Worker advances dependency-ready work in submission order within available execution and transport credits. It assigns no priority by call kind. Asynchronous input preparation and independent completions can overlap; shared request state and cooperative collectives retain their correctness ordering. `Free` revokes acquisition on acceptance so a pending allocation cannot prevent its release; acknowledgement still waits for existing physical readers to retire.

The service preserves configured payload bounds, queue depth, backpressure and ready-response delivery. `queue_depth` bounds the batches in flight on one rank, and in-flight execution occupies that capacity. `Close` stops admission and acknowledges after accepted work, delivered responses, and safe retirement drain. Protocol errors retain the existing error response and fatal-shutdown behavior.

`Close` ends request processing; resource release follows when the owning scope exits. Request shutdown through this protocol rather than closing an endpoint underneath a running Worker. Cyclic garbage collection is temporarily disabled only during the request loop and restored before scope cleanup.

## CallKind bindings

ModelRunner resolves configured devices and domains into ModelEntry bindings. Each CUDAStream retains its physical stream and dependency events. A partitioned stream owns a Green Context or borrows its parent's actual SM allocation; child streams close before that parent. ModelRunner owns InputBuffers and batch/module CUDAGraph resources; DenoisingRunner owns request-slot denoising graphs. Calls on one numerical entry serialize access to its workspace and graphs. Independent entries receive separate computation streams and mutable resources while sharing read-only parameters.

Green Context bindings enqueue computation and communication within their assigned SM partition, including graph capture. Streamed communication uses a transport stream in the same context, with producer and consumer dependencies. Default full-device execution retains its process-group and peer-reduction providers. ModelEntry owns its bound communicator and borrows shared numerical resources from ModelRunner.

Warmup, capture, eager execution and graph replay use these same bindings. ModelRunner groups homogeneous numerical inputs by actual capability, entry, device, stream and shape, then returns outputs aligned to their call indices. Batch retains the submitted protocol columns. Each completion group selects its entry's stream before binding inputs and carries that stream through output capture, publication and failure retirement. Computations whose predicates are false reserve no numerical resources. Construction rejects unsupported or conflicting bindings without changing the requested graph or device policy.

Persistent cross-call buffers and scalar relays use exportable CUDA physical allocations. A publication borrows eligible backing directly or materializes ordinary CUDA tensor spans into shared storage. A reader receives the allocation's file descriptor only after the source endpoint validates its complete locator and grants ownership. Producer events order copies; the read releases its mapping before acknowledging the source. Source allocations, read capacity and published storage remain retained through physical completion. These transfers do not require source and destination processes to share device ordinals.

Cleanup drains computation and abandons unsubmitted output work. Submitted CPU jobs finish before their mux sessions close. Graphs release before their backing storage and attention metadata; pinned staging and all retained views release while producer streams are still alive. Streams and Green Contexts close after those consumers, and ProcessGroups closes last. Failures during partial construction release acquired resources and preserve the allocation error. Free and Finish revoke logical acquisition while storage and transfer owners retain existing readers until physical completion permits reuse.

## Model loading and diagnostics

Weights remain fixed for a Worker's lifetime. Loading, quantization and model identity are established at construction; changing weights requires a new Worker. The loader accepts `auto`, `safetensors`, `pt`, `dummy`, and `layered`. Standard indexed safetensors shards, component sources, parameter conversion and runtime tensor/pipeline partitioning share the ordinary component loading path.

`graph_policy` (`--graph-policy off|auto|full`) is the graph master switch. Prefill graph eligibility, shape buckets and memory tuning remain separate settings. Captured graphs live with their computation bindings and release on shape replacement or Worker closure.

`--video-graph-shapes` declares the video request shapes, as `SECONDSxTOKENS` items, whose denoising ladders warmup makes resident: one captured graph per ladder step on every request slot, before the worker reports ready. A shape a deployment does not declare still serves; its first request on a slot captures that ladder on its own path. The declared shapes also bound how many numerical sizes stay prepared, so declaring more shapes than a deployment serves costs startup time and device memory.

`UNISERVE_TORCH_PROFILER_DIR` enables PyTorch capture. `UNISERVE_PROFILE_ACTIVITIES` selects `CPU,GPU`, and `UNISERVE_PROFILE_START_STEP` / `UNISERVE_PROFILE_STEPS` select the execution window. Prefix, stack and shape recording use `UNISERVE_PROFILE_PREFIX`, `UNISERVE_PROFILE_WITH_STACK` and `UNISERVE_PROFILE_RECORD_SHAPES`. `UNISERVE_CUDA_PROFILER` controls the CUDA profiler API within that window. `UNISERVE_NVTX` independently enables NVTX ranges. Normal errors and forward statistics remain available without a custom event recorder.

For Python call profiling, set `UNISERVE_DIAGNOSTIC_DIR` and pass `--worker-python scripts/profile_worker.sh` to the server. This dedicated launcher runs the standard `cProfile` module around the complete Python process, including model loading and warmup, and writes one `.pstats` file per process on normal exit. Close the server gracefully to flush the file. Worker has no call-profiler window or fault-injection controls; failure tests terminate their child processes externally.
