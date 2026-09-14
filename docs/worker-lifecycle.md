# Worker startup and service lifecycle

The caller owns the endpoint and Worker in one resource scope:

```python
from uniserve_worker.bootstrap.ipc import WorkerIpcEndpoint
from uniserve_worker.worker import Worker

with (
    WorkerIpcEndpoint(
        config.ipc.service_name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.max_inflight,
    ) as endpoint,
    Worker.from_config(config) as worker,
):
    worker.bind(endpoint)
    worker.run()
```

Context managers enter from left to right and exit in reverse order. The endpoint opens before model loading, allowing the frontend to establish the bounded IPC connection during startup. Worker resources close before the endpoint. If model construction fails, the endpoint still closes. The production entry is `uniserve_worker.bootstrap.launch.run_worker`.

## Construction and ownership

`Worker.from_config(config)` resolves the checkpoint through `uniserve_models`, loads its typed numerical composition through `uniserve.loading`, and constructs execution resources, physical storage, lanes and data-plane transports. Tokenizer, image processing and flow prompt assets remain with the worker's input consumers. It performs no numerical warmup or graph capture. Construction owns rollback until it successfully returns a Worker; failed construction releases acquired resources while preserving the original exception.

`Worker(model, ...)` accepts an already-loaded model and constructs the same execution resources without requiring IPC. ModelRunner resolves attention selection and binds persistent model inputs and workspaces before Worker measures the remaining memory grant. Request capacity is then resolved against that grant and shared by Worker and ModelRunner; sizing before those allocations would overestimate available storage.

`build_worker_layout()` combines model geometry, component bindings, enabled operations, and lane admission limits into the final `WorkerInfo`. Resource allocation, warmup, and the IPC handshake consume that same report. Admission limits do not reduce the storage reservations required by warmup and graph capture. The configuration identity includes the final advertised capabilities and limits, while excluding the endpoint incarnation and transient free-memory measurement.

Distributed initialization selects the local device and establishes the process world before loading rank-local weights. `bootstrap.distributed.initialize_entries()` constructs ModelEntry bindings and component meshes, including local meshes for temporal decoders. ProcessGroups owns the process groups created during bootstrap and releases component groups before its default world; a process world supplied by the embedding caller is borrowed. `load_worker_model()` loads weights against these bindings, after which Worker selects attention providers and allocates execution resources. Triton selects its bundled assembler for the target GPU architecture; Worker does not modify compiler paths or probe system toolkit locations.

`from_config()` explicitly owns distributed initialization during construction and transfers that ownership to the returned Worker. Failed construction closes the acquired environment and preserves the original exception. A directly used `ProcessGroups` context closes on both normal and exceptional exit. The outer Worker context owns the complete lifetime of the returned object.

Worker context exit releases execution resources, pending results, graphs, model references, data-plane transports, and owned distributed resources. Cleanup attempts every release. If the context body raised, cleanup errors become notes on that original exception. If the body succeeded, a cleanup failure propagates normally. `close()` is idempotent and remains available to callers managing lifetimes explicitly with `try/finally`. A closed Worker cannot enter another context, bind, run, warm up, or execute.

The IPC endpoint remains caller-owned throughout. Worker removes completion notifications and its borrowed endpoint reference after asynchronous producers stop; Worker never closes the endpoint. Keep the endpoint open until the Worker context exits or `close()` finishes. Native endpoint `close()` is idempotent and rejects concurrent closure during an endpoint operation.

## Binding and execution

`Worker.bind(endpoint)` borrows an open endpoint, initializes service queues and in-flight result state, registers completion notifications, and returns the same Worker. Binding the same or a different endpoint again raises without replacing the binding. A closed endpoint is rejected. Binding does not own resource cleanup.

`Worker.run()` requires a binding and permits one blocking invocation. It prepares fixed modules, performs numerical warmup and graph capture, exercises startup scenarios, and checks readiness before receiving any request, including `Info`. It then drives admission, dependencies, execution progress and response delivery in the calling process and thread. It does not create a service thread or child process. Existing execution resources may perform asynchronous CPU, transfer, and device work.

`run()` returning or raising does not close Worker. The surrounding resource scope owns cleanup, including failures from binding, warmup, service execution, or caller code. Let startup and runtime exceptions leave that scope; partially initialized or failed execution is not a retry contract. A second `run()` invocation is rejected even before scope exit.

Direct consumers use `Worker.submit(batch)` without binding IPC. It returns the BatchState owned by Worker. The caller drives `Worker.advance()` to progress dependencies, CPU work and retirement, then calls `Worker.poll(state)` to consume available BatchOutput fragments. A `None` result means no fragment is ready. Continue advancing and consuming until the fragment's `done` field is true; polling that state again is invalid. The caller must preserve request and collective submission order and keep the Worker scope alive throughout this process.

Explicit `warmup()` is optional for direct execution; full graph policy captures required configurations on first use before the catalog is sealed. Successful warmup is retained for the Worker's lifetime and reused by a subsequent `run()`; warmup failures leave cleanup to the owner. Warmup prepares numerical providers and graph captures, exercises synthetic requests, retires startup collective identities, and checks device memory grants before marking readiness. Successful scenarios retire their synthetic requests through the ordinary execution protocol. A failed scenario propagates its error immediately and leaves resource release to the enclosing Worker scope.

## Protocol and shutdown

Each physical run is submitted once. `Submit` begins execution and returns the first available fragment; `Poll` claims only subsequent, undelivered fragments. A run can have one outstanding response. Terminal delivery and safe resource retirement release its state. Completed results cannot be collected again. Duplicate submissions return `InvalidDescriptor` before applying admissions, commands, or computation.

IPC version 58 requires a matching Rust frontend and Python native extension. Version mismatch is rejected at the transport boundary; rebuild both artifacts together. Request identity is `(engine_id, request_id, request_epoch)`; retained buffers keep the exact originating identity across request-slot reuse. Each computation is identified by its logical `(batch_id, request_index)`, preserved when a batch is split into physical worker runs. The physical `run_id` orders submissions and polling independently of those producer coordinates.

Physical `run_id` values strictly increase in each Worker's actual transport submission order. Worker stores one high-water mark for its lifetime; no completed-run history is retained. The Rust executor allocates physical IDs when dispatching to each Worker, independently of logical batch allocation. Logical batches may become ready out of order, and responses retain their original `batch_id`. ID gaps are valid. Counter exhaustion is an error. Waiting timeouts continue waiting on the original transport request; they never submit it again. Rank death uses the existing scoped failure and replacement handling, retiring affected work rather than resubmitting its physical run.

Worker advances dependency-ready work in submission order within available execution and transport credits. It assigns no priority by operation kind. Asynchronous input preparation and independent completions can overlap; shared request state and cooperative collectives retain their correctness ordering. `Free` revokes acquisition on acceptance so a pending allocation cannot prevent its release; acknowledgement still waits for existing physical readers to retire.

The service preserves configured payload bounds, pipeline depth, backpressure and ready-response delivery. In-flight execution and unclaimed fragments occupy pipeline capacity. `Close` stops admission and acknowledges after accepted work, claimed responses, and safe retirement drain. It discards unclaimed continuations once their execution retires. Protocol errors retain the existing error response and fatal-shutdown behavior.

`Close` ends request processing; resource release follows when the owning scope exits. Request shutdown through this protocol rather than closing an endpoint underneath a running Worker. Cyclic garbage collection is temporarily disabled only during the request loop and restored before scope cleanup.

## Computation bindings

ModelRunner resolves configured devices and domains into ModelEntry bindings. Each CudaStream owns its physical stream, optional Green Context, SM partition and dependency events. ModelRunner owns InputBuffers and batch/module CudaGraph resources; DiffusionRunner owns request-slot denoising graphs. Bindings that execute serially on the same stream may share input storage. Independent streams receive independent mutable staging and graph state while sharing read-only model parameters.

Green Context bindings enqueue NCCL operations directly on their assigned stream, including graph capture. Default full-device execution retains its process-group and peer-reduction providers. ModelEntry owns its bound communicator and borrows shared numerical resources from ModelRunner.

Warmup, capture, eager execution and graph replay use these same bindings. ModelRunner groups each call's numerical ForwardRow inputs by actual entry, device, stream, shape and supported mixed execution, then returns outputs aligned to the original row indices. ScheduleBatch retains the submitted protocol columns; completion groups preserve failure and publication boundaries independently of numerical grouping. Construction rejects unsupported or conflicting bindings without changing the requested graph or device policy.

Cleanup drains computation and abandons unsubmitted output work. Submitted CPU jobs finish before their mux sessions close. Graphs release before their backing storage and attention metadata; pinned staging and all retained views release while producer streams are still alive. Streams and Green Contexts close after those consumers, and ProcessGroups closes last. Failures during partial construction release acquired resources and preserve the allocation error. Free and Finish revoke logical acquisition while storage and transfer owners retain existing readers until physical completion permits reuse.

## Model loading and diagnostics

Weights remain fixed for a Worker's lifetime. Loading, quantization and model identity are established at construction; changing weights requires a new Worker. The loader accepts `auto`, `safetensors`, `pt`, `dummy`, and `layered`. Standard indexed safetensors shards, component sources, parameter conversion and runtime tensor/pipeline partitioning share the ordinary component loading path.

`graph_policy` (`--graph-policy off|auto|full`) is the graph master switch. Prefill graph eligibility, shape buckets and memory tuning remain separate settings. Captured graphs live with their computation bindings and release on shape replacement or Worker closure.

`UNISERVE_TORCH_PROFILER_DIR` enables PyTorch capture. `UNISERVE_PROFILE_ACTIVITIES` selects `CPU,GPU`, and `UNISERVE_PROFILE_START_STEP` / `UNISERVE_PROFILE_STEPS` select the execution window. Prefix, stack and shape recording use `UNISERVE_PROFILE_PREFIX`, `UNISERVE_PROFILE_WITH_STACK` and `UNISERVE_PROFILE_RECORD_SHAPES`. `UNISERVE_CUDA_PROFILER` controls the CUDA profiler API within that window. `UNISERVE_NVTX` independently enables NVTX ranges. Normal errors and forward statistics remain available without a custom event recorder.

For Python call profiling, set `UNISERVE_DIAGNOSTIC_DIR` and pass `--worker-python scripts/profile_worker.sh` to the server. This dedicated launcher runs the standard `cProfile` module around the complete Python process, including model loading and warmup, and writes one `.pstats` file per process on normal exit. Close the server gracefully to flush the file. Worker has no call-profiler window or fault-injection controls; failure tests terminate their child processes externally.
