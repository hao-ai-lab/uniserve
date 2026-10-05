# Execution profiling

UniServe's execution sanitizer reads profiler captures and produces hints about host waits, GIL ownership, copies and language crossings. It does not change execution, reject a request or establish that an operation is unnecessary. Inspect each hint in its workload context, then measure any optimization without the diagnostic instrumentation.

## Capture and inspect a CUDA timeline

Set `UNISERVE_NVTX=1` before starting the worker. The native executor emits synchronous ranges named `uniserve.worker.admit`, `prepare`, `inputs`, `execute`, `retire`, `poll`, `result` and `close`, with the batch ID as a numeric payload. Shared resource reclamation uses `uniserve.worker.reap` without a batch payload. The switch is read on first native use. Python numerical ranges remain nested within those stages. Native ranges use the NVIDIA NVTX SDK directly and issue no Python callbacks or CUDA operations.

Capture a representative workload with Nsight Systems. `CAPTURE` is an output prefix and `WORKLOAD` is the caller's Python entry point; select both explicitly. For serving, replace the Python command with the normal `uniserve serve` command and run the fixed workload against it.

```bash
UNISERVE_NVTX=1 nsys profile \
  --trace=cuda,nvtx,python-gil --cuda-trace-all-apis=true \
  --cuda-graph-trace=graph --sample=process-tree \
  --cudabacktrace=sync:0,memory:0 --python-backtrace=cuda \
  --export=sqlite --output="$CAPTURE" \
  .venv/bin/python "$WORKLOAD"

.venv/bin/python -m uniserve.sanitizer trace "$CAPTURE.sqlite"
.venv/bin/python -m uniserve.sanitizer trace "$CAPTURE.sqlite" --json
```

The reader uses the Nsight Systems 2025.4 SQLite export schema. It opens the file read-only. `--scope PREFIX` selects synchronous NVTX ranges by name, defaulting to `uniserve.`; repeated `--rule NAME` options select individual rules. Hints do not cause a failing exit status. Invalid inputs or unreadable exports do.

| Rule | Execution expectation | Evidence and interpretation |
| --- | --- | --- |
| `host-sync` | Batch advancement should enqueue work or query readiness without waiting for unrelated device completion. | A Runtime or Driver device, context, stream or event synchronization call occurred inside the selected scope. Immediate result consumption, initialization, shutdown and failure cleanup may require it. Stream event waits and queries are not host waits. |
| `gil-held-wait` | A native GPU wait should release the GIL so independent Python work can proceed. | A CUDA host synchronization call overlaps a GIL ownership interval on the same thread. The hint reports overlap duration and retains any other threads in the process waiting for the GIL during that overlap. No duration threshold establishes harm; trace overhead, interpreter identity and workload dependencies limit the conclusion. |
| `sync-copy` | Host/device copies should overlap independent work when their consumer allows it. | A synchronous copy API has a correlated host/device transfer. An immediately consumed result or startup transfer can justify synchronous execution. |
| `pageable-copy` | Repeated asynchronous host/device transfers should use reusable pinned host buffers when useful. | A correlated copy uses pageable host memory. An Async API may stage or block; a one-time transfer may not justify pinning. |
| `device-roundtrip` | Intermediate numerical values and device decisions should stay on device when practical. | A D→H copy completes before an H→D submission on the same thread, device and NVTX scope. This ordering does **not** establish that the copies carry the same data; independent inputs or required CPU computation can explain it. |

Each hint retains API names, nanosecond intervals, global thread and CUDA correlation IDs, enclosing scope, batch number when available, native stacks when captured, and correlated copy sizes, streams and memory kinds. Nested numerical scopes inherit the native batch number. Backtrace activity records contribute stacks to their API call rather than another invocation. Copy correlation includes the process, keeping rank-local correlation IDs separate.

GIL evidence comes from Nsight's `GIL Trace` NVTX domain, including the original holding and waiting intervals. The reader matches domains within each process and ownership on the CUDA caller's thread. It omits unclosed intervals and reports the coverage gap. Missing GIL tracing means ownership is unknown; CUDA wait duration and cProfile cannot supply it. A captured waiter establishes temporal contention evidence, not a proven deadlock or a dependency on that GPU operation. The export does not identify individual Python interpreter locks. CPU-only GIL contention, mutex waits and callback dependencies are outside this rule's coverage.

CUDA Graph execution does not itself need the GIL once submitted. Python preparation, submission, callbacks, result handling and final-reference destruction can still acquire or hold it. Use the same GIL rule for graph and eager workloads; it describes observed waits rather than assuming that graph replay eliminates language interaction. Nsight's [Python GIL tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-gil-tracing) supports CPython 3.9 or later on Arm SBSA, Linux x86 and Windows, excluding interpreters built with `Py_GIL_DISABLED=1`.

Coverage counts describe recorded activities. A missing activity, missing stack or unmatched scope is a coverage gap. No hints does not mean no unnecessary synchronization. GPU graph-level capture does not expose every graph node; Python metadata accesses and dependency relationships between independent batches are not inferred from CUDA timing. Use Nsight's timeline and native reports for GPU gaps, utilization and stream dependencies.

For bounded worker capture, the existing `UNISERVE_TORCH_PROFILER_DIR`, `UNISERVE_PROFILE_START_STEP`, `UNISERVE_PROFILE_STEPS` and `UNISERVE_CUDA_PROFILER` settings can drive `--capture-range=cudaProfilerApi`. This window brackets numerical execution steps: the first batch's earlier preparation and the last batch's later completion may lie outside it. Capture the whole workload when those phases are under investigation. Stacks and detailed tracing add overhead; keep capture duration proportionate to the question.

CUDA's [synchronization semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) depend on operation and memory kind. Nsight's [analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) describes the export and built-in reports. The sanitizer's rules are advisory interpretations of those activities.

## Inspect Python/native call edges

Standard cProfile captures some Python/native calls and callbacks. Select the native function names to inspect; the analyzer matches builtin labels, with no assumptions about installation paths.

```bash
.venv/bin/python -m cProfile -o "$CALLS" "$WORKLOAD"
.venv/bin/python -m uniserve.sanitizer calls "$CALLS" \
  --native _uniserve_ipc
```

The report shows repeated edges in both directions, call counts and inclusive time. Use it to locate repeated metadata conversion or per-row callbacks for further investigation. A repeated batch numerical call is expected. Inclusive time includes the callee's work and must not be summed or interpreted as FFI overhead.

cProfile supplies no temporal ordering or batch attribution. Its capture does not automatically cover native worker threads; many C API attribute accesses and extension constructors are absent. The report lists the selected native functions actually captured and reports an unknown coverage result when none match. Use this evidence alongside the CUDA timeline, not as a complete crossing count.
