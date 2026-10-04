# Execution

UniServe schedules requests in Rust and executes numerical calls through a Python and PyTorch backend. Models expose computational capabilities through ordinary modules; the worker binds their attention, matrix multiplication and expert layers to execution resources. Request identities, KV allocations, sampling state, streams and CUDA graphs belong to the engine and worker.

The worker's native `Executor` owns bounded batch admission, dependency ordering, collective launch order, completion, and result delivery. Direct Python callers and the rank service use the same submission path. Capacity remains occupied until the result is consumed, and a full queue leaves the submitted batch number available for retry. Independent requests can advance while another waits for storage; ranks sharing collective computation preserve launch order and wait for prior result delivery before starting the next batch.

`BatchRunner` supplies tensor preparation, numerical execution, output materialization, and physical resource retirement through the executor's backend interface. Those operations run on the worker thread. Dependency callbacks only notify a native `Submission` and wake the service; repeated or late notifications cannot launch a batch twice. Python batch resources contain no duplicate launch, completion, or delivery state. The native executor and request pool have no Python dependency; the production binding currently retains Python numerical and resource objects.

The worker's native `RequestPool` binds each admitted request epoch to the engine-assigned slot. It tracks pending calls, accepts progress without allowing late results to roll it back, and rejects stale epochs after slot reuse. A `Request` owns immutable admission parameters and native progress values; result observers can retain its epoch after the slot is reused. Python numerical views retain diffusion tensors and staging, which the executor drains before retirement. Direct Python callers and serving share the same request implementation. The request core is independent of Python and is also used by the TVM-FFI experiment.

The native `HostLane` supplies bounded threads for image preparation, media encoding and KV import copies. Capacity is reserved before inputs exist. A `HostTask` carries its action, result and input lease; deferred actions submit once their inputs are CPU-readable. Dependencies, results and physical completion share native waits, which release the Python GIL. Numerical callbacks retain ordinary Python and PyTorch code.

Cancelling an unsubmitted host task returns its capacity. Cancelling queued work resolves its result immediately but retains capacity and inputs until a thread dequeues it. Running actions finish normally. Input leases release only after the producer completes successfully; failed or cancelled producers cannot establish safe reuse. Result callbacks run outside native locks after capacity returns. Closing cancels unsubmitted tasks before joining workers, so submitted dependents can finish; abort stops admission without waiting for potentially stalled device work.

The native `BufferPool` backs scheduler-placed products with fixed device arenas. It rejects overlapping physical ranges and accepts release only for the current binding issued by that pool. Ranks that hold a subset of the logical products can use compact physical placement. Tensor views retain the arena backing; storage owners must retire device accesses and transport readers before releasing a binding for reuse.

The native `TensorStore` owns values that pass between calls. A `Buffer` becomes readable after production and result commit; CUDA consumers wait on its producer fence. A `TensorRead` retains the acquired tensor view through consumer completion; logical release rejects new readers but preserves existing reads. Physical reuse waits for local readers, transfer tickets, remote readers, and producer and consumer CUDA events. Reclamation queries completion without synchronizing unrelated streams.

An import shares resident coverage and fetches only missing regions into the reserved destination. Readers of an earlier shard keep that view while `TensorImport` coordinates the transfers needed by full-tensor consumers. Failed or cancelled imports retain their destinations until physical transfer completion. Request relays use bounded scalar lanes with fixed addresses; the store admits another call into a lane only after every value in that lane retires.

The native `LatentPool` owns diffusion trajectories in two fixed page banks. Preparation writes the inactive bank; applying a `LatentUpdate` advances the committed generation and switches banks. Imports reserve their destination pages before transfer and become visible after producer fences are ordered on the consuming stream. Exported bank versions remain immutable until every reader retires, and cancellation retains occupied pages through physical completion. Python supplies allocation, staging, gather and scatter operations over this backing.

Transport backends share a native `TransferCapacity` for bytes and read credits. A `ReadReservation` takes an entire fetch's credits before any read starts. The native `TransferPool` orders copies after both the producer fence and the submitting thread's destination stream. Its `TransferTicket` exposes consumable views before physical completion, so consumers can enqueue work behind the read fence. Cancellation revokes consumption; credit returns only after device access drains. Borrowed local views retain their source grant through every consumer stream. A failure to establish physical completion keeps the affected resources and credits occupied and surfaces through the pool and storage owners.

Local, shared-memory and CUDA VMM transports retain their source buffers in the native `BufferRegistry`. Releasing a buffer revokes further reads; its retirement waits for producer completion and outstanding readers. Backend callbacks perform the physical reclamation outside the registry lock, allowing completion observers to submit more work. CUDA VMM separates the original source from its exported pool chunk: the source retires after its copy finishes, while remote acknowledgment words govern chunk reuse.

## Token and canvas calls

Prefill, token decode and token denoising use separate homogeneous numerical calls. A prefill can write context without projecting logits or sampling. Readout canvases attend to that context without changing its KV entries; their candidate probabilities are normalized over the full vocabulary.

Block-diffusion generation retains each request's canvas tokens, stopping history and self-conditioning embeddings in its request slot. Each scheduled denoising call advances one step. A completed block publishes its tokens together, and a causal prefill commits those tokens to context before the next block starts. EOS, token limits and stop strings apply to the delivered prefix of the block.

The engine may queue a successor before its predecessor completes. Device continuation predicates make successors of a stopped canvas no-ops. The worker validates block and step progression before execution, and completion is accepted only for the current request epoch and submitted call. Cancellation retains request resources until outstanding readers retire; only then can another request reuse the slot.

## Prepared execution

Startup prepares bounded input storage and captures the configured prefill, decode, canvas and image-denoising shapes. Each call stages its own inputs into the retained buffers and selects a graph that holds its rows. Padding carries no request progress. Once prefill, decode or canvas graph preparation is sealed, a call outside its captured capacity is rejected. Graphs, staging buffers and captured numerical plans are owned by their execution context.

Canvas generation captures the numerical pass, vocabulary projection, sampler and state commit together. Readout graphs evaluate the shared canvas prefix and gather only the answer positions for the final projection. A readout tail that exchanges tokens with expert peers executes once at the agreed exchange capacity, keeping all peers on the same collective sequence.

The worker logs `uniserve-kernel-table` JSON after startup and when serving resolves additional call sites. Each record names the computation, layer paths, representation and selected provider. These records describe the executed configuration; they are not performance measurements.

The scheduler's allocator does not advise its heap for transparent huge pages. Worker launches default `MIMALLOC_ALLOW_THP` to `0` and propagate an explicit operator value to local and remote ranks. This keeps automatic heap collapse outside the default serving allocation policy while preserving operator control of worker allocation.

## Replicas and experts

`--data-parallel-size` starts independent replicas, each with its own scheduler, request storage and KV pool. `--worker-ranks` gives the tensor-parallel rank count per replica when placement is not supplied with `--workers`. Requests go to a live replica with the fewest in-flight requests; ties rotate. Cancellation remains routed to the replica that accepted the request.

`--expert-parallel` partitions routed experts across one-rank replicas. `--expert-exchange` selects `alltoall`, `megamoe` or `dwdp`; the available representations and resource requirements follow the [expert providers](experts.md). Collective expert execution agrees capacities and layer participation across replicas, including replicas without local request rows. DWDP accesses distributed weights while retaining independent request progress.

Prefix caching is enabled by default. `--disable-prefix-cache` disables cross-request prefix reuse and retention; request-local KV allocation and retirement still follow the [cache contract](cache.md).
