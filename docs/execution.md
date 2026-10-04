# Execution

UniServe schedules requests in Rust and executes numerical calls through a Python and PyTorch backend. Models expose computational capabilities through ordinary modules; the worker binds their attention, matrix multiplication and expert layers to execution resources. Request identities, KV allocations, sampling state, streams and CUDA graphs belong to the engine and worker.

The worker's native `RequestPool` binds each admitted request epoch to the engine-assigned slot. It tracks pending calls, accepts progress without allowing late results to roll it back, and rejects stale epochs after slot reuse. A `Request` retains numerical state for its observers until the executor drains its work and the pool retires it. Direct Python callers and serving share this lifecycle through the same native extension.

The native `BufferPool` backs scheduler-placed products with fixed device arenas. It rejects overlapping physical ranges and accepts release only for the current binding issued by that pool. Ranks that hold a subset of the logical products can use compact physical placement. Tensor views retain the arena backing; storage owners must retire device accesses and transport readers before releasing a binding for reuse.

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
