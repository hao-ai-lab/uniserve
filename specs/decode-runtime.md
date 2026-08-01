# Transactional asynchronous generation runtime RFC

Status: Implementation-ready
NOTE: EVERYTHING SHOULD BE IMPLEMENTED BEFORE MARKED AS DONE.

## Normative language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY define requirements. A requirement applies to every configured model route, sampling mode, transport, and topology unless a narrower capability is named explicitly.

## Abstract

UniServe uses one bounded operation protocol for autoregressive text generation, multimodal understanding, speculative decoding, image generation, generated-image feedback, KV publication, and repeated Und/Gen interleave. Model execution, device-side selection, semantic commit, and public delivery advance independently. Device work is connected by stream dependencies, host work is connected by futures, and steady-state serving never blocks a host thread to observe device progress.

The protocol has four cross-layer data records: `Operation`, `VersionRef`, `ProductRef`, and `CompletionRecord`. A small `Control` command enum carries ordered commit, close, and release decisions. Readiness, ownership, access hazards, resolved state, committed state, and published state are fields in existing scheduler and worker tables; they are not parallel object hierarchies.

Sampling and speculative acceptance resolve on device. A successor may consume a generation-tagged device product before the host observes it, provided all memory bounds are known and a device predicate makes invalid work a safe no-op. CPU grammar, stop-string processing, encoding, and transport remain asynchronous continuations: they may suspend one request lineage but never block the scheduler or an unrelated request.

A scheduler batch may contain Und and Gen partitions. Each partition is physically submitted under a declared execution capability and has independent readiness and accounting. Domain-homogeneous submission is universally valid. A route may use tensorized mixed submission only after proving that Gen rows cannot alter Und structure, state, or publication.

## Background

The serving decode path this protocol governs is host-paced today: a worker call returns after its sampled tokens are host-visible, the scheduler interprets each token before constructing the successor operation, and stop, grammar, Gen-trigger, and KV-publication decisions ride on that per-token host round trip. The evidence below establishes that this loop, not model compute, bounds serving throughput, and it fixes the constraints the protocol is designed against.

### Fixed per-step cost bounds decode

A concurrency-1 measurement on the SenseNova-U1 route records 8.67 ms per decode token with the model forward accounting for 0.38 ms of it, and per-token cost within 3% of its floor from 64 SMs to the full 152-SM device ([`green-context-sm-allocation.md`](green-context-sm-allocation.md)). The per-token bound is therefore per-step scheduling, launch, synchronization, and completion overhead rather than compute or memory bandwidth, and its relative weight grows as the model shrinks: an 8B text tower does not yield a proportional throughput advantage over Qwen3-32B.

The current I2T operating points are 1698.3, 1728.8, and 1751.9 aggregate output token/s at [`r4`](../artifacts/benchmark/sensenova_u1_i2t/sensenova_beans_i2t/uniserve_r4/summary.md), [`r8`](../artifacts/benchmark/sensenova_u1_i2t/sensenova_beans_i2t/uniserve_r8/summary.md), and [`r16`](../artifacts/benchmark/sensenova_u1_i2t/sensenova_beans_i2t/uniserve_r16/summary.md). These are non-streaming responses with natural-EOS lengths, so they carry no TTFT or TPOT samples and are not model-efficiency comparisons against fixed-work ShareGPT text points.

### Identical work separates on runtime structure alone

Two Qwen3-32B ShareGPT `r8` runs execute the same 200-prompt workload on the same model, precision, and hardware and emit essentially identical output tokens (≈43.3k):

| Execution structure | Output tok/s | Mean TPOT | Mean TTFT | Artifact |
| --- | ---: | ---: | ---: | --- |
| Host-paced per-token loop | 689.4 | 110.6 ms | 674.2 ms | [summary](../artifacts/benchmark/decode_runtime/qwen3_sharegpt/uniserve_r8/summary.md) |
| Device continuation for the eligible greedy subset | 1345.2 | 17.1 ms | 89.4 ms | [summary](../artifacts/benchmark/runtime_qualification/qwen3_sharegpt/uniserve_r8/summary.md) |

A 1.95x aggregate and 6.5x TPOT separation under identical emitted work is attributable to execution, completion, and scheduling structure, not to model kernels. At the measured 23.98 time-weighted active users, the 1345.2 aggregate corresponds to 56.1 output token/s per user.

### Continuation currently covers only a deterministic greedy subset

The scheduler queues a same-request successor against worker-local sampled-token state only when committing one token cannot change host control flow (`can_queue_decode_successor` and `device_token_relay_eligible` in `crates/engine/scheduler/src/scheduler.rs`): replayable non-Gen text with no grammar, no stop strings or stop-token IDs, temperature at most zero, zero min-tokens, no logprobs, no bad-word or allowed-token constraints, neutral penalties, no logit bias, and no token-conditional Gen trigger policy. The common worker sampler writes token and finish products on device for greedy, stochastic, constrained, penalty, and logprob-bearing rows and stages bounded completion metadata and logprob values through query-only pinned copies without synchronously extracting a live device scalar. Scheduler-level continuation eligibility remains narrower than worker sampling capability, so stochastic, Gen, logprob, grammar, and publication-bearing requests still wait for host semantic resolution before the scheduler issues a successor.

### Pipelining and burst loops are not continuity

Worker call pipelining overlaps batches from different requests; it does not shorten one request's causal chain when the successor is constructed only after host interpretation of the predecessor, so device bubbles reappear at low concurrency, long sequences, and interleave. Amortizing the fixed cost with a worker-internal forward-sample loop reduces launch overhead but keeps sampling synchronous, hides per-operation completion from the host, and cannot place stop, grammar, cancellation, or Gen-transition decisions at exact points. The protocol therefore relays generation-tagged device products with per-operation completions instead of batching host round trips.

### Interleave serializes twice

A `gen_output` request is excluded from same-request continuation entirely, and a token-conditional publication policy disables worker-side token deferral, so Und decode ahead of a Gen transition observes every token on the host even though almost no token triggers the transition. KV publication snapshots the full committed prefix per layer on every publication, retains only the newest snapshot, and the same-node transport copies device memory to the host synchronously on the publishing thread (`publish` in `uniserve_worker/runtime/kv_store.py`, `ShmTransport.publish` in `uniserve_worker/runtime/transfer.py`). No destination watermark, incremental suffix, or producer-readiness event exists, and the host token read doubles as the publication's producer-completion barrier. Publication is therefore specified here as an independently fenced non-state operation with declared extents rather than as a deletable host read.

### Mixed batches couple lifecycles before they contend for SMs

On the interleave workload, 96–97% of flow operations share a forward with text rows; packed and exclusive admission are indistinguishable on aggregate throughput at every measured concurrency, packed token gaps inherit the denoise cadence, exclusive admission thins text batches and worsens per-request TPOT tails, and packed attention is measured not to be the dominant cost of sharing ([`mixed-batch-scheduling.md`](mixed-batch-scheduling.md)). The coupling that costs is lifecycle coupling: one batch-scoped completion, sampling groups that finalize together, and shared failure and backpressure. This protocol therefore gives Und and Gen independent completion, accounting, failure, and backpressure while retaining shared admission, and treats residual co-residency interference as a measured quantity; SM partitioning allocates compute only and cannot remove per-step host cost.

### One protocol instead of per-workload fast paths

Separate fast paths for greedy, stochastic, speculative, Gen, and distributed KV each carry a private lifecycle and cannot compose. The records and invariants in this RFC exist so the capabilities combine: stochastic sampling with speculative acceptance, token streaming with incremental KV publication, interleave with cancellation, distributed transport with exact versioning, and mixed admission with independent completion.

## Scope

This RFC defines:

- The identity, dependency, readiness, commit, publication, and reclamation protocol for generation work.
- Zero-blocking sampling for greedy and stochastic modes.
- Device-continuous speculative decoding and bounded ordinary decoding.
- Exact handling of grammar, stop tokens, stop strings, logprobs, cancellation, and output ordering.
- KV versioning, incremental publication, replica installation, and transfer lifetime.
- Gen transition, flow execution, artifact materialization, and repeated Und/Gen interleave.
- Client-observed TTFT, TPOT, image-latency, and modality-transition-latency semantics for interleaved output.
- Mixed-service admission, physical execution capabilities, and interference accounting.
- Single-process, tensor-parallel, staged, and disaggregated execution.

This RFC does not define Green Context sizing or SM placement; the SM-budget measurements for that decision are recorded in [`green-context-sm-allocation.md`](green-context-sm-allocation.md). The runtime exposes the measurements and partition boundaries required to evaluate that architecture separately.

The construction checkpoint sequence, staged cutover, validation economy, correctness qualification, and performance-protection gates for this runtime are defined in [`decode-runtime-construction.md`](decode-runtime-construction.md).

The Rust host engine ingress value, generate gateway surface, and `WorkerTopology` deployment language are defined in [`engine-surface.md`](engine-surface.md). The HTTP and model-tokenize funnel that produces the submit unit are defined in [`serving-surface.md`](serving-surface.md).

## Required outcomes

- No blocking host/device observation in the steady-state path for greedy sampling, stochastic sampling, logprobs, speculative decoding, grammar, stop strings, KV publication, Gen, feedback, or interleave.
- Exact protocol, state-transition, constraint, RNG-coordinate, cache-effect, cancellation, tensor-shape, and output-ordering equivalence to a serial depth-one oracle under the same model, precision, sampling parameters, processor order, seed, cache state, and image-quality controls; generated payloads satisfy the applicable numerical and quality contracts.
- Continuous same-request device submission whenever successor inputs and allocation bounds are device-representable.
- Bounded memory and work under speculation, cancellation, slow clients, slow CPU processors, transfer backpressure, and mixed workloads.
- Deterministic commit identity, ordering, lineage, and constraint enforcement independent of batching, completion order, execution-window depth, and interleave timing; legal free-running token and image payloads are not required to be bit-identical across execution geometries.
- Independent Und and Gen completion, accounting, backpressure, and publication even when both domains are admitted in one scheduler batch.
- UEval exposes comparable latency distributions for first text, token progress, decoded images, and every visible text/image boundary.
- End-to-end decode performance in the reference-parity band while preserving correctness and workload semantics.
- One protocol serves greedy, stochastic, speculative, logprob, Gen, and interleave execution; no workload has a private fast path, and continuous-submission eligibility is a declared route capability.

## Feasibility basis

The design requires bounded worker call pipelining, CUDA streams and events, pinned host memory, preallocated device and completion arenas, paged KV storage, asynchronous transfer tickets, host futures, and event-driven scheduler wakeups. These mechanisms are available in the deployed PyTorch and CUDA stack; no persistent scheduling kernel, CUDA host callback, or synchronous transport fallback is required.

### Reference mechanisms in SGLang

The local SGLang tree at commit `c6a7c98ae429760ed3b2df8d3a11600c3855d74a` provides non-normative implementation evidence.

| Evidence in `refs/sglang` | Requirement adopted by UniServe |
| --- | --- |
| `FutureMap` relays sampled tokens, sequence lengths, draft probabilities, top-k state, and recurrent hidden state through request-slot-indexed device buffers. | Successors gather generation-tagged device products without extracting host scalars. |
| EAGLE verification computes acceptance on device and uses it for accepted-path compaction, KV movement, bonus-token selection, and draft extension. | Acceptance, selected prefix, logical length, next token, and KV visibility are outputs of one device-resolved operation. |
| DFlash publishes structural sequence state before independent draft-KV maintenance completes. | An operation may expose independently fenced products as soon as each product is valid. |
| CUDA graph runners publish read-done events after shared inputs have been snapshotted. | Mutable shared tables track exact producer and reader events rather than relying on whole-forward lifetime guesses. |
| Pinned D2H arenas and copy streams decouple completion copying from model compute. | Completion observation uses event queries and never synchronizes a compute stream. |
| Request accounting separates allocated KV from committed KV. | Reserved, initialized, visible, committed, and replicated KV extents are distinct fields. |
| Decoupled speculative messages identify an authoritative base, require contiguous commits, and let close dominate queued work. | Distributed products name an exact base version and digest; commit is contiguous and close has an exact cutoff. |

Relevant evidence is concentrated in `python/sglang/srt/managers/overlap_utils.py`, `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`, `python/sglang/srt/speculative/dflash_worker_v2.py`, `python/sglang/srt/speculative/triton_ops/dflash.py`, `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`, and `python/sglang/srt/speculative/decoupled_spec_io.py`.

The useful mechanism is per-operation, generation-tagged, independently fenced dataflow. UniServe implements that mechanism within its scheduler, worker, sampler, KV, transfer, and frontend ownership boundaries.

The reference establishes feasibility of device token relay, device-resident acceptance and compaction, independently fenced structural products, and generation-aware distributed commit. It is not the zero-blocking contract for UniServe: reference paths that require CPU sequence lengths still copy them through pinned memory and wait for the private copy stream, while the GPU-only backend path avoids that boundary. UniServe therefore qualifies device-actual sequence lengths for every performance-critical backend rather than inheriting a synchronized CPU-mirror path.

## Protocol budget

The cross-layer data protocol contains exactly four stable records:

1. `Operation` describes bounded work and its dependencies.
2. `VersionRef` identifies an exact request state point.
3. `ProductRef` identifies a bounded generation-tagged value; the worker-owned product entry supplies its physical storage and readiness event.
4. `CompletionRecord` is the fixed-layout host observation of an operation.

The request-runtime command channel contains one `Control` enum with `commit`, `close`, and `release` variants. Administrative worker commands such as health, model lifecycle, and cache reset remain outside the request-runtime protocol defined here.

Nested tuples and enums inside these records are serialization fields, not independently owned runtime abstractions. Existing batch and transport envelopes frame lists of these records but carry no semantic lineage or lifetime state. Scheduler request state, worker session state, KV entries, transfer entries, and output journals remain tables owned by their existing modules.

The following facts are fields rather than standalone classes:

- A transaction is one operation row plus its reserved store entries.
- An operation's output declaration is its `outputs` field.
- Resolved, committed, and published positions are cursors in the scheduler request row.
- A readiness fence is an event identifier stored in a product or resource entry.
- Resource access ownership is a logical-reference count plus producer and reader event identifiers in the resource entry.
- Backpressure state is one per-request credit vector whose dimensions are resource kinds.
- A state family is the point-indexed output range of one operation.
- Mixed execution is a capability value on a batch partition.
- CPU processor state is scheduler-local state addressed by request and semantic point.

## Identities

`RequestKey` is `(authority_id, session_id, epoch)`. The epoch changes whenever an admitted request identity is reused.

`OpId` is unique within one scheduler authority lifetime. Every operation and product reference includes `RequestKey`, so an `OpId` cannot alias across requests or epochs.

A state point is `(producer_op_id, point_index)`. For a token-producing operation, point `0` names the exact parent and points `1..K` name successive output prefixes. For a Gen operation, points name declared flow or transition boundaries. An operation that does not advance request state produces products rooted at its parent point.

The protocol uses exactly two digests. The operation plan digest names what an operation was registered to run: it covers immutable registration fields, including the exact identity of any device-resolved parent products. The semantic digest names what the operation's selected result was: it covers the fixed parent semantic digest, the operation plan digest, the selected point, and the semantic output delta. The worker computes the semantic digest on the host from a ready pinned completion record; device continuation relies on exact producer, point-product, output-generation, and plan-digest identity and does not run a digest algorithm on the accelerator.

Scalar lengths, batch slots, page indices, and sequence numbers are accounting fields and MUST NOT serve as state identity.

## Four protocol records

### `Operation`

```text
Operation {
  request_key
  op_id
  parent: VersionRef
  work
  route
  domain
  advances_state
  bounds {
    max_points
    max_tokens
    max_kv_pages
    max_latent_bytes
    max_completion_bytes
    max_transfer_bytes
  }
  inputs: [ProductRef]
  outputs: [ProductRef]
  predicate: none | ProductRef
  rng: none | (seed, semantic_index_base, draw_layout)
  control_seq
  plan_digest
}
```

`Operation` is immutable after registration. The scheduler allocates its identity, bounds, parent, output references, control sequence, and credits. The worker atomically binds every declared output reference and one completion entry to physical storage before enqueueing device work.

Every output reference has the operation's request key and operation ID, a unique output index, a generation acquired from product credit state, and bounds no larger than the operation's declared resource maxima.

`work` is one closed nested variant:

| Work variant | State effect | Required role |
| --- | --- | --- |
| `Token(Extend)` | Advances the authoritative lineage by a bounded prompt, feedback, or forced-token span. | Prefill, generated-artifact feedback, and host-resolved processor input. |
| `Token(Decode)` | Advances the authoritative lineage by one bounded sampled point. | Ordinary greedy or stochastic decoding. |
| `Token(Verify)` | Advances the authoritative lineage by one selected point from a bounded draft tree. | Target speculative verification, bonus selection, and accepted target-KV visibility. |
| `Draft` | Produces auxiliary branch products without advancing the authoritative lineage. | Draft tokens, probabilities, hidden state, and draft-KV maintenance. |
| `Encode(Vision)` | Produces an immutable auxiliary product without advancing the authoritative lineage. | Input-image vision features for I2T, interleave, or feedback ingestion. |
| `Encode(Latent)` | Produces an immutable auxiliary product without advancing the authoritative lineage. | Input-image latent features required by a route before vision ingestion. |
| `Transfer(Product)` | Does not advance state. | Bounded stage-to-stage or worker-to-worker movement of an immutable declared product. |
| `Transfer(KvPublish)` | Does not advance state. | Incremental publication of a fixed committed local KV view for Gen conditioning or a destination replica. |
| `Transfer(KvInstall)` | Does not advance state. | Installation of an immutable artifact or remote KV product at an exact fixed base. |
| `Gen(Transition)` | Advances the authoritative lineage to a bounded Gen state. | Conditioning validation, latent allocation, and schedule initialization. |
| `Gen(Flow)` | Advances the authoritative lineage by a bounded denoise quantum. | Version-owned latent updates and flow progress. |
| `Materialize` | Does not advance state. | VAE decode, device-to-host image staging, encoding input, and artifact publication. |

Sampling is device postprocessing within `Token(Extend)`, `Token(Decode)`, and `Token(Verify)` rather than an independently scheduled work kind. An `Encode` operation produces an immutable product; a following `Token(Extend)` consumes that product at an exact parent and is the only operation that advances text state for image ingestion. Generated-artifact feedback follows the same pair. These variants are serialization fields within `Operation`, not separately owned runtime records or class hierarchies.

`predicate` is a device boolean or bounded mask derived from already registered device products. A false predicate makes the operation a side-effect-free no-op that still produces a valid completion status. Predicated work cannot mutate committed state, shared KV visibility, public output, committed RNG coordinates, or reusable storage. A cancellation received after device submission may allow already enqueued branch-local work to finish; exact close dominance prevents that work from becoming committed, published, cached, or remotely visible.

### `VersionRef`

```text
VersionRef {
  request_key
  producer_op_id
  point:
    fixed(point_index, semantic_digest)
    | device(selected_point: ProductRef, producer_plan_digest)
}
```

`VersionRef` names one exact state point. A device form allows a successor to consume a selected speculative prefix or Gen transition before host observation. Any component that requires a fixed host identity waits asynchronously for the corresponding completion and then records the fixed form.

The admission root is a fixed `VersionRef` whose producer is the request-admission operation and whose point is zero.

### `ProductRef`

```text
ProductRef {
  request_key
  producer_op_id
  output_index
  generation
  kind
  storage_class
  dtype
  shape_bound
  point_range
}
```

`ProductRef` never contains an unbounded shape. The actual logical length is either host-static or another device product with a declared maximum.

The scheduler assigns the output index and logical generation from an acquired product credit, so a successor can name the product before its producer resolves. The worker product table binds the reference to `(store, slot, physical_generation, byte_offset, producer_event)` during atomic registration. The logical generation protects protocol-reference reuse; the independently checked physical generation protects slot reuse. Physical slot identities, tensors, and CUDA event objects remain worker-local.

The worker store's entry APIs check the generation on registration, lookup, stream-wait insertion, completion packing, transfer submission, and reclamation; no other module compares generations. A stale generation is a protocol fault.

The producer records the table entry's producer event after every write that makes the product valid. A consumer resolves the reference, inserts a stream wait on that event, and records its own read-done event in the same store entry.

### `CompletionRecord`

```text
CompletionRecord {
  request_key
  op_id
  completion_slot_generation
  status
  selected_point
  logical_lengths
  token_span
  finish_flags
  product_generations
  semantic_digest
  error_code
  timing_counters
}
```

The device-written portion of the record layout is fixed for a route capability and copied asynchronously into pinned host memory. Variable-size values stay in bounded arenas and are addressed by slot, generation, offset, and length. After the copy event is query-ready, the worker validates the pinned fields and computes the canonical semantic digest on the host before emitting the `CompletionRecord`; this computation neither observes a live device value nor delays device successors.

The host reads a completion record only after querying its copy event successfully. Completion observation does not imply semantic commit, public delivery, or physical reclamation.

### `Control`

```text
Control =
  commit {
    request_key
    control_seq
    expected_parent
    selected: fixed VersionRef
    public_event_limit
    disposition
  }
  | close {
    request_key
    control_seq
    cutoff: fixed VersionRef
    reason
  }
  | release {
    request_key
    op_id
  }
```

Controls are idempotent by `(request_key, control_seq, variant, canonical_content_digest)`. The worker applies them in per-request sequence order. Duplicate controls with equal identity and content have no effect; identity reuse with different content is a protocol fault.

`close` dominates every uncommitted descendant beyond its cutoff. `release` drops the scheduler's logical ownership but does not override unfinished producer or reader events.

## Owned runtime state

### Scheduler request row

The scheduler's existing request/session state stores:

- Request identity, route, domain phase, and admission limits.
- The latest device-resolved `VersionRef`.
- The latest fixed semantically committed `VersionRef`.
- The latest public event sequence and byte offset.
- Registered operations and ordered controls.
- CPU grammar, decoder, stop-string, and custom-processor states.
- RNG coordinate allocation and token-processor deltas.
- Output journal entries.
- The request credit vector.
- Cancellation cutoff, terminal reason, and progress timestamps.

The scheduler is the sole writer of semantic commit order and public event order.

### Worker session row

The worker's existing session state stores:

- Registered operation rows and their output entries.
- Device-resolved selected points and product references.
- Execution and completion slot generations.
- KV mappings and their extents.
- Latent and artifact storage entries.
- Producer events, reader events, and logical-reference counts.
- Applied control sequence, terminal cutoff, and reclaimable generations.

The worker is the sole writer of device product readiness and physical reclamation.

### Store entry fields

Every reusable execution slot, completion slot, device product, KV allocation, latent, staging range, and transfer buffer records:

```text
owner_request
owner_op
generation
logical_references
producer_event
reader_events
state
capacity
actual_extent_if_observed
```

A store entry is reclaimable only when logical references are zero, its producer event is complete, every reader event is complete, and no registered transfer can still access it. Reclamation is discovered by event query; it never waits.

The worker store retains strong references to every Python tensor, view, staging allocation, and event object until the entry is reclaimable. Language-level reference loss cannot authorize physical reuse.

A mutable shared table records a reader event for each consuming stream. When a graph or kernel snapshots the table into call-local storage, the reader event is recorded immediately after that snapshot; otherwise it follows the final read. Writers wait in-stream on those reader events rather than extending lifetime to an unrelated whole-forward barrier.

## Core invariants

1. A request epoch has exactly one semantically committed state lineage.
2. Every state-advancing operation names one exact parent version.
3. State points may resolve out of order globally but become semantic only in ancestor and control-sequence order for one request.
4. A non-state operation cannot move the resolved or committed cursor.
5. Public events are committed in event-sequence order and are rooted at a semantically committed state point.
6. Every device-accessible dependency has a generation-tagged product reference and a producer event.
7. No storage generation is reused before its producer and all registered readers complete.
8. No unresolved CPU semantic decision is bypassed by semantic commit.
9. Cancellation and terminal close have an exact cutoff and dominate every uncommitted descendant beyond it.
10. RNG coordinates depend on request identity and semantic sample coordinates, not batch position, launch order, or speculative outcome.
11. Prefix cache and remote publication expose semantically committed versions only.
12. Resource exhaustion returns backpressure without partial registration, synchronous reclamation, or unbounded allocation.
13. Every physical batch partition has independent completion, error, work, and performance accounting.
14. Duplicate protocol delivery is idempotent; conflicting reuse of an identity is contained as a protocol fault.

## Invariant enforcement

Each invariant is enforced by a named chokepoint API, a closed type, or a static check rather than by per-module convention. Code that reaches protocol state only through these chokepoints cannot violate the corresponding invariant silently; a violation surfaces as a deterministic protocol fault, a rejected registration, or a build failure.

| Enforcement point | Owner | Enforced invariants |
| --- | --- | --- |
| Operation planning and credit reservation | Scheduler | Exact parentage (2), RNG coordinate assignment (10), and atomic reservation with backpressure (12). |
| Ordered control emission | Scheduler commit path | Single committed lineage (1), ancestor and control-sequence order (3), cursor immobility for non-state operations (4), public event order (5), CPU semantic gating (8), and close-cutoff dominance (9). |
| Atomic registration transaction | Worker executor | Generation-tagged references with recorded producer events (6), no partial side effect on rejection (12), and duplicate idempotency with identity-conflict containment (14). |
| Store entry APIs | Worker stores | Generation validation on every access (6) and producer- and reader-event-safe reuse (7). |
| Submission boundary | Executor | Stream waits and reader-event registration for every input product, so consuming code cannot bypass invariants 6 and 7. |
| Commit-rooted publication | Worker KV store and frontend output journal | Committed-only prefix-cache and remote visibility (11) and committed-root ordered public delivery (5). |
| Partitioned batch structure | Scheduler batch and executor | Independent per-partition completion, error, work, and performance accounting (13). |
| Closed protocol types | Protocol crates, FlatBuffers schema, and Python decoding | Exhaustive `work` and `Control` variants and bounded record layouts at compile and decode time. |
| Forbidden-synchronization checks | Static analysis and runtime traces | The steady-state zero-blocking contract on every request-path call site. |

Model, kernel, and frontend code interacts with the protocol only through these enforcement points; the invariant list specifies the chokepoints and is not a review checklist applied to every module.

## Steady-state zero-blocking contract

The steady-state interval starts when the first request operation is submitted and ends when the terminal public event is committed. During that interval, scheduler, executor, worker controller, model driver, sampler, serializer, grammar runtime, KV publication, transfer, tracing, and metrics code MUST NOT block a host thread on device progress.

Forbidden operations in the interval include:

- `cudaDeviceSynchronize`, `cudaStreamSynchronize`, and `cudaEventSynchronize`.
- `torch.cuda.synchronize`, `Event.synchronize`, and equivalent accelerator waits.
- `.item()`, `.tolist()`, `.numpy()`, or synchronous `.cpu()` on a device tensor used by request progress.
- Device-to-host copy into pageable storage or any copy whose readiness is not represented by a queryable event.
- Reusing a slot, page, product, latent, replica range, or staging range by waiting for a prior user.
- Waiting for the oldest submitted completion while another completion or request is runnable.
- Synchronous transport reads, writes, registration refresh, or remote status waits.
- Logging or metric extraction that materializes a device scalar on the request thread.

Permitted actions include:

- Querying an event, transfer ticket, or CPU future without waiting.
- Enqueueing a stream wait on a producer event.
- Enqueueing asynchronous H2D, D2H, D2D, peer, IPC, or RDMA work.
- Reading pinned host storage after its event query reports completion.
- Parking the scheduler on its combined command, completion, CPU-future, transfer, timer, and worker-death wake set when no work is runnable.

Initialization, explicit administrative export, controlled shutdown, and fatal device recovery may synchronize outside the steady-state interval.

## Operation lifecycle

### 1. Plan

The scheduler selects an exact parent, assigns RNG coordinates and control sequence, computes hard upper bounds, and verifies route capabilities. Planning reads host-owned state only.

### 2. Reserve

The scheduler atomically reserves logical credits for one operation or one bounded registration window. The worker independently performs one atomic physical reservation and registration transaction for the submitted window. A worker rejection leaves no registered product or storage entry; the scheduler releases its logical reservation and leaves request state unchanged. No cross-process atomic-memory primitive or distributed two-phase commit is required.

### 3. Register

The worker validates the predeclared product references, binds each one to a fresh physical slot under its declared generation, allocates the completion entry, and records producer-event identities before any operation becomes runnable. A transport acknowledgment reports only whether the whole registration transaction became visible; it is envelope metadata rather than a semantic runtime record.

### 4. Submit

The executor inserts stream waits for input product events, registers reader events for shared inputs, and enqueues the route's device work. The call may use host-known maxima while device products carry actual lengths and predicates.

### 5. Resolve

The route's existing device postprocessing stage computes selected tokens or points, logical extents, finish flags, token-state deltas, KV visibility, transition flags, and digests. Each product event is recorded immediately after that product becomes valid.

### 6. Continue

A successor may be planned from a device `VersionRef` when its inputs, bounds, route, and predicate are representable without host observation. The successor waits on producer events in its streams and may execute before the parent's completion reaches the host.

### 7. Observe

The completion stream packs the fixed completion record, enqueues D2H copy to pinned memory, and records a copy event. The progress engine queries all pending completion events and dispatches every ready record; it does not impose global FIFO order.

### 8. Decide

The scheduler validates identity, parent, digest, operation status, CPU semantic controls, and cancellation cutoff. It emits an ordered `commit` or `close` control.

### 9. Publish

The frontend commits output journal entries in per-request sequence order after their state root is semantic. A slow transport consumes output credits and cannot retain model execution slots.

### 10. Retire

The scheduler emits `release` when no logical successor, CPU task, output record, or transfer requires the operation. The worker reclaims each resource independently when its recorded events and references permit.

No lifecycle step waits synchronously for another step.

## Prefix-addressable state

A token operation with capacity `K` exposes point `0` as its parent and points `1..K` as successive token prefixes. The device-selected point is a bounded scalar product. Token spans, penalty deltas, KV visible extents, and digests are indexed by point or can be reconstructed from bounded prefix deltas.

Speculative acceptance selects one point from the verifier operation. Stop-string resolution may later select an earlier point from the still-provisional suffix. Cancellation selects the latest permissible fixed point. These decisions use the same `VersionRef` and commit protocol.

Token extent and target-KV extent are independent point fields. A decode point may contain a newly selected semantic token whose KV row is produced by its successor; a verifier point may contain an accepted bonus token whose target KV is likewise not initialized yet. Attention, publication, prefix caching, and Gen conditioning use the declared KV extent rather than inferring it from token count.

An operation output becomes reachable atomically: every declared product reference, physical store generation, and event identity is registered before the operation is visible to consumers. Independently ready products may then transition to ready in any order.

## Execution window

The scheduler maintains a bounded per-request window of registered operations. Window extension is permitted when:

- The parent is fixed or device-representable.
- Every dynamic input is a registered `ProductRef`.
- Every actual-dependent kernel supports a device actual or a host-known safe bound.
- A device predicate makes invalid, terminal, cancelled, or rerouted descendants side-effect free.
- Required credits are already reserved.
- No unresolved CPU decision is required to construct the next operation.

The route declares a maximum safe depth derived from completion slots, product slots, KV reservations, graph slots, and rollback capacity. Admission uses that bound; runtime pressure may reduce depth to one without changing semantics.

Greedy and stochastic decoding use the same window mechanism. Pipeline depth is a bounded scheduling policy, not a sampling-mode branch.

The existing batch envelope may carry a topologically ordered registration window containing multiple operations from one request, and a later envelope may name outputs declared by an earlier registered envelope. The worker acknowledges atomic registration before the scheduler may reference that envelope from a later submission, registers every declared reference before scheduling dependency-ready physical partitions, and emits one `CompletionRecord` per operation. Transport framing therefore creates neither an unresolved-reference race nor a whole-batch completion barrier.

### Shapes and CUDA graphs

A route capability states, for every dynamic input, whether the backend consumes a device actual, a device mask or predicate within a fixed host bound, or a host-fixed shape. Continuous submission is allowed only for the first two forms. A host-fixed actual suspends that request until its completion becomes observable, without blocking a host thread.

The route capability table also fixes maximum window depth, maximum speculative points, attention sequence-length ownership, append-offset ownership, graph predicate support, sampler processor coverage, tensor-parallel selection ownership, Gen conditioning requirements, and tensorized mixed support. Startup rejects a performance-critical route whose declared backend cannot consume device sequence lengths and device append offsets for the operations that require continuous submission. Capability discovery does not occur on the request path.

Graph identity includes route, domain, attention regime, tensor-parallel topology, dtype, bounded shape bucket, and output layout. Captured graphs read stable arena addresses and device actuals or predicates. Every graph input and output slot is an ordinary generation-protected store entry with producer and reader events.

Graph capture and eager execution produce the same products and completion layout. A route declares graph qualification at startup; request progress never performs an opportunistic capture or a synchronous shape probe.

## Completion and progress

Execution storage and host-observation storage are separate. A model call releases its execution slot after all device consumers of call-local storage have recorded read completion. Its completion slot remains live until the pinned record has been observed and released.

The progress engine maintains ready queues for:

- Worker completion events.
- CPU grammar and decoder futures.
- Transfer tickets.
- Output transport capacity.
- Timers and cancellation.
- Worker and peer health.

One scheduler tick drains a bounded number from each ready queue, applies ordered controls, reclaims query-ready resources, and schedules runnable work. This prevents completion storms from starving admission or Gen quanta.

## Sampling

### Device sampler

The common sampler runs greedy, temperature, top-k, top-p, min-p, typical, repetition, frequency, presence, bad-word, forced-token, and logprob processing through the same device-resident interface. A route may support a subset only if admission rejects unsupported parameters before execution.

Processor order is fixed by the public sampling contract and is shared by ordinary and speculative paths. Invalid distributions, NaNs, all-masked rows, and unsupported parameter combinations produce deterministic error completion records.

The sampler writes selected token, selected probability, requested logprobs, finish candidate, token-state delta, and next-sampling metadata into bounded products. It does not materialize a device scalar on the host.

### RNG

Each random draw is addressed by `(request_key, semantic_token_index, processor_stage, draw_index)`. Batching, execution-window depth, accepted speculative length, replay, and completion order do not alter coordinates. The common sampler maps these coordinates to a counter-based Philox value and performs deterministic inverse-CDF selection over the post-processor distribution. It does not use batch-order-dependent generator advancement or `torch.multinomial`.

Speculative proposal randomness and target sampling randomness occupy disjoint declared coordinate spaces. Rejected proposals do not consume target coordinates for later semantic tokens.

### Token processor state

Frequency, presence, repetition, bad-word, and forced-token state is stored as a committed base plus per-operation bounded deltas. A provisional branch sees its ancestors' deltas; a sibling does not. Commit folds the selected prefix delta into the base, and release discards unselected deltas after their reader events complete.

Requested logprobs stay in a device product until completion packing. Output formatting reads the pinned copy asynchronously.

## Speculative decoding

One verifier operation consumes an exact base `VersionRef`, bounded draft products, target KV inputs, and deterministic RNG coordinates. Its device postprocessing computes:

- Accepted prefix length.
- Selected verifier point.
- Accepted token span and optional bonus token.
- Target and draft logical lengths.
- KV visible extent and compaction plan.
- Next draft inputs.
- Token-processor delta.
- Finish and invalidation predicates.
- Completion digest.

Structural products needed by the next verifier or draft operation receive an early producer event. Maintenance products such as compacted draft KV receive their own later event. A successor waits only on the products it reads.

KV capacity for the maximum verified span is reserved before submission. Accepted visibility is a device-selected extent within that reservation. Unaccepted bytes never become semantic, enter prefix cache, or become eligible for remote publication.

The worker may register bounded descendants from the device-selected point. A CPU grammar or stop-string decision can retract provisional descendants to a fixed prefix; predicates and exact parentage prevent those descendants from becoming semantic.

Proposal depth adapts from observed acceptance, memory pressure, and service latency only at operation boundaries. Adaptation changes work selection, not RNG coordinates or correctness.

## CPU semantic continuations

Arbitrary CPU grammar matchers, incremental detokenizers, stop-string automata, custom processors, PNG encoding, and transport serialization execute in bounded futures. Their input names a fixed state point or a pinned completion span.

A request requiring a CPU-produced mask cannot submit the dependent sampler until that future is ready. The request is suspended in the scheduler's future wait set; no host thread waits and unrelated requests continue.

CPU state is immutable by semantic point. A task publishes a successor state only if its input point remains on the selected lineage. Cancellation marks queued tasks irrelevant; already running tasks may finish, but their output cannot commit after the cutoff.

### Grammar

GPU-compatible grammar masks may be represented as device products and preserve device continuation. A CPU-only matcher produces a bounded host mask, enqueues asynchronous H2D copy, and exposes a product event. Admission bounds matcher state and pending tasks.

### Stop tokens and EOS

Stop-token and EOS decisions are device outputs and may select a prefix point immediately. Public delivery still observes ordered commit.

### Stop strings

Incremental detokenization and stop-string matching operate on pinned token spans. The output journal retains enough provisional bytes and token-to-byte boundaries to retract any suffix within the declared stop window. Semantic commit cannot advance beyond the earliest unresolved stop-string boundary.

### Custom processors

A custom CPU processor must declare snapshotability, deterministic inputs, output bounds, and maximum outstanding tasks. A processor that cannot satisfy the declaration serializes only its request at the processor boundary and does not weaken the zero-blocking contract.

## KV state and publication

The KV store keys each logical view by a fixed or provisional `VersionRef`. Each entry records:

- Reserved page range.
- Initialized extent.
- Device-visible extent.
- Semantically committed extent.
- Published extent by destination.
- Page mapping generation.
- Quantization scale identity and extent.
- Producer and reader events.
- Logical references and replica state.

Reserved capacity is not initialized data. Initialized data is not necessarily visible to attention. Visible data is not necessarily semantic. Semantic data is not necessarily replicated.

Append operations reserve worst-case pages and write under copy-on-write mappings. A device-selected point determines visible extent. Commit fixes the semantic extent. Unselected pages remain unreachable and are reclaimed after all device readers finish.

Prefix-cache installation requires a fixed committed version, exact model and cache-layout identity, exact quantization metadata, and a digest. Cache hits produce an exact `VersionRef`, never a scalar-length approximation.

KV publication is a non-state-advancing operation rooted at a fixed committed version. It selects the missing committed extent for one destination, acquires source and destination entries, enqueues transfer, and installs the destination mapping only after the transfer ticket and digest validation succeed.

The concrete execution algebra separates local publication from remote installation. `Transfer(KvPublish)` carries no source product and publishes the committed local text-KV view needed by a Gen transition. `Transfer(KvInstall)` carries an immutable source product and installs generated-image feedback into text KV. Token sampling policy controls only token selection; publication is never conditional work inside `Extend`, `Decode`, or `Verify`.

Every Gen route declares its conditioning form and exact required extent. When the fixed trigger version contains a semantic tail whose KV row is not initialized, the scheduler first issues a bounded `Token(Extend)` closure operation that consumes the already selected tail, emits no additional public token, and produces a fixed version with the required KV extent. `Transfer(KvPublish)` accepts only a fixed version whose initialized and committed extents cover the declared conditioning extent. A route whose transition consumes an independently fenced conditioning product names that product directly and still validates the exact fixed trigger version before semantic commit.

Publication is incremental. A destination reports its exact installed base version and extents. A later operation transfers only compatible committed suffixes. A mapping or scale change creates new immutable destination entries and cannot mutate data visible to an active consumer.

Transport submission, completion, retry policy, and peer health are asynchronous. An unavailable required transport capability blocks route admission; serving does not fall back to synchronous copying.

## Gen and repeated interleave

Gen transition is a state-advancing operation whose device products include transition eligibility, exact conditioning version, latent allocation, flow schedule identity, and output bounds.

Each flow quantum is an operation rooted at the latest Gen version. The quantum has a declared maximum step count and wall-time target so the scheduler can alternate Gen and Und service. Latent updates are version-owned and cannot overwrite a generation still read by another quantum, materializer, or feedback operation.

Artifact materialization is a non-state-advancing operation rooted at a fixed Gen point. Device decode, D2H staging, image encoding, and public delivery have independent readiness. Text or further flow work may continue while materialization progresses when the route permits it.

Generated-image feedback installs an immutable artifact product, computes the declared visual representation, and creates the next Und operation from an exact parent version. The artifact, derived visual tokens, KV extension, and later Gen transition all use the same operation lifecycle.

Repeated Und/Gen travel therefore has one lineage, one ordered commit cursor, and domain-tagged operations. The existing output journal records an event sequence, modality, semantic root, and server public-commit timestamp for every visible text or image event. Transition count, image step count, artifact count, text token count, and every phase duration are measured as work invariants.

## Batch and mixed execution

The existing scheduler batch carries one or more partitions. A partition contains operations with one execution domain, route capability, shape class, attention regime, and independent completion range.

Domain-homogeneous physical submission is the universal correctness capability. A route that declares tensorized mixed submission combines independently described partitions only at the final runner boundary; scheduler lineage, allocation, readiness, completion, failure, and accounting remain partitioned.

A route may declare tensorized mixed submission when a conformance proof establishes all of the following:

- Und attention masks, sequence lengths, position IDs, KV mappings, graph keys, collective order, and sampler rows are identical to domain-homogeneous execution.
- Gen rows cannot change an Und row's shape, padding, kernel choice, graph capture, RNG coordinate, KV visibility, or completion event.
- Every domain has independent output slices, product events, completion records, failures, and work counters.
- Batch permutation tests reproduce the serial oracle's protocol, state, shape, constraint, RNG-coordinate, and numerical-conformance properties across all supported mixtures and shapes.
- Interference measurements meet the declared service envelope.

SenseNova production interleave requires the tensorized mixed capability because its parity target includes shared-batch execution. A failed tensorized-mixed qualification requires an explicit resource-partitioning architecture decision; it cannot be resolved by silently substituting domain-homogeneous calls in a passing candidate. Routes that do not declare tensorized mixed submission use domain-homogeneous physical calls while preserving joint admission, fairness, and accounting. The capability is route-static and does not introduce a per-request compatibility path.

### Service policy

The scheduler assigns bounded quanta to text decode, image flow, transfer, materialization, and CPU continuation. Admission controls bound memory and outstanding operations per domain. Progress uses age, deadline, and work-normalized deficit so large Gen quanta cannot starve text decode and dense text batches cannot indefinitely postpone Gen.

Mixed-service telemetry attributes queue time, launch time, device time, completion-copy time, commit delay, and public delay to the operation and domain. It also records co-resident domain work so Und/Gen interference can be validated rather than inferred.

## Tensor parallel and distributed execution

The scheduler assigns one operation identity and collective sequence to all tensor-parallel ranks. Every rank validates request key, operation identity, parent, shape bounds, and collective order before enqueue.

Sampling uses one declared ownership strategy: either a designated rank owns selection and publishes products to peers, or a deterministic sharded sampler produces an identical selected point on all ranks. The strategy is fixed per route capability.

Each rank maintains independent producer and reader events for its storage. An operation completion is observable only after every required rank reports the same identity, selected point, and digest or a contained error.

Staged tower execution connects stages with `ProductRef`-equivalent transport entries carrying exact identity, bounds, and readiness. Host orchestration never reads a device scalar to choose the next stage.

Disaggregated draft, verify, KV, or Gen workers exchange exact base `VersionRef`, digest, contiguous extent, operation identity, and generation-tagged transfer entries. A receiver rejects gaps, stale epochs, conflicting digests, and descendants beyond a close cutoff.

## Preemption, replay, and failure

Preemption stops future admission for a request, records an exact fixed checkpoint, closes uncommitted descendants, and releases resources through normal event-driven reclamation. It never forces device synchronization.

A replayable checkpoint contains committed tokens, RNG coordinates, processor state, exact KV identity and extents, Gen latent identity when the route declares snapshot support, output journal position, and control sequence. State that cannot be captured exactly makes the route non-preemptible at that point.

Replay verifies the checkpoint digest and resumes from the same semantic coordinates. It cannot use approximate latent reconstruction, scalar-length parent matching, or a different sampling order.

A request-local protocol, processor, capacity, or validation fault emits an error completion and closes that request at its latest safe point. A worker-fatal device error fails every request whose authoritative state cannot be proven, rebuilds worker-owned stores outside steady state, and requires a new epoch for replay.

Transfer failure leaves the source version authoritative and the destination entry uninstalled. CPU-future timeout closes only the affected request. A cancellation race is resolved by control sequence and exact cutoff.

## Resource bounds and backpressure

Admission reserves one finite credit vector per request. Its dimensions are:

- Registered operations.
- Execution and completion slots.
- Device products.
- KV pages and rollback deltas.
- Latent and artifact bytes.
- Pinned completion and staging bytes.
- Transfer bytes and tickets.
- CPU tasks.
- Output journal bytes.

Every route declares per-request and worker-wide maxima for each dimension. The credit vector is acquired before operation registration and released by logical release plus event-safe physical reclamation; no resource kind has an independent reservation protocol.

Credit exhaustion returns `WouldBlock`, reduces execution-window depth, or pauses admission. It cannot invoke a blocking wait, reuse a live generation, drop a semantic result, or grow an unbounded queue.

Slow clients consume output credits but do not retain model execution storage. Slow CPU continuations consume CPU-task and provisional-lineage credits. Slow transfers consume transfer credits and do not mutate source ownership.

## Observability

Every operation emits host timestamps for planned, registered, submitted, completion-ready, completion-observed, semantically committed, publicly committed, and reclaimed transitions. Device event timing is sampled asynchronously and never read on the request critical path.

Required counters include:

- Registered and in-flight operations by route and domain.
- Effective execution-window depth.
- Device-continuation ratio.
- Completion-ready to observed delay.
- Observed to semantic-commit delay.
- Semantic to public-commit delay.
- Product, KV, latent, completion, staging, transfer, CPU-task, and output credit pressure.
- Predicated no-op and retracted work.
- Speculative proposed, accepted, and committed tokens.
- KV reserved, initialized, visible, committed, published, and transferred extents.
- Und and Gen queue, launch, device, completion, and commit time.
- Gen steps, images, transitions, text tokens, and artifact materializations.
- Client-observed TTFT, TPOT, image latency, and transition latency, including overall and directional transition distributions.
- Transition timestamp coverage, per-request modality-segment signature, and text-to-image and image-to-text sample counts.
- Forbidden synchronization detections.
- Per-GPU and per-user output throughput.

Trace identity is `(request_key, op_id, control_seq, domain)`. Metrics code reads pinned completion fields or asynchronous device timing results only after readiness.

## Implementation ownership

The design extends existing ownership areas and does not require a parallel runtime hierarchy.

| Existing owner and placement | Responsibility |
| --- | --- |
| Worker wire in `crates/protocol/worker-wire`, IPC schema in `crates/protocol/worker-ipc-core/schema/worker.fbs`, Python decoding in `uniserve_worker/batch.py`, and the worker interface in `uniserve_worker/worker/protocol.py` | The four records, controls, validation, serialization, and typed worker boundary. |
| Scheduler state in `crates/engine/scheduler/src/scheduler.rs` and `crates/engine/scheduler/src/generation` | Operation planning, exact parentage, credits, ordered controls, CPU futures, fairness, and public ordering. |
| Scheduler resources, grammar, and logits in `crates/engine/scheduler/src/resources.rs`, `grammar.rs`, and `logits.rs` | Bounded credit state, CPU continuation state, and processor-plan construction. |
| Executor boundary in `crates/engine/executor` and worker IPC in `crates/engine/worker-ipc` | Batch partitions, nonblocking submit and any-ready completion, topology agreement, and control delivery. |
| Worker execution in `uniserve_worker/execution/executor.py` and session state in `uniserve_worker/runtime/request_session.py` | Atomic registration, stream dependencies, operation submission, device resolution, completion packing, and event-safe retirement. |
| Sampling in `uniserve_kernel/python/uniserve_kernel/sampling.py` and logits preparation in `uniserve_worker/nn/logits.py` | Device sampling, RNG coordinates, processor deltas, logprobs, and finish candidates. |
| Product, KV, and latent stores in `uniserve_worker/runtime/product_store.py`, `kv_store.py`, `kv_pool.py`, and `latent_store.py`, with scheduler KV ownership in `crates/engine/kv` | Version-keyed storage, physical generation binding, extents, copy-on-write append, semantic visibility, and cache installation. |
| Transfer and staging in `uniserve_worker/runtime/transfer.py` and `host_staging.py` | Registered ranges, asynchronous tickets, destination installation, pinned completion copies, and generation checks. |
| SenseNova route and common runner in `uniserve_worker/models/sensenova/model.py` and `uniserve_worker/execution/model_runner.py` | Transition products, flow quanta, latent updates, materialization, and feedback products. |
| Frontend serving output in `crates/frontend/serving/src/text/output` and `crates/frontend/serving/src/chat/output` | Incremental decode, stop-string state, encoding, output journal, transport credits, and public commit. |
| Scheduler trace and benchmark harness in `crates/engine/scheduler/src/trace.rs` and `uniserve_eval` | Operation lifecycle, domain work, visible-modality boundary timestamps, TTFT, TPOT, image latency, transition latency, zero-sync evidence, and performance-gate artifacts. |

The four protocol records are plain data carriers. Readiness queries, credit accounting, store reclamation, and ordered commit are methods on their owning modules rather than independently instantiated managers.

### Role protocol surface

Each contribution role uses a bounded protocol surface, and the owning runtime layers apply the rest of the protocol on its behalf.

| Role | Protocol surface used directly | Applied by the runtime on its behalf |
| --- | --- | --- |
| Model route author | `work` variant semantics, declared input and output products, and the route capability declaration. | Generation checks, stream waits, event recording, registration, credits, and controls. |
| Kernel and sampler author | Device tensor layouts, bounded shape contracts, and RNG coordinate mapping. | Protocol records, events, and lifetime state. |
| Scheduler contributor | `Operation`, `VersionRef`, `Control`, the credit vector, and request-row cursors. | Physical slots, CUDA events, and device storage binding. |
| Worker runtime contributor | Store-entry fields, generations, producer and reader events, registration, and reclamation. | Semantic commit order and public event order. |
| Frontend contributor | Output journal entries, per-request commit order, and output credits. | Device state, product lifetime, and KV state. |
| Evaluator contributor | Client-visible events, artifacts, and metric definitions. | Every runtime record and owned table. |

## Risks and containment

| Risk | Containment |
| --- | --- |
| Device continuation consumes excessive memory. | Hard operation, product, completion, KV, and rollback credits reduce the window before admission pressure becomes unsafe. |
| Generation reuse exposes stale data. | Every lookup and transfer validates request, producer, slot, and generation; physical reuse waits for producer and reader events. |
| CPU grammar or stop processing limits one request. | Future-based suspension, bounded provisional horizon, immutable point state, and continued service for unrelated requests. |
| Speculative rollback corrupts token or KV state. | Prefix-addressable points, branch-local deltas, exact parent digests, and commit-only cache visibility. |
| Cancellation races with completion or publication. | Monotonic control sequence, exact cutoff, ordered public journal, and close dominance. |
| Mixed Und/Gen execution changes Und structure. | Domain-homogeneous default and a route-static tensorized capability proof with independent completion and permutation tests. |
| Gen monopolizes the device. | Bounded flow quanta, work-normalized deficit, age and deadline input, and per-domain admission. |
| Completion processing becomes a scheduler bottleneck. | Any-ready queries, bounded drain budgets, separate arenas, and lifecycle delay metrics. |
| Distributed peers disagree on state. | Exact base version, digest, contiguous extent, epoch validation, and all-rank completion agreement. |

## Acceptance

The runtime is accepted when:

- The four-record protocol and owned table fields cover every configured generation route without a parallel transaction or lifecycle hierarchy.
- Every correctness and protocol property defined in [`decode-runtime-construction.md`](decode-runtime-construction.md) passes for supported sampling, speculation, KV, Gen, interleave, cancellation, topology, and failure combinations.
- Runtime evidence shows zero steady-state blocking host/device observation.
- Eligible routes demonstrate same-request continuation through device products.
- CPU-only continuations suspend one lineage without blocking scheduler progress.
- Storage remains bounded and generation-safe under delayed completion, cancellation, slow clients, slow CPU work, transfer backpressure, and mixed service.

Checkpoint eligibility, performance protection, and benchmark acceptance for each construction stage are governed by [`decode-runtime-construction.md`](decode-runtime-construction.md).
