# Transactional asynchronous generation runtime specification

## Status and scope

This specification defines UniServe generation execution from scheduler registration through worker reclamation for text, multimodal understanding, image generation, generated-image feedback, colocated transfer, and repeated Und/Gen interleave. The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative.

The configured public and model capability surface is defined by [`serving-surface.md`](serving-surface.md). Construction boundaries are defined by [`decode-runtime-construction.md`](decode-runtime-construction.md). Qualification is defined by [`generation-runtime-qualification.md`](generation-runtime-qualification.md).

## Required outcomes

The runtime MUST provide:

1. One immutable operation algebra from scheduler registration through physical execution.
2. Semantic identity independent of physical allocation, batching, execution depth, completion order, and topology scheduling.
3. Query-only steady-state progress with no host thread blocked on accelerator or transport readiness.
4. Bounded same-request continuation before host observation whenever successor inputs and predicates are device-representable.
5. Exact depth-one semantics at every advertised execution-window depth.
6. Independent Und and Gen completion, backpressure, failure containment, accounting, and publication.
7. Fixed resource bounds under delayed completion, cancellation, slow clients, slow CPU continuations, transfer pressure, and mixed service.
8. Complete lifecycle, continuation, resource, domain, synchronization, and public-transition observability.

## Protocol budget

Cross-layer semantic execution uses exactly four data records and one three-variant control record:

- `Operation`
- `VersionRef`
- `ProductRef`
- `CompletionRecord`
- `Control::{Commit, Close, Release}`

Batch, partition, transport, and physical-call envelopes MAY group registered operations and physical resources, but MUST NOT introduce independent work, lineage, transaction, output, control, or lifetime authority.

`Operation` is the sole registered semantic work row. Before registration the scheduler MAY use an ephemeral builder. Registration consumes that builder and stores one immutable `Operation` plus one scheduler-owned apply record keyed by `(request_key, op_id)`. The apply record contains only semantic-commit effects, completion validation expectations, public visibility policy, and resource release policy; it MUST NOT duplicate work, parent, route, domain, bounds, products, predicate, RNG, control sequence, or digests.

The worker derives physical calls directly from registered operations and model row declarations. A physical call MAY contain row references, tensor contexts, graph keys, device and weight bindings, streams, and staging leases. It MUST reference registered operation identities, derive output bindings from `Operation.outputs` and row ABI, and contain no semantic base-version or transaction authority.

## Semantic identities

The semantic identity set is:

- `RequestKey`: stable request identity across frontend, scheduler, worker, evaluator, and topology ranks.
- `OpId`: stable operation identity within the request.
- Exact point identity: a fixed `VersionRef` or a generation-tagged device-selected `VersionRef`.
- Plan digest: canonical digest of immutable registration facts.
- Semantic digest: canonical digest of the selected point and semantic effect.

Scalar lengths, list positions, batch slots, page numbers, completion slots, transport tickets, event addresses, stream IDs, and worker-local transaction counters MUST NOT serve as semantic identity.

The plan digest covers, in canonical order:

1. Request key and operation ID.
2. Exact parent version.
3. Work variant, route, domain, and state effect.
4. Every finite work and resource bound.
5. Ordered input and output product references.
6. Predicate reference and predicate sense.
7. RNG coordinates and draw layout.
8. Control sequence.

The plan digest excludes physical KV page IDs, storage slots, events, streams, batch position, physical partition, transport allocation, and topology scheduling order. Rust, FlatBuffers, Python, simulator, and worker registration MUST use one canonical field table and byte encoding. Digest tests MUST permute batching, allocation, depth, topology submission, and completion order.

## `Operation`

An `Operation` contains:

```text
request_key
op_id
parent: VersionRef
work: Work
route
domain
state_effect
bounds
inputs: ordered ProductRef[]
outputs: ordered ProductRef[]
predicate
rng
control_seq
plan_digest
```

Every bound is finite and validated before registration. Bounds cover points, tokens, append extent, KV reservation, latent storage, artifact storage, completion bytes, pinned bytes, transfer bytes, CPU tasks, and output-journal bytes applicable to the work leaf.

Registration is atomic across the request session and every required store. A failed reservation or registration publishes no partial operation and restores all reserved capacity. The operation becomes immutable at successful registration.

### Closed work algebra

| Work leaf | Semantic role |
| --- | --- |
| `Token(Extend)` | Consume one or more known tokens and advance text/KV state within declared bounds. |
| `Token(Decode)` | Produce one bounded sampled text point and its continuation products. |
| `Token(Verify)` | Select an authoritative point from a bounded target verification tree. |
| `Token(Draft)` | Produce a bounded proposal tree when a production device drafter is configured. |
| `Encode(Vision)` | Produce immutable vision features without advancing the authoritative text lineage. |
| `Encode(Latent)` | Produce immutable latent or conditioning features without advancing the authoritative text lineage. |
| `Transfer(Product)` | Move an exact bounded immutable product between configured local stages or ranks. |
| `Transfer(KvPublish)` | Publish an exact committed KV suffix rooted at a fixed semantic point. |
| `Transfer(KvInstall)` | Install an exact compatible KV suffix at a configured consumer. |
| `Gen(Transition)` | Establish exact conditioning and the first version-owned Gen state. |
| `Gen(Flow)` | Advance a bounded version-owned Gen quantum. |
| `Materialize` | Decode a terminal Gen product into an immutable artifact product. |

Every configured leaf MUST have a validated depth-one physical route. `Token(Draft)` and `Token(Verify)` remain closed protocol and worker leaves but are not configured or advertised until a production drafter route declares the complete contract.

### State effects

Each work leaf declares exactly one fixed state effect. A state effect identifies which authoritative version can advance and which outputs remain auxiliary. A predicate-false operation MUST produce a successful side-effect-free completion with zero semantic advance and no mutation of committed state, shared visibility, public output, RNG coordinates, or reusable storage.

## `VersionRef`

`VersionRef` names one exact state point:

```text
Fixed(request_key, semantic_point, semantic_digest)
Device(product_ref, selected_index, generation)
```

The fixed form is host-resolved and immutable. The device form names a bounded selected point in a generation-tagged product. A successor MAY consume the device form before host observation after waiting on its producer event on the consumer stream. Conversion to fixed identity occurs only after asynchronous completion readiness and generation validation.

Every device form declares its maximum number of candidate points and exact digest layout. An out-of-range selected index, stale generation, owner mismatch, or digest mismatch is a request-local protocol fault and cannot mutate committed state.

## `ProductRef`

`ProductRef` contains store kind, slot, generation, owner request, producer operation, capacity, and product-kind identity. It is registered before producer resolution and can be bound as an input of a registered successor.

Every reusable product entry records:

- Owner request and producer operation.
- Generation and logical reference count.
- Capacity and observed extent.
- Producer event and every reader event.
- Entry state and exact product kind.
- Fixed or device-selected semantic root where applicable.
- Model, layout, dtype, quantization, scale, and topology identity required for compatibility.

Producer and consumer streams use event dependencies rather than host waits. Reuse requires producer completion, all reader events complete, logical references released, generation validation, and store transition through the canonical reclamation state.

## `CompletionRecord`

`CompletionRecord` uses a fixed device layout with request key, operation ID, generation, status, predicate outcome, selected point, observed extents, semantic digest inputs, finish candidates, logprob span, work counters, and bounded product metadata required by the operation.

The worker writes completion data on device, issues an asynchronous copy into registered pinned storage, records a completion-copy event, and returns control. Progress code queries the event. Only after readiness does it validate generation and identity, compute any host digest, and publish a partition report.

Completion observation MUST NOT allocate pageable progress buffers, perform synchronous device copies, extract live scalars, or wait on an event. A malformed extent or identity is handled before semantic commit and leaves the latest fixed point authoritative.

## `Control`

Controls are:

```text
Commit(request_key, control_seq, op_id, selected_point, semantic_digest)
Close(request_key, control_seq, cutoff_point, reason_digest)
Release(request_key, op_id)
```

Commit and close use ordered identity `(request_key, control_seq, variant)`. Equal identity with equal canonical content is idempotent; equal identity with different content is a protocol fault. The worker applies commit and close in per-request sequence order.

Release uses identity `(request_key, op_id, release)` outside the semantic control sequence. It is idempotent, cannot advance lineage, cannot supersede close, and cannot physically reclaim an entry until all producer and reader events and logical references permit reclamation.

Close defines an exact fixed cutoff. Every descendant beyond that cutoff is semantically dominated even if registered or submitted. Completion, commit, and release processing MUST preserve this rule under reordering and duplication.

## Owned runtime state

### Scheduler request row

The scheduler request row is the sole owner of semantic commit order and public event order. It contains fixed semantic and public cursors, registered operation identities, scheduler apply records, ordered control sequence, capability binding, logical credit ledger, CPU-continuation futures, stop-string decoder state, output journal, cancellation cutoff, and committed sampling checkpoint references.

The scheduler MUST NOT own physical KV pages, worker store slots, CUDA events, or device readiness. It reserves logical capacity through the negotiated credit vector.

### Worker session row

The worker session row is the sole owner of physical readiness and reclamation. It contains registered operations, physical store bindings, physical KV page mappings, device products, completion storage, route state, control replay state, producer and reader events, and worker-local failure state.

Physical KV page allocation occurs atomically inside worker registration after logical scheduler reservation. `Operation` carries logical KV bounds and exact parent/product identity, not physical page IDs.

### Stores

Completion, product, KV, latent, artifact, staging, transfer, and snapshot stores share the entry lifecycle and generation discipline defined by `ProductRef`. Specialized stores MAY extend metadata but MUST NOT redefine ownership, selected-point identity, or reclamation semantics.

## Core invariants

1. An operation is immutable after registration.
2. A semantic point has exactly one request lineage and canonical digest.
3. Only the scheduler commits semantics and public order.
4. Only worker sessions and stores determine device readiness and physical reuse.
5. A false predicate is a semantic no-op.
6. Close dominates every descendant beyond its fixed cutoff.
7. Physical allocation never contributes to semantic identity.
8. Every consumer waits on the exact producer event and contributes a reader event before reuse.
9. Completion observation is query-only until its copy event reports readiness.
10. RNG coordinates depend only on request key and semantic sampling coordinates.
11. Resource reservation precedes registration and registration precedes submission.
12. Semantic commit validates plan, point, extent, generation, and digest against the scheduler apply record.
13. Public commit is ordered after semantic commit and preserves the output journal cursor.
14. Release is asynchronous, idempotent, and subordinate to event-safe reclamation.

## Generalized bounded request window

Each route declares a maximum unresolved window and an exact feature bitset for which that depth is legal. Depth one is the serial oracle. Higher depth uses a bounded state graph:

1. An operation emits exact selected-point, continuation predicate, finish candidate, next input, sampling-state delta, logical extent, and domain-transition products.
2. A successor consumes those products with `VersionRef::Device` and `ProductRef` before host observation.
3. Every provisional descendant has an exact predicate, parent, finite resource reservation, and close cutoff.
4. Credits can reduce the active window to any smaller depth without changing semantic coordinates.
5. Host observation fixes semantic and public cursors but is not required to launch a device-representable successor.
6. CPU semantic work bounds the provisional horizon for only the affected lineage.

Successor eligibility MUST be derived from the exact route capability, available products, operation bounds, and remaining credits. It MUST NOT be a mode-specific greedy gate or a separate Gen loop.

Stochastic sampling, penalties, requested logprobs, minimum-token floors, allowed-token masks, bad-word state, forced-token state, and device-representable finish predicates use the same window. Stop strings and configured CPU processors MAY cap the provisional horizon; registered descendants remain retractable through exact points and predicates.

## Device-resident sampling state

The common sampler operates on a committed base plus bounded ancestral deltas. Each sampled or verified point emits a generation-tagged `SamplingState` delta product containing:

- Token-count updates used by repetition, frequency, and presence penalties.
- Bad-word automaton state and allowed-token or forced-token transition state.
- RNG next-coordinate metadata.
- Finish candidates and minimum-token state.
- Logprob references required by descendants and completion packing.
- The exact ancestor reference and selected semantic point.

A successor resolves the visible state from the committed base and its ancestral delta chain on device. Sibling deltas remain isolated. Commit folds the selected ancestral delta into the worker-resident base; release discards unselected deltas after all readers complete. Scheduler replay stores fixed committed checkpoints and MUST NOT reconstruct live operation input from host token history.

The processor order is fixed:

1. Base logits and model-specific numerical normalization.
2. Static allowed-token and forced-token constraints.
3. Bad-word transition mask.
4. Minimum-token EOS/stop mask.
5. Repetition, frequency, and presence penalties.
6. Logit bias.
7. Temperature.
8. Top-k, top-p, min-p, and typical filtering in declared order.
9. Distribution validation and inverse-CDF selection.
10. Requested logprob extraction and sampling-state delta construction.

Ordinary Decode and configured Verify use the same processor implementation and error semantics. NaNs, infinite invalid values, an all-masked distribution, malformed masks, and unsupported combinations produce deterministic request-local error completions.

### RNG mapping

Every random draw is addressed by `(request_key, semantic_token_index, processor_stage, draw_index, draw_layout)`. The canonical mapping MUST define exact Philox key and counter words, integer-to-uniform conversion, vocabulary traversal order, and inverse-CDF boundary behavior shared by the Rust simulator and Python/CUDA production implementation.

Batch order, execution depth, accepted proposal length, completion order, retry, and replay MUST NOT change a draw coordinate. Proposal and target randomness, when configured, use disjoint declared draw spaces; rejected proposals cannot advance target coordinates.

## CPU semantic continuations

CPU work is a bounded future keyed by request and exact point. The shared facility handles stop-string decoding, custom logit processors, image encoding, and administrative serialization. It reserves a CPU-task credit before launch, exposes query-only readiness, has a declared timeout and result bound, and suspends only dependent descendants of that request.

Scheduler progress MUST continue for unrelated requests while a CPU future is pending. A timeout or invalid result becomes a request-local failure at the latest fixed point. Split-token stop strings can retract provisional descendants to the exact accepted prefix without changing other requests or committed public order.

## Speculation boundary

`Token(Draft)` and `Token(Verify)` are unadvertised until a configured production drafter exists. Worker depth-one verification remains a protocol oracle.

A configured drafter MUST emit bounded device proposal products. Verify MUST consume the exact base, bounded proposals, target KV state, RNG coordinates, and branch-local sampling state. It MUST emit early structural products containing acceptance, selected point, accepted span, bonus token, logical extents, next proposal inputs, sampling-state delta, finish predicates, digest data, and KV visibility/compaction decisions. A successor Draft or Verify consumes those products before host observation.

Unaccepted KV bytes never become committed, cached, or published. Structural and maintenance products use independent readiness fences when their critical paths differ.

## KV state and publication

KV ownership tracks reserved, initialized, attention-visible, committed, and destination-published extents; physical page mapping generation; model and layout identity; quantization and scale identity; producer and reader events; logical references; and replica state.

Append reserves worst-case logical capacity before registration and worker physical capacity atomically during registration. Writes use copy-on-write where a live committed version could otherwise be overwritten. Completion selects the exact visible extent, commit fixes the semantic extent, and rejected or cancelled pages reclaim only after event safety.

Prefix-cache installation accepts only fixed committed roots with exact model, layout, quantization, scale, and digest identity. KV publication is incremental, immutable, asynchronously ticketed, rooted at a fixed committed point, and separate from sampling. KV install validates exact base, suffix continuity, mapping generation, and compatibility before visibility.

## Cross-modal continuation

The token sampler emits a device transition predicate for every configured direct or generated branch trigger. A predicated Gen Transition consumes the exact selected text version, conditioning product, and validated KV extent and produces device eligibility, conditioning identity, latent allocation, schedule identity, and finite bounds.

Gen Flow consumes version-owned latent products and step predicates in bounded service quanta. It cannot overwrite a live latent generation. Successors may be registered up to the route's unresolved-window bound. The terminal flow predicate gates Materialize.

Materialize produces an immutable artifact product with separate device-decode, D2H, CPU-encoding, and public-readiness events. Other eligible work proceeds while those stages resolve.

Feedback Encode and Token Extend consume the immutable artifact and exact selected parent. Resumed Decode or a later Gen Transition can be registered in the same bounded graph when products and CPU semantics permit. Host `steps_done`, decoded trigger tokens, and image bytes are observations and public-delivery inputs, not launch authority for device-representable edges.

Repeated Und/Gen travel uses one request lineage, one semantic cursor, one public cursor, and domain-tagged operation identities. Every transition, Gen update, artifact, and text token contributes exact work and timing counters.

## Batch and mixed execution

The scheduler groups registered operations into independently completed partitions by domain, route, shape, attention form, graph eligibility, topology, and sampling ownership. A batch envelope carries partition membership and physical launch metadata but no independent semantic lineage.

Und and Gen partitions have independent completion reports, error containment, backpressure, credit accounting, and public readiness. Tensorized mixed execution is advertised only for exact row combinations that pass homogeneous-versus-mixed structural, numerical, graph, collective, permutation, and interference qualification. Capability is never inferred from the existence of any mixed row declaration.

Service uses bounded domain quanta, request age, deadline contribution, and work-normalized fairness. The runtime records queue, launch, device, completion, semantic-commit, public-commit, and co-residency time by domain.

## Tensor parallel and local stages

All tensor-parallel ranks consume the same operation identity, plan digest, collective sequence, sampling owner, and point bounds. The declared sampling rank produces selected device products; peers receive them through device transport without host semantic selection. Completion joins identical selected points, extents, and semantic digests across ranks before scheduler commit.

Configured local stages transfer exact generation-tagged products with bounded asynchronous tickets. Transport validates producer, consumer, exact base, extent, epoch, and digest. Gap, stale epoch, duplicate conflict, digest conflict, and close-cutoff conflict are deterministic faults.

Cross-node disaggregated execution and request preemption are not configured. Capability declarations report them as unsupported. Snapshot export and restore are administrative operations outside steady-state request execution and do not create public request controls.

## Zero-blocking execution

The steady-state interval begins when the first request operation is submitted and ends when the terminal public event is committed and release is issued. During this interval, scheduler, executor, worker controller, model driver, attention, vision, sampler, completion, transport, tracing, and metrics code MUST use query-only progress.

Every request-path shape and bound is supplied from host-known registration metadata or remains device-resident. Vision grid bounds, cumulative sequence maximums, paged attention lengths, append offsets, page mappings, live sequence lengths, selected points, and Gen step predicates MUST NOT be read back synchronously.

Every steady-state transfer returns a bounded ticket. Request threads MUST NOT perform sleep polling, synchronous RDMA or shared-memory reads, synchronous publication, durable snapshot fallback, event synchronization, stream synchronization, device synchronization, live tensor scalar conversion, live device-to-host list conversion, blocking future results, or resource waits.

A repository gate roots static reachability at every configured frontend progress, scheduler, executor, worker execute, model forward, attention, vision, sampler, completion, transfer, tracing, and metrics entrypoint. It rejects forbidden operations unless a call site is proven to belong exclusively to startup, administrative export, or recovery outside steady state.

Runtime instrumentation records every device wait, scalar observation, synchronous transport operation, sleep poll, and blocking reclaim attempt with request and operation identity. Formal qualification requires zero detections.

## Exact capabilities

Each route capability declares:

- Supported work leaves and exact depth-one physical lowering.
- Maximum unresolved window and feature-specific legal depth.
- Maximum points, token work, KV extent, latent work, artifact bytes, and transfer bytes per operation.
- Device actual-length and append-offset support.
- Graph eligibility and attention backend predicates.
- Complete sampler processor bitset, processor order revision, RNG layouts, and sampling ownership.
- Tensor-parallel and local-stage topology with collective and transport identity.
- Gen conditioning form, exact extent requirements, latent and materialization bounds.
- Tensorized mixed row combinations and service envelope.
- Snapshot scope and preemptibility, which are unsupported for configured routes.
- Every credit maximum and store capacity binding.

Startup validates the capability digest across frontend, scheduler, worker, and evaluator and proves that every advertised operation lowers to one depth-one physical route. A fallback must be contained within the declaration and preserve numerical, zero-blocking, graph, and performance contracts.

## Resource bounds and backpressure

One finite logical credit vector covers:

1. Registered operations.
2. Execution slots.
3. Completion slots and pinned bytes.
4. Product entries and bytes.
5. KV pages and rollback bytes.
6. Latent entries and bytes.
7. Artifact entries and bytes.
8. Staging entries and bytes.
9. Transfer tickets and bytes.
10. CPU tasks and result bytes.
11. Output-journal bytes.
12. Route-specific bounded auxiliary state.

Negotiation binds each logical dimension to physical capacity across every configured rank and stage. Registration reserves the complete worst-case vector atomically. Credit exhaustion returns backpressure or reduces unresolved depth without blocking, partial registration, storage reuse, semantic loss, or unbounded queue growth.

Cancellation, failure, slow-client, slow-CPU, slow-transfer, and delayed-event paths use the same credits and reclamation lifecycle. The output journal is bounded independently of worker progress; a slow client cannot retain unlimited device state.

## Failure containment

Request-local protocol, processor, capacity, timeout, and output errors close the request at the latest provable fixed point. Worker-fatal errors invalidate affected physical sessions and require explicit process-level recovery. Peer and transport failures preserve exact epoch and product ambiguity rules.

No failure path may publish a partial registration, reuse a live generation, advance an unverified point, publish uncommitted KV, skip ordered close, or block unrelated request progress. Duplicate, stale, gap, digest, extent, and cutoff conflicts have deterministic classifications and trace records.

## Operation lifecycle and observability

One lifecycle trace keyed by `(request_key, op_id, control_seq, domain)` records:

1. Planned.
2. Logical resources reserved.
3. Worker registration complete.
4. Submitted.
5. Device execution started.
6. Producer ready.
7. Completion copy ready.
8. Completion observed.
9. Semantically committed.
10. Publicly committed.
11. Release issued.
12. Physically reclaimed.

Device timing is copied or queried only after asynchronous readiness. Timestamp order and conservation are validated against scheduler, worker, store, control, and frontend events.

Metrics expose:

- Maximum and effective unresolved window depth.
- Eligible and actual device continuations by route, work, sampling mode, and modality edge.
- Predicated, executed, retracted, and committed work.
- Every lifecycle delay distribution.
- Current, maximum, backpressured, and reclaimed values for every credit dimension and store generation.
- Predicate outcomes, sampling processor coverage, RNG layout, and requested logprob work.
- Speculative proposal, acceptance, and compaction work when configured.
- Reserved, initialized, attention-visible, committed, and published KV extents.
- Domain queue, launch, device, completion, commit, public, and co-residency time.
- Gen steps, materialization phases, artifacts, feedback edges, and transition coverage.
- Client TTFT, TPOT, image latency, overall transition latency, and directional transition latency distributions.
- Forbidden synchronization detections.
- Aggregate, per-GPU, and per-user throughput with exact work denominators.

The evaluator consumes these fields and validates completeness. It MUST NOT infer a missing lifecycle phase, continuation edge, resource dimension, or synchronization result from aggregate timing.

## Production lineage proof

Architecture evidence follows this order:

1. Static reachability from a configured public request and admitted capability.
2. Exact product and state lineage across scheduler, worker, stores, completion, commit, and public delivery.
3. Production trace proving successor registration and submission before parent completion observation.
4. Depth-one end-to-end oracle across every admitted feature combination and completion ordering.
5. Formal qualification at one immutable source revision.

A manually assembled operation, worker primitive, simulator property, or greedy-only public profile cannot qualify a broader production claim.

## Acceptance

Runtime conformance requires all of the following at one immutable clean revision:

1. Cross-language protocol and digest parity for every record, work leaf, control variant, malformed bound, duplicate, and conflict.
2. Atomic reserve/register, exact point, ordered control, predicate, product lifetime, KV, latent, artifact, transfer, rollback, and reclamation properties.
3. Depth-one equivalence at every advertised window depth for every admitted sampling processor and modality edge.
4. Production traces showing every claimed same-request continuation before parent host observation.
5. Static and runtime zero-blocking gates with zero production-reachable violations or detections.
6. Complete lifecycle and resource observability for every measured request.
7. Resource, cancellation, failure, delayed-event, slow-consumer, slow-CPU, and slow-transfer stress with fixed maxima and no unrelated-lineage stall.
8. Homogeneous, mixed, tensor-parallel, and configured local-stage structural and numerical conformance.
9. The fixed correctness, architecture, performance, provenance, and reference protocol in [`generation-runtime-qualification.md`](generation-runtime-qualification.md).
