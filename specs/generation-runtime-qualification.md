# Generation runtime qualification protocol

## Status and scope

This implementation-time protocol defines correctness, architecture, performance, provenance, and evidence acceptance for the runtime specified by [`decode-runtime.md`](decode-runtime.md), the serving surface specified by [`serving-surface.md`](serving-surface.md), and the construction sequence specified by [`decode-runtime-construction.md`](decode-runtime-construction.md). The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative.

The benchmark measurement interface and artifact bundle are documented in [`docs/benchmark-protocol.md`](../docs/benchmark-protocol.md). The canonical executable profile definitions are in [`uniserve_eval/profiles.json`](../uniserve_eval/profiles.json).

## Evidence levels

| Level | Evidence |
| --- | --- |
| L0 | The mechanism is absent, contradicted, or unreachable. |
| L1 | A type, primitive, simulator property, or isolated worker test exists. |
| L2 | A configured production request lineage reaches the mechanism. |
| L3 | A public end-to-end request exercises the mechanism and a correctness oracle verifies it. |
| L4 | The exact immutable source revision passes the complete formal protocol with provenance and observability. |

An architecture claim is qualified only at L4. Primitive, reachability, end-to-end, and formal evidence are recorded separately.

## Frozen environment and provenance

Every formal artifact binds:

- Source commit, tree digest, clean-state proof, executable critical-path digest, generated protocol digest, and build commands.
- Python environment, native extension ABI, Rust target, dependency lockfiles, CUDA runtime and driver, kernel providers, and environment variables affecting execution.
- Model identities, revisions, required file counts, sizes, and content-tree digests.
- Dataset identities, revisions, selected-row algorithm, selected-row digest, selected-row count, preprocessing, and local artifact digest.
- Server profile, route capability digest, topology, GPU identity, NUMA binding, cache policy, capacity settings, and sampling owner.
- Benchmark profile, normalized request digest, sampling and image controls, load semantics, warmup, measured-run count, metric definitions, and acceptance thresholds.
- Canonical artifact root, manifest digest, server and harness commands, process snapshots, request records, logs, and result checks.

Formal points require a clean immutable candidate worktree and a fresh server. A source, protocol, profile, evaluator, model, dependency, or executable change invalidates every affected artifact.

## Frozen workload points

### Per-candidate protection points

| Point | Profile selection | Work and load | Primary protected metrics |
| --- | --- | --- | --- |
| Qwen3 ShareGPT | `qwen-uniserve-sharegpt-r16` | 200 selected ShareGPT rows, open-loop seed-42 arrival at 16 requests/s, greedy fixed output work | Aggregate output tokens/s, mean TTFT, mean TPOT |
| SenseNova T2I | `sensenova-uniserve-t2i-c32` | 32 selected MJHQ prompts, immediate concurrency 32, one 2048×1152 image, 50 denoise updates | Images/s |
| SenseNova I2T | `sensenova-uniserve-i2t-c32` | 32 selected Beans images, immediate concurrency 32, natural EOS under the fixed ceiling | Aggregate output tokens/s |

### Major-boundary points

| Point | Profile selection | Purpose |
| --- | --- | --- |
| SenseNova default travel | `gate/sensenova/default-travel` | Fixed requested text/image/text work, output structure, image dimensions, denoise updates, transition work, and provenance. |
| SenseNova greedy UEval | `sensenova-uniserve-interleave-c4` | Public TTFT, TPOT, image latency, aggregate and directional transition distributions under concurrency four. |

### Architecture-claim point

The stochastic interleave profile is `sensenova-uniserve-stochastic-interleave-c4`. It fixes:

- SenseNova-U1 model and route profile.
- UEval revision `fdeac6654b113d20e7e89d496167a4a5bb55bc66`.
- Seed-42 row selection, 32 rows, natural EOS, and an 8192-token ceiling.
- Temperature `0.7`, top-p `0.9`, top-k `50`, min-p `0.05`, repetition penalty `1.05`, frequency penalty `0.1`, presence penalty `0.1`, and sampling seed `42`.
- 2048×1152 output, 50 denoise updates, text guidance `4.0`, image guidance `1.0`, no CFG normalization, guidance interval `[0.0, 1.0]`, timestep shift `3.0`, image thinking disabled, and image epsilon `0.02`.
- One TP2 UniServe server, one harness, immediate concurrency four, one warmup request, enabled cache reads and writes, and one measured run.
- Required lifecycle, window, continuation, resource, synchronization, domain, Gen, public-transition, and per-request work evidence defined below.

The profile is distinct from greedy UEval and default travel. Its normalized request, selected-row, benchmark-definition, model, capability, source, executable, and environment digests MUST match across every depth comparison.

### Reference points

Qwen3 ShareGPT reference points use the pinned SGLang source and matched selected rows, tokenizer, request work, sampling controls, arrival semantics, cache policy, capacity, dtype, and hardware. SenseNova T2I and I2T reference points use the pinned vLLM-Omni source with declared feature and capacity mismatches preserved in the artifacts.

The accepted performance-reference source is `8c33a6fcc0cfc6da9e5a14a698b0efbba3e0c8f5`. Its candidate artifacts are:

- [`artifacts/qualification/decode_runtime/checkpoint/qwen3_sharegpt/uniserve_r16`](../artifacts/qualification/decode_runtime/checkpoint/qwen3_sharegpt/uniserve_r16)
- [`artifacts/qualification/decode_runtime/checkpoint/sensenova_mjhq_t2i/uniserve_c32`](../artifacts/qualification/decode_runtime/checkpoint/sensenova_mjhq_t2i/uniserve_c32)
- [`artifacts/qualification/decode_runtime/checkpoint/sensenova_beans_i2t/uniserve_c32`](../artifacts/qualification/decode_runtime/checkpoint/sensenova_beans_i2t/uniserve_c32)
- [`artifacts/qualification/decode_runtime/default_travel_tp1`](../artifacts/qualification/decode_runtime/default_travel_tp1)

Each artifact is admitted as a comparator only when its source, protocol, workload, environment, correctness, and manifest contracts match the active point. An incompatible artifact remains evidence for its own revision and a fresh matched reference run is required.

The construction anchor and the pre-change comparator roots for profiles introduced after that anchor are declared by [`qualification/generation-runtime/references.json`](../qualification/generation-runtime/references.json). Those declarations are immutable inputs to candidate manifests and never selected from observed candidate results.

## Correctness gates

### C0: Contract and artifact integrity

Parse every specification, profile, manifest, schema, and acceptance record. Validate relative links, JSON schemas, profile expansion, command tokenization, selected workload point set, normalized request definitions, model and dataset contracts, environment requirements, capability digests, metric definitions, comparator identities, and canonical artifact roots.

Acceptance requires no broken link, invalid schema, unresolved contract conflict, mutable input, undeclared capability, ambiguous metric, or profile-dependent point selection.

### C1: Repository presubmit

Run:

```text
just lint
just test-rust
just test-python-fast
just test-python-integration
just test-python-e2e
```

After a repair, rerun the smallest affected package or module first and then every invalidated C1 command. Required prerequisites are provisioned at their specified versions. Acceptance requires zero failures and zero required-test skips.

### C2: Protocol and state properties

The core commands are:

```text
cargo test -p uniserve-worker-wire
cargo test -p uniserve-sim --test control_plane
.venv/bin/python -m pytest tests/python/unit/contracts/test_batch_protocol.py tests/python/contract/worker
```

The candidate manifest adds affected scheduler and worker property suites. Acceptance requires cross-language layout and digest parity, atomic reserve/register, idempotence, conflict containment, exact points, ordered controls, predicates, bounded resources, event-safe lifetime, and schedule-independent identity.

### C3: Real-device runtime integration

Run the affected CUDA integration coverage for completion storage, product lifetime, predicated execution, sampling, device continuation, KV publication, model execution, mixed calls, graph selection, transport, and worker warmup on the declared production GPU and provider.

Acceptance requires exact semantic and lifetime invariants, fixed numerical and image-quality thresholds, the declared provider, every affected route, and no substitute or undeclared fallback.

### C4: Public production lineage

Run `UNISERVE_RUN_GPU_E2E=1 just test-python-gpu` and one bounded public request for each affected configured route and topology through frontend, scheduler, worker, model, sampler, completion, commit, and public delivery.

Acceptance requires correct output and work, finite resource use, complete lifecycle and public trace, zero failures, zero forbidden synchronization detections, and every claimed successor submitted before parent completion observation.

### C5: Semantic, pressure, and failure qualification

Compare the depth-one serial oracle with every advertised depth and admitted feature combination. Run applicable cancellation, duplicate, stale-generation, slow-consumer, slow-CPU, slow-transfer, delayed-event, pressure, peer-failure, and recovery scenarios.

Acceptance requires identical protocol identity, semantic points, RNG coordinates, constraint decisions, public roots, output order, and fixed work. Numerical tensors and image outputs use thresholds frozen independently of the candidate result. No ambiguous state, leak, live reuse, partial registration, or unrelated-lineage stall is permitted.

## Performance and architecture gates

### P0: Executable equivalence

Compare candidate and parent dependency footprints, resolved binary and module sources, protocol and profile digests, evaluator digests, configured request-path executable hashes, and candidate manifest impact classes without launching a server.

P0 is valid only when the manifest proves no configured executable critical path or measurement contract changed. Otherwise P1 is mandatory.

### P1: Per-candidate performance protection

Run the fixed Qwen3 ShareGPT, SenseNova T2I, and SenseNova I2T protection points serially. Compare each metric with the construction anchor, compatible performance reference, and immediately preceding accepted candidate.

Correctness, work, provenance, profile, environment, and artifact checks must pass. Every larger-is-better and smaller-is-better metric permits at most 20% regression against every applicable comparator. This gate protects performance; it does not prove an optimization claim.

### P2: Major-boundary multimodal protection

After P1, run default travel and greedy UEval with one fresh server per point. Default travel requires exact requested and realized work identity for direct elapsed-time comparison. Greedy UEval requires complete TTFT, TPOT, decoded-image latency, aggregate transition, and directional transition distributions.

Every primary metric permits at most 20% regression against every applicable comparator. Missing work, transitions, timestamps, correctness, provenance, or complete requests invalidates the point.

### P3: Stochastic architecture-claim qualification

Run the frozen stochastic interleave profile at depth one and every advertised depth. The runtime artifact MUST contain, for every request and operation:

- Complete lifecycle timestamps keyed by `(request_key, op_id, control_seq, domain)`.
- Maximum and effective window depth.
- Eligible and actual continuations classified by sampling mode and modality edge.
- Parent completion-observed and successor registered/submitted timestamps.
- Every credit dimension and store generation pressure.
- Predicate, retraction, selected point, sampling-state delta, RNG coordinate, KV extent, Gen step, artifact, feedback, and public-transition evidence.
- Forbidden device wait, scalar observation, synchronous transport, sleep poll, and blocking reclaim counters.

Acceptance requires every eligible stochastic and cross-modal successor to be submitted before parent completion observation, effective depth greater than one while eligible work exists, depth-one semantic and public oracle agreement, zero forbidden detections, all 32 samples successful, complete realized transition coverage, and no primary metric more than 20% worse than every applicable frozen comparator.

### P4: Final production qualification

Freeze the exact source commit and environment and run, in predeclared serial order:

1. C0 through C5.
2. The complete P1 triplet.
3. P2 default travel and greedy UEval.
4. P3 stochastic depth comparison.
5. Aligned reference-system points.
6. TP1 and every configured multi-rank topology.
7. Every advertised tensorized mixed combination and its interference profile.
8. Resource pressure, cancellation, topology fault, and recovery matrices.

Acceptance requires L4 evidence for every configured claim at one revision. Qwen3 ShareGPT aggregate output throughput MUST reach at least 95% of the aligned SGLang point while correctness and latency gates pass. Every performance-protection metric remains within 20% of applicable comparators. No evidence from another revision or invalid run is spliced into final results.

### P5: Optimization retention

Before implementation, record the measured end-to-end gap, full production lineage, profiler-attributed boundary, dynamic frequency, normalized cost, removable upper bound, production reachability, correctness constraints, proposed mechanism, and controlled ablation.

Measure the exact accepted parent and candidate under one locked protocol after C1-C4. Retain the optimization only when the primary end-to-end metric improves by at least 1.1x, all correctness and work invariants match, no protected metric exceeds its regression threshold, and matched profiling confirms the intended boundary changed. Otherwise restore the parent behavior completely and record the measured negative result only in the historical builder log.

## Architecture proof matrix

| Proof | Coverage | Acceptance |
| --- | --- | --- |
| Protocol identity | Every work and control leaf, malformed bounds, duplicate/conflicting messages, allocation, batch, depth, topology, and completion permutations | Stable exact digests and no physical identity in semantic digests. |
| Static zero-blocking | Frontend progress, scheduler, executor, worker, model, attention, vision, sampler, stores, transport, tracing, and metrics | Zero production-reachable forbidden call sites outside classified startup/export/recovery. |
| Runtime zero-blocking | Greedy, stochastic, processors, logprobs, stop, vision, KV, Gen, feedback, mixed, TP, local stages, and cancellation | Zero forbidden detections. |
| Serial oracle | Every admitted sampling combination and modality edge, cancellation, replay, mixed permutation, and TP ownership | Identical identity, selected points, decisions, RNG, committed output, and fixed numerical conformance. |
| Production reachability | Public request through response delivery | Every claimed successor precedes parent host observation in the production trace. |
| Resource stress | Delayed events, completion disorder, slow client/CPU/transfer, cancellation, and failure storms | Fixed maxima, deterministic backpressure, no partial registration, live reuse, or semantic loss. |
| Mixed and interleave | Homogeneous and mixed execution over supported shapes, batches, graphs, and transition directions | Identical Und structure and semantics, independent completion, complete timing, and declared interference. |
| Topology and recovery | TP ranks, configured local stages, transfer failure, stale epoch, gap, digest and cutoff conflicts | All-rank agreement, exact bases, contained failure, and deterministic epoch handling. |

## Artifact validity

Canonical qualification counts only valid, complete, serial artifacts produced by the declared immutable candidate and profile. Exploratory, warmup, failed, interrupted, parallel, pre-repair, mismatched, degraded, or superseded artifacts do not enter acceptance tables.

Every point starts with a fresh server and clean GPU. At most one server and one harness run at a time. Automatic retry count is zero. A repeated failure mode receives at most one confirmation before artifact execution stops for diagnosis.

Final artifacts include the candidate manifest, acceptance record, source and environment contracts, commands, process snapshots, complete request records, lifecycle and resource traces, metrics, correctness results, comparator checks, and canonical artifact manifest. Acceptance records name the exact source commit, tree digest, executable digest, profile digest, and artifact digest.

## Completion

Production qualification is complete only when P4 passes against the clean immutable S22 candidate, every configured capability reaches L4, every required artifact is canonical and provenance-complete, and no executable or contract change follows measurement.
