# Generation runtime construction protocol

## Status and scope

This implementation-time protocol governs construction of the runtime specified by [`decode-runtime.md`](decode-runtime.md) and the serving surface specified by [`serving-surface.md`](serving-surface.md). The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative.

The sequence establishes complete ownership and capability boundaries while keeping every configured route usable. Qualification checks, profiles, metrics, and acceptance thresholds are defined by [`generation-runtime-qualification.md`](generation-runtime-qualification.md).

## Construction principles

Each retained candidate MUST establish one complete system boundary across every affected language, process, store, route, test, configuration, and document. It MUST leave one canonical implementation and MUST satisfy the positive behavioral contract for every configured route.

The construction sequence prioritizes ownership and protocol boundaries capable of changing end-to-end behavior:

1. Canonical contracts and immutable experiment control.
2. Serving and capability closure.
3. Single operation, identity, KV allocation, and physical lowering ownership.
4. Exact capabilities, zero-blocking execution, transport, and observability.
5. Device sampling state and the generalized bounded request window.
6. Cross-modal continuation, mixed execution, topology, pressure, and recovery.
7. Current-state documentation and final qualification.

An edit proposed as a performance optimization MUST additionally satisfy the independent optimization-retention contract in the qualification specification. Correctness or architecture work can be retained without a measured speedup when every regression guard passes.

## Immutable candidate discipline

The parent of each candidate is the immediately preceding accepted commit. The controller creates an isolated candidate commit object without advancing the development branch, records an affected-scope manifest, and validates that commit in a clean linked worktree.

The affected-scope manifest declares:

- Candidate parent and expected development ref.
- Source paths and generated artifacts affected by the boundary.
- Protocol, profile, model, evaluator, kernel, dependency, and executable impact classes.
- Required correctness checks from C0 through C5.
- Required performance and architecture points from P0 through P5.
- Construction anchor, performance reference, immediately preceding accepted candidate, and applicable major-boundary comparator roots.
- Ordered serial commands, prerequisites, timeouts, and canonical output roots.
- Critical-path and contract digests used to justify P0 when no measurement is required.

The controller MUST reject a dirty validation worktree, a candidate whose tree differs from the declared commit, a parent mismatch, an undeclared affected path, an incomplete required point set, a reused mutable artifact root, and an acceptance attempt that would not fast-forward the development ref.

A passing candidate advances the development ref by fast-forward to the exact validated commit. A failing candidate is repaired and materialized as a replacement candidate with the same accepted parent. No failing or unqualified candidate advances the accepted sequence.

Any executable, model-profile, evaluator, kernel, dependency, workload, or benchmark-contract change invalidates affected evidence. A documentation-only change can reuse executable evidence only when the controller proves unchanged executable critical-path, protocol, profile, and qualification digests.

## Validation order

Validation proceeds in increasing scope and stops at the first failure:

1. C0 contract, schema, link, profile-expansion, and provenance checks.
2. C1 repository presubmit and smallest affected package tests.
3. C2 cross-language protocol and state properties.
4. C3 real-device runtime integration for the affected mechanism.
5. C4 public production-lineage requests and trace checks.
6. C5 serial semantic, pressure, cancellation, topology, and failure oracles.
7. P0 executable equivalence or the required P1-P3 measurements.
8. P5 when the candidate is an optimization.

All artifact-producing points run serially with at most one server and one harness. Automatic retry is zero. A crash, timeout, invalid artifact, work mismatch, failed prerequisite, or failed check stops execution for diagnosis. One confirmation run is permitted only for a valid but surprising measurement and preserves both results.

## Construction boundaries

### S00: Canonical contracts and frozen profiles

Establish one generation runtime, serving surface, construction, and qualification contract. The configured claims are the three-model serving surface, the four-record operation protocol, device-continuous ordinary sampling and cross-modal interleave where declared, colocated tensor-parallel and local-stage transfer, and exact unsupported declarations for grammar, adapters, production speculation, cross-node disaggregation, and preemption.

Freeze greedy protection profiles and a separate stochastic interleave profile before observing its results. Bind model and dataset revisions, selected-row procedure, seeds, sampling processors, image controls, topology, load semantics, cache policy, trace requirements, metric definitions, acceptance thresholds, and artifact roots. Record immutable manifests for the construction anchor, the accepted performance reference, and a pre-change comparator root for each newly introduced profile.

Required gates: C0 and P0.

### S01: Candidate and acceptance controller

Make the immutable candidate-ref, isolated validation worktree, affected-scope manifest, serial command runner, stop-on-failure behavior, comparator selection, and fast-forward-only acceptance path executable. Point selection MUST come only from the manifest and active qualification contract.

Required gates: C0, C1 controller tests, and P0 dry expansion of the exact selected point set.

### S02: Serving request funnel

Establish `GenerateReqInput` as the sole internal generate admission value and `ResolvedModel` as the closed load-bound owner of `tokenize`. Every configured request lowers through the one funnel and submits one `TokenizedGenerateReqInput` to the engine gateway.

Required gates: C1, public schema contracts, tokenize and stream-assembly oracles, and P1 when a configured executable path changes; otherwise P0 with critical-path equivalence.

### S03: Sampling and model capability closure

Make the configured public schema, model descriptions, frontend, scheduler, worker, stores, capabilities, dependencies, and documentation implement the exact serving surface. Shared CPU continuations, request-session rollback, and base-weight identity remain runtime facilities.

Required gates: C1, C2, retained sampling, stop, CPU-continuation, request-session, and capability-digest properties, plus P1.

### S04: Public protocol closure

Make retained OpenAI chat and image generation, health, metrics, version, and model discovery the complete serving protocol set. The configured frontend contains one request lowering and response assembly ownership path.

Required gates: C1, configured HTTP end-to-end streaming and cancellation properties, workspace ownership audit, and P1 when the serving binary or request path changes; otherwise P0.

### S05: Configured speculation and transport boundary

Advertise neither Draft nor Verify without a configured producer and retain their depth-one worker behavior as closed protocol leaves. Configured transport consists of colocated tensor-parallel and local-stage product, conditioning, and KV transfer through exact products.

Required gates: C1, C2, capability closure, configured transfer oracle, and P1.

### S06: Configured model preprocessing

Make tokenization, rendering, parsing, model-profile, and omni framing ownership match the configured Qwen3, SenseNova, and Bagel descriptions. Every retained fixture is derived from the current serving contract.

Required gates: C1, retained tokenizer, template, parser, description, public chat, and image route fixtures, plus P1 when preprocessing or executable layout changes; otherwise P0.

### S07: KV allocation and semantic identity

Move physical KV page allocation into atomic worker registration. Operations carry logical capacity and exact semantic parent/product identity. Make plan, semantic, and control identity canonical and independent of physical allocation.

Required gates: C1, C2, allocation, batch, depth, topology, completion permutation properties, registration rollback, cross-language digest parity, and P1.

### S08: Scheduler operation ownership

Consume the ephemeral scheduler builder at registration. Retain only the immutable `Operation` and scheduler apply record keyed by `(request_key, op_id)` after registration.

Required gates: C1, C2, serial simulator oracle, cancellation, ordered control, configured route smokes, and P1.

### S09: Direct physical lowering

Derive physical calls directly from registered operations and model row declarations. Output bindings come from row ABI and `Operation.outputs`; base versions come only from registered operation inputs.

Required gates: C1-C3, depth-one execution for every configured leaf, output-shape, product-binding, graph-selection, worker-registration stress, P1, and P2.

### S10: Executable route capabilities

Make every route capability an exact executable declaration of work, unresolved window, processor coverage, RNG, graph, attention, topology, Gen, transfer, snapshot, mixed execution, and credits. Startup proves depth-one lowering for every advertised leaf.

Required gates: C1-C3, capability digest parity, lowering proof, deterministic unsupported-combination rejection, and P1.

### S11: Device and host shape ownership

Supply every production attention, vision, shape, append, and extent decision from host-known registration metadata or device-resident products for every configured model and backend.

Required gates: C1-C3, static reachability checks for completed call graphs, text, vision, Gen, graph, and eager numerical/shape oracles, P1, and a route-specific profiler trace. P5 applies when this boundary is proposed as an optimization.

### S12: Asynchronous transfer tickets

Use bounded asynchronous tickets for steady-state product, KV, conditioning, and local-stage transfer. Keep administrative snapshot export and restore outside steady state with separate capability and trace classification.

Required gates: C1-C3, delayed producer and reader events, ticket backpressure, cancellation, peer failure, exact-base validation, administrative recovery tests, P1, and a transfer-sensitive route smoke.

### S13: Zero-blocking enforcement

Make static call-graph analysis and runtime instrumentation enforce query-only request progress over every configured route and mechanism.

Required gates: C1-C4, full static reachability gate, runtime detector coverage for greedy, stochastic, image encode, KV, Gen, feedback, TP, local-stage transfer, cancellation, tracing, and metrics, plus P1 and P2 with zero detections.

### S14: Lifecycle and resource observability

Emit one operation lifecycle and resource/window metric surface keyed by canonical identity and correlated with public events.

Required gates: C1-C4, timestamp order and completeness, asynchronous device timing, counter conservation, evaluator serialization, public-transition correlation, P1, and P2.

### S15: Device sampling state and exact RNG

Produce worker-resident committed sampling bases and bounded per-operation ancestral deltas. Use one exact cross-language counter-based RNG mapping and one processor order for ordinary sampling and configured Verify.

Required gates: C1-C3, ordinary/Verify oracle, every admitted processor, batch, depth, completion, retry and replay permutations, delta lifetime and sibling isolation, and P1. A stochastic production trace MUST show successor product consumption before parent observation.

### S16: Generalized token continuation

Construct bounded successors for every device-representable admitted stochastic, penalty, logprob, minimum-token, static-mask, forced-token, and finish combination from capability and product availability.

Required gates: C1-C5, public depth-one versus advertised-depth oracle, production trace showing successor submission before parent completion observation, exact semantic and public cursor properties, P1, and P3.

### S17: CPU semantic horizons

Bound CPU semantic horizons and retract provisional descendants by exact point while unrelated requests continue. One shared future mechanism owns stop strings and configured CPU processors.

Required gates: C1-C5, split-token stop strings, slow and timed-out CPU tasks, exact rollback, cancellation cutoff, slow-client isolation, fixed resource maxima, P1, and P3.

### S18: Gen transition and flow continuation

Emit device transition predicates and chain predicated Gen Transition and bounded Gen Flow operations in the generalized request window.

Required gates: C1-C5, exact conditioning and KV extent, latent generation safety, false-predicate no-op, flow bounds, depth-one numerical and artifact-quality oracle, and P1-P3.

### S19: Materialization and feedback continuation

Chain Materialize, immutable image feedback, Encode, Token Extend, and resumed Decode or Gen successors through exact products and parents.

Required gates: C1-C5, repeated text-image-text travel, independent device, D2H, encode and public readiness, image dimensions and denoise work, output order, cancellation, and P1-P3.

### S20: Independent domain partitions and mixed execution

Complete independent Und and Gen partitions and qualify every advertised tensorized mixed row combination.

Required gates: C1-C5, batch, shape, graph, collective and completion permutations, homogeneous-versus-mixed structural and numerical oracle, independent error and backpressure, exact domain accounting, P1-P3, and the fixed mixed-service interference envelope.

### S21: Topology, pressure, and failure containment

Qualify tensor-parallel sampling ownership, configured local-stage transfers, resource pressure, cancellation, peer failure, and exact recovery behavior consistent with the unsupported preemption declaration.

Required gates: C1-C5, all-rank identity, duplicate, stale, gap, digest and cutoff conflict properties, delayed events, slow client, CPU and transfer pressure, cancellation and failure storms, fixed credit maxima, P1-P3 on every configured topology, and the topology/failure subset of P4.

### S22: Current-state specification and clean qualification tree

Publish a current-state runtime specification and release-quality user documentation, complete the source and prose ownership audit, freeze the final executable digest, and prepare the clean immutable qualification candidate.

Required gates: C0-C5, complete requirement-to-evidence table, workspace ownership and capability audit, link and schema checks, and P0 for documentation-only changes whose executable and contract digests are unchanged. Any executable change returns to its owning boundary and invalidates affected evidence.

## Candidate manifest schema

The canonical manifest is a versioned JSON object with these required top-level fields:

```json
{
  "schema_version": 1,
  "candidate": {
    "parent": "<commit>",
    "commit": "<commit>",
    "development_ref": "refs/heads/<name>"
  },
  "boundary": "S00",
  "affected_scope": {
    "paths": [],
    "impact_classes": []
  },
  "checks": [],
  "comparators": {},
  "artifact_root": "artifacts/qualification/generation_runtime/<boundary>/<candidate>",
  "acceptance": {
    "fast_forward": true,
    "update_ref": false
  }
}
```

The candidate plan schema in `schemas/generation-runtime-candidate-plan.schema.json` and immutable candidate schema in `schemas/generation-runtime-candidate.schema.json` are authoritative. Commands are arrays of tokens, not shell strings. Check entries declare stable IDs, gate classes, serial order, prerequisites, timeout, artifact role, and whether they mutate only the declared artifact root.

An acceptance run with `update_ref=false` validates and emits a signed-by-digest acceptance record without changing the development ref. `update_ref=true` is permitted only after every declared check passes and the ref still names the manifest parent.

## Completion

Construction is complete only when S00 through S22 are accepted in order from immediately preceding accepted parents and final P4 qualification binds the exact clean S22 commit. Evidence from another source tree, profile revision, executable, model asset, dependency environment, or failed run cannot qualify the final runtime.
