# S02 serving request funnel recovery specification

## Status and purpose

This implementation-time recovery specification records the available provenance of the S02 candidate and defines the procedure for restoring the S02 serving request funnel. [`serving-surface.md`](serving-surface.md), [`decode-runtime-construction.md`](decode-runtime-construction.md), and [`generation-runtime-qualification.md`](generation-runtime-qualification.md) remain the normative product contracts. This file governs evidence handling, reconstruction scope, implementation order, and acceptance claims for S02.

S02 establishes `GenerateReqInput` as the sole internal generate admission value, `ResolvedModel` as the closed load-bound owner of `tokenize`, and `TokenizedGenerateReqInput` as the single value submitted through the engine gateway. Every configured OpenAI generation request follows this ownership chain:

```text
OpenAI request
  -> fallible wire lowering
  -> GenerateReqInput
  -> ResolvedModel::tokenize
  -> TokenizedGenerateReqInput
  -> EngineGateway::submit
  -> ServeEvent stream
  -> OpenAI response assembly
```

## Provenance record

| Item | Recorded value | Evidentiary role |
| --- | --- | --- |
| Repository | `https://github.com/Boreas618/UniServe.git` | Canonical remote |
| Accepted parent and `main` anchor | `d538df72d7640f7f19802f830d7ccda8141182dc` | Reconstruction parent |
| S02 candidate object identifier | `9687bc949e03ea3267dbf37450aaab2061830001` | Byte-level recovery target |
| Recovery branch | `agent/dead-node-recovery` | Persisted evidence anchor |
| Recovery commit | `8059348440df1295c907e3e18fb0b15b7f9ea80d` | Exact `AGENTS.md` snapshot and 44 recorded deletions |
| Original worktree | `/workspace/UniServe` | Cursor workspace identity |
| Workspace storage identifier | `f0286d290504e5205c2dbec17d873255` | Cursor metadata lookup key |
| Git-status content record | `agentKv:blob:24522c50ff7a0052a8e7d38f9f716335471fce8fb72ccaa33797d2739af50790` | Full 113-path status snapshot |
| Exact recovered instruction digest | `85e9363935159a6116f6e448995aba793d40c0fbb0ee2de009188502f099f4df` | SHA-256 identity of `AGENTS.md` |

The candidate identifier occurred in Cursor editor history under `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/`. The qualification controller created candidate commits through `git commit-tree`, so the identifier denotes a Git commit object whose tree captured the candidate worktree at candidate-materialization time.

GitHub commit lookup returned HTTP 404 for the candidate identifier on 2026-08-04. The accessible local clone reported a missing object for the same identifier. These observations place the candidate object bytes in the dead-node Git object store or a storage snapshot derived from that node.

## Evidence classes and claim limits

| Class | Available material | Permitted claim |
| --- | --- | --- |
| Exact source content | `AGENTS.md` content snapshot with matching SHA-256 | Byte identity for that file |
| Exact tree operation | Forty-four `D` path records preserved in commit `8059348` | Identity of the deletion set |
| Structural source metadata | Sixty-two modified paths, seven added paths, editor line counts, open-editor history | Scope, ownership, and file-size constraints |
| Candidate provenance | Candidate object identifier and qualification artifact paths | Identity and intended qualification scope |
| Product contract | Current S02 and serving-surface specifications | Semantic reconstruction requirements |
| Intermediate fragments | Cursor tool diffs and earlier content records | Design clues with per-fragment provenance |

Byte-identity claims require the candidate Git object or a complete content snapshot with a verifiable digest. Specification-conformance claims require a fresh S02 implementation and qualification under the repository contracts. Intermediate fragments contribute design evidence after their creation order and file target are established.

## Cursor metadata inventory

The workspace editor state recorded these source and candidate artifact sizes:

| Path | Recorded line count |
| --- | ---: |
| `crates/frontend/serving/src/input.rs` | 293 |
| `crates/frontend/serving/src/model.rs` | 1477 |
| `qualification/generation-runtime/plans/S02.json` | 521 |
| `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/candidate.json` | 525 |
| `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/c0_contract.json` | 17 |
| `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/checks/c1_test_python_integration.log` | 26 |
| `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/checks/p1_qwen3_sharegpt_r16.log` | 9 |
| `artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/checks/p1_sensenova_t2i_c32.log` | 8 |

Cursor history also referenced the following candidate image artifact:

```text
artifacts/qualification/generation_runtime/S02/9687bc949e03ea3267dbf37450aaab2061830001/benchmarks/sensenova_mjhq_t2i/uniserve_c32/samples/1e8f41540a69e5f4a340ca113e9991797d32dc1c4ca3da8f5faca8f7d7933fcf.png
```

A Cursor WebStorage cache entry retains data for that resource. The cache-entry file has SHA-256 `5b4bb77abfc156016fa5d27e436991c5b0c454521e9959cfeaed1c4fb81834a9`; this digest covers the cache representation and serves as provenance evidence for the artifact reference.

Editor history proves that these files were opened. The line counts prove the editor model sizes at the persisted workspace-state point. Qualification outcome claims require the corresponding artifact bodies and manifest digests.

## Recorded affected scope

The Cursor conversation-start snapshot contains 113 paths: 62 modified, 44 deleted, and 7 added. The raw status symbols are retained below because the `Cargo.lock` entry carries a distinct index/worktree presentation from the other modified entries.

```text
M Cargo.lock
 M crates/bin/uniserve/src/cli.rs
 M crates/frontend/engine-gateway/src/lib.rs
 M crates/frontend/engine-gateway/tests/canonical_generation.rs
 D crates/frontend/model-profile/profiles/thinkmorph.json
 M crates/frontend/model-profile/src/assets/config.rs
 M crates/frontend/model-profile/src/dialect/mod.rs
 M crates/frontend/model-profile/src/lib.rs
 M crates/frontend/protocol-adapters/Cargo.toml
 D crates/frontend/protocol-adapters/src/grpc/mod.rs
 M crates/frontend/protocol-adapters/src/lib.rs
 D crates/frontend/protocol-adapters/src/native/convert.rs
 D crates/frontend/protocol-adapters/src/native/events.rs
 D crates/frontend/protocol-adapters/src/native/mod.rs
 D crates/frontend/protocol-adapters/src/native/schema.rs
 M crates/frontend/protocol-adapters/src/openai/chat_completions/convert.rs
 M crates/frontend/protocol-adapters/src/openai/chat_completions/mod.rs
 M crates/frontend/protocol-adapters/src/openai/chat_completions/response.rs
 M crates/frontend/protocol-adapters/src/openai/chat_completions/validate.rs
 M crates/frontend/protocol-adapters/src/openai/completions/convert.rs
 M crates/frontend/protocol-adapters/src/openai/error.rs
 M crates/frontend/protocol-adapters/src/openai/images.rs
 D crates/frontend/protocol-adapters/src/openai/lora.rs
 M crates/frontend/protocol-adapters/src/openai/mod.rs
 D crates/frontend/protocol-adapters/src/openai/structured_outputs.rs
 M crates/frontend/protocol-adapters/src/openai/utils.rs
 D crates/frontend/protocol-adapters/src/raw_generate/convert.rs
 D crates/frontend/protocol-adapters/src/raw_generate/mod.rs
 D crates/frontend/protocol-adapters/src/raw_generate/response.rs
 D crates/frontend/protocol-adapters/src/raw_generate/types.rs
 D crates/frontend/protocol-adapters/src/raw_generate/validate.rs
 M crates/frontend/serving/Cargo.toml
 D crates/frontend/serving/src/chat/backend/hf.rs
 D crates/frontend/serving/src/chat/backend/mod.rs
 M crates/frontend/serving/src/chat/mod.rs
 M crates/frontend/serving/src/chat/protocol/request.rs
 M crates/frontend/serving/src/chat/template/mod.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_dsml.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v32/encoding.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v32/fixtures/test_input.json
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v32/fixtures/test_output_uniserve_parity.txt
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v32/mod.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v32/tests.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/encoding.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/fixtures/test_input_1.json
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/fixtures/test_input_2.json
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/fixtures/test_output_1.txt
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/fixtures/test_output_2.txt
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/mod.rs
 D crates/frontend/serving/src/chat/template/renderer/deepseek_v4/tests.rs
 M crates/frontend/serving/src/chat/template/renderer/hf/mod.rs
 M crates/frontend/serving/src/chat/template/renderer/mod.rs
 D crates/frontend/serving/src/chat/template/renderer/selection.rs
 D crates/frontend/serving/src/dialect_generation/defaults.rs
 D crates/frontend/serving/src/dialect_generation/mod.rs
 A crates/frontend/serving/src/input.rs
 M crates/frontend/serving/src/lib.rs
 A crates/frontend/serving/src/model.rs
 A crates/frontend/serving/src/omni/mod.rs
 D crates/frontend/serving/src/text/backend/hf/mod.rs
 D crates/frontend/serving/src/text/backend/mod.rs
 M crates/frontend/serving/src/text/error.rs
 D crates/frontend/serving/src/text/lower.rs
 M crates/frontend/serving/src/text/mod.rs
 D crates/frontend/serving/src/text/request.rs
 D crates/frontend/serving/src/text/structured_output.rs
 A crates/frontend/serving/tests/assembly.rs
 M crates/frontend/serving/tests/chat.rs
 M crates/frontend/serving/tests/chat_template.rs
 A crates/frontend/serving/tests/common/mod.rs
 A crates/frontend/serving/tests/tokenize.rs
 M crates/server/app/Cargo.toml
 M crates/server/app/src/config.rs
 D crates/server/app/src/grpc/mod.rs
 D crates/server/app/src/grpc/tests.rs
 M crates/server/app/src/http/mod.rs
 M crates/server/app/src/http/routes/http_client_tests.rs
 D crates/server/app/src/http/routes/inference/generate.rs
 D crates/server/app/src/http/routes/inference/mod.rs
 D crates/server/app/src/http/routes/inference/native.rs
 D crates/server/app/src/http/routes/lora.rs
 M crates/server/app/src/http/routes/mod.rs
 M crates/server/app/src/http/routes/openai/chat_completions.rs
 M crates/server/app/src/http/routes/openai/completions.rs
 M crates/server/app/src/http/routes/openai/images.rs
 M crates/server/app/src/http/routes/openai/mod.rs
 M crates/server/app/src/http/routes/openai/models.rs
 M crates/server/app/src/http/routes/tests.rs
 M crates/server/app/src/lib.rs
 D crates/server/app/src/lora.rs
 M crates/server/app/src/state.rs
 M crates/support/benchmarks/Cargo.toml
 M crates/support/benchmarks/src/lib.rs
 D crates/support/benchmarks/src/native_events.rs
 M crates/support/benchmarks/src/semantic.rs
 M crates/support/benchmarks/src/synthetic_payloads.rs
 M crates/support/examples/Cargo.toml
 M crates/support/examples/src/lib.rs
 M crates/support/testkit/src/lib.rs
 A qualification/generation-runtime/plans/S02.json
 M qualification/generation-runtime/references.json
 M scripts/check_generation_runtime_contract.py
 M scripts/compare_benchmark_runs.py
 M tests/python/e2e/test_native_sim_http.py
 M tests/python/unit/eval_driver/test_driver.py
 M tests/python/unit/eval_driver/test_family_b_harness.py
 M tests/python/unit/eval_driver/test_image_quality.py
 M uniserve_eval/harness/cli.py
 M uniserve_eval/harness/report.py
 M uniserve_eval/harness/runner.py
 M uniserve_eval/harness/schemas/summary.schema.json
 M uniserve_eval/harness/spec.py
 M uniserve_eval/profiles.json
```

The 44 deletion records correlate with the serving-surface contract's closed public route set and single-funnel ownership. The correlation supports reconstruction planning; candidate-tree inspection remains the authority for the candidate's exact implementation.

## Recovery strategy

### Route A: candidate-object restoration

Route A has priority whenever a dead-node disk image, volume snapshot, Git object pack, repository archive, or object-store backup becomes available. Import the object database into an isolated clone and establish a recovery reference only after Git verifies the object as a commit:

```bash
git fsck --full
git cat-file -t 9687bc949e03ea3267dbf37450aaab2061830001
git cat-file -p 9687bc949e03ea3267dbf37450aaab2061830001
git update-ref refs/recovery/s02-candidate 9687bc949e03ea3267dbf37450aaab2061830001
git worktree add ../uniserve-s02-candidate refs/recovery/s02-candidate
git diff --name-status d538df72d7640f7f19802f830d7ccda8141182dc 9687bc949e03ea3267dbf37450aaab2061830001
```

The recovered commit must report parent `d538df72d7640f7f19802f830d7ccda8141182dc` or provide an explicit parent-chain explanation from the candidate manifest. Compare its affected path set, `S02.json`, source line counts, candidate manifest, artifact digests, and qualification roots against this record. Push the verified object under a recovery reference before further edits.

### Route B: complete content-snapshot restoration

Route B applies when complete file bodies emerge from Cursor content-addressed records, editor backups, filesystem snapshots, or artifact archives. Each recovered body receives a SHA-256 digest, source-record identifier, creation order, and target path. Materialize the 113-path tree in an isolated worktree based on `d538df7`, then create a commit object and compare its tree identifier with `9687bc949e03ea3267dbf37450aaab2061830001`.

Tree equality establishes byte-level restoration. Tree divergence establishes a provenance-preserving partial source recovery whose remaining scope proceeds through Route C.

### Route C: specification-driven S02 reconstruction

Route C creates a new branch from `d538df7`. Commit `8059348` supplies the exact deletion inventory and recovered instructions as forensic evidence; `d538df7` supplies the complete compilable parent tree.

```bash
git switch --create agent/s02-reconstruction d538df72d7640f7f19802f830d7ccda8141182dc
```

The implementation is derived from the current serving-surface and construction contracts. The 113-path snapshot defines the observed candidate scope and supplies a review checklist. Every retained divergence from that path set receives a current ownership or dependency rationale in the S02 candidate plan.

## S02 implementation contract

### Admission values

`crates/frontend/serving/src/input.rs` owns `GenerateReqInput`, `TokenizedGenerateReqInput`, canonical sampling input, public modality selection, image-generation controls, request identity, stream contract, resource bounds, cache bounds, and scheduling bounds. Public adapters construct one `GenerateReqInput` through a fallible conversion. Model-private prompt recipes, token placement, runtime behavior selection, and output filtering reside behind `ResolvedModel::tokenize`.

### Closed model resolution

`crates/frontend/serving/src/model.rs` owns the closed `ResolvedModel` enum with `Qwen3`, `SenseNova`, and `Bagel` variants and the concrete description values `Qwen3Desc`, `SenseNovaDesc`, and `BagelDesc`. Resolution is fallible and exhaustive. Each description binds tokenizer state, identity, limits, stop tokens, modality support, prompt framing, ingest rules, generation defaults, output filtering, and configured parser policy before request admission.

### Model-owned tokenization

`ResolvedModel::tokenize` is an inherent method and the sole model-owned transition from `GenerateReqInput` to `TokenizedGenerateReqInput`. Qwen3 tokenization covers HF chat rendering, tokenization, and Qwen3 parser policy. SenseNova and Bagel tokenization cover multimodal framing, image placement, ingest binding, negative-prompt encoding, image-control normalization, runtime behavior descriptors, and output-filter policy.

### Runtime and gateway ownership

`ServingRuntime` stores one concrete `ResolvedModel` and one engine gateway. Its asynchronous generation operation accepts `GenerateReqInput`, invokes `ResolvedModel::tokenize` once, submits one `TokenizedGenerateReqInput`, and returns an ordered `ServeEvent` stream. `crates/frontend/engine-gateway` exposes the single concrete submission boundary.

### Public adapters and routes

`crates/frontend/protocol-adapters` owns OpenAI request shapes, fallible lowering, public error mapping, and response assembly. `crates/server/app` owns route registration and invokes the serving runtime. The configured public generation surface comprises `POST /v1/chat/completions` and `POST /v1/images/generations`; health, metrics, version, and model-discovery endpoints retain their declared roles.

OpenAI response assembly preserves event order, streaming usage, finish reason, text deltas, reasoning deltas, image artifacts, request identity, and configured parser semantics. The model descriptions own SenseNova and Bagel output filtering over committed events.

### Ownership consolidation

The reconstructed tree contains one owner for each of wire lowering, internal admission, model tokenization, engine submission, and response assembly. The recorded deletions identify prior ownership surfaces across gRPC inference, native/raw generation, structured output, LoRA routing, dialect generation, model-selection renderers, separate text lowering, and duplicate server routes. Current configured behavior is re-derived into the canonical owners named above.

### Qualification control

`qualification/generation-runtime/plans/S02.json` declares the complete affected scope, impact classes, ordered checks, comparators, artifact root, and fast-forward acceptance behavior under `schemas/generation-runtime-candidate-plan.schema.json`. Its artifact root uses `artifacts/qualification/generation_runtime/S02/{candidate}`. Candidate materialization creates an immutable commit object, validates it in an isolated worktree, and preserves the exact candidate and artifact digests.

## Reconstruction work sequence

1. Fix the S02 type and ownership design from the three normative specifications and this evidence record.
2. Implement `input.rs` with the complete public admission and tokenized-engine values.
3. Implement `model.rs` and `omni/mod.rs` with closed model resolution, load-bound tokenizer ownership, and configured multimodal framing.
4. Rebuild `ServingRuntime` around one `generate` path and one gateway submission.
5. Re-derive OpenAI chat and image lowering into `GenerateReqInput` and assemble responses from `ServeEvent`.
6. Re-derive server state, route registration, configuration, and model discovery around one resolved model and one serving runtime.
7. Reconcile model-profile ownership, dependency manifests, support crates, examples, and evaluation request builders with the canonical types.
8. Re-derive tokenize and stream-assembly tests from the current contract and replace tests whose subjects moved to new owners.
9. Reconstruct `S02.json` from the final affected tree, required gates, comparator identities, and canonical artifact paths.
10. Materialize the immutable candidate, execute its declared gates serially, preserve every artifact body and digest, and advance the development reference through the S01 fast-forward acceptance path.

## Acceptance criteria

Source acceptance requires all of the following properties at one immutable candidate revision:

1. Every configured generation route constructs exactly one `GenerateReqInput`.
2. The loaded `ResolvedModel` value owns the only `tokenize` transition.
3. Every admitted request produces exactly one `TokenizedGenerateReqInput` and exactly one engine-gateway submission.
4. Qwen3, SenseNova, and Bagel descriptions own their complete configured tokenization, framing, parser, image-control, ingest, and output-filter policies.
5. Streaming and collected response assembly preserve ordered `ServeEvent` semantics and public usage fields.
6. The source tree assigns one canonical owner to each request-funnel stage and every dependency has a configured consumer.
7. The public schema and route set match [`serving-surface.md`](serving-surface.md).
8. `S02.json` enumerates the final affected scope and the exact C1, public-schema, tokenize, stream-assembly, and performance-equivalence or P1 gates required by [`decode-runtime-construction.md`](decode-runtime-construction.md).
9. Candidate provenance binds parent commit, candidate commit, source tree, executable inputs, profiles, comparators, commands, artifact root, and acceptance record.
10. The accepted candidate commit is reachable from the canonical GitHub remote through a named reference.

The claim attached to Route A or a tree-equal Route B result is byte-level recovery of candidate `9687bc949e03ea3267dbf37450aaab2061830001`. The claim attached to Route C is S02 specification conformance at its newly materialized candidate identifier.
