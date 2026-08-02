# Serving surface specification

## Status and scope

This specification defines the configured UniServe public inference surface, request funnel, model-resolution boundary, and capability closure. The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative.

The serving surface contains OpenAI-compatible chat and image generation for the configured model set, health and metrics endpoints, and one internal generate admission path. Runtime execution, topology, and process ownership after engine admission are defined by [`decode-runtime.md`](decode-runtime.md) and the route capability bound at server startup.

## Configured model set

The server resolves exactly one of these model descriptions at load time:

| Model description | Configured behavior |
| --- | --- |
| `Qwen3Desc` | Text chat, HF tokenization and chat template, Qwen3 reasoning parsing, and the configured Qwen3 tool parser. |
| `SenseNovaDesc` | Image input, text output, image output, and repeated text/image interleave through description-owned framing, ingest, generation controls, and output filtering. |
| `BagelDesc` | Image input, text output, and image output through description-owned framing, ingest, generation controls, and output filtering. |

Model resolution MUST produce the closed value:

```rust
enum ResolvedModel {
    Qwen3(Qwen3Desc),
    SenseNova(SenseNovaDesc),
    Bagel(BagelDesc),
}
```

Resolution MUST be fallible and exhaustive. The configured set MUST use no name registry, automatic family router, plugin factory, or trait-object tower in front of `ResolvedModel`. A reusable HF tokenizer implementation MAY be shared, but its configured instance MUST be bound into the resolved description before request admission.

## Public HTTP surface

The server exposes these endpoints:

| Method and path | Contract |
| --- | --- |
| `GET /health` | Process and route readiness. |
| `GET /metrics` | Runtime metrics defined by the generation runtime. |
| `GET /version` | Build and protocol provenance when operational metadata is enabled. |
| `GET /v1/models` | The single configured served-model identity and capabilities. |
| `POST /v1/chat/completions` | Streaming and non-streaming text, image-input, image-output, and interleaved generation admitted by the resolved model. |
| `POST /v1/images/generations` | Image generation adapter for configured omni descriptions. |

The public schema contains chat messages, stream controls, modalities, supported image inputs, sampling controls, stop controls, and image-generation controls. It does not contain a second text-completions request class, native/raw generate request, gRPC inference request, runtime reset/load request, adapter selection, structured-output request, or grammar request.

## One generate funnel

Every public generation request follows one ownership chain:

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

`crates/frontend/protocol-adapters` owns OpenAI request and response shapes. `crates/frontend/serving` owns `GenerateReqInput`, `TokenizedGenerateReqInput`, `ResolvedModel`, and `ServingRuntime`. The engine owns admitted request execution and emits model-independent `ServeEvent` values.

Wire lowering MUST construct exactly one `GenerateReqInput`. It SHOULD use `TryFrom` or an equivalent fallible constructor with no model execution authority. The wire request MAY remain available to response assembly but MUST NOT become a parallel runtime request object.

`ResolvedModel::tokenize` MUST be the only model-owned arrow between `GenerateReqInput` and `TokenizedGenerateReqInput`. That arrow MUST be an inherent method rather than a `From`, `TryFrom`, or `Into` implementation because it depends on the load-bound model description.

`ServingRuntime` MUST hold one concrete `ResolvedModel` and one engine gateway and expose an asynchronous generation method equivalent to:

```rust
impl ServingRuntime {
    pub async fn generate(
        &self,
        request: GenerateReqInput,
    ) -> Result<impl Stream<Item = Result<ServeEvent>>>;
}
```

## `GenerateReqInput`

`GenerateReqInput` is the sole internal generate-class admission value. It contains:

- Request identity, streaming mode, and public output contract.
- Text or messages and supported image inputs.
- The single canonical sampling configuration.
- Stop-token, stop-string, minimum-token, bad-word, allowed-token, and logit-bias controls supported by the configured sampler capability.
- Cache and scheduling bounds required for admission.
- `modalities`, which selects text output, image output, or an admitted combination.
- Optional `negative_text` for classifier-free guidance.
- Optional `image_gen` parameters containing dimensions, denoise steps, guidance controls, seed, and image-count bounds.

The value MUST NOT contain grammar, structured-output, adapter, drafter, disaggregated-transfer, or preemption controls. Model-private prompt recipes, token placements, generation policies, and output filters MUST NOT appear as public variants on this type.

## `TokenizedGenerateReqInput`

`ResolvedModel::tokenize` produces `TokenizedGenerateReqInput` as the unit submitted to the engine. It contains token IDs, normalized multimodal payloads, the canonical sampling configuration, resolved modalities, negative-prompt tokens when present, resolved image-generation controls, engine behavior descriptors, and finite resource bounds.

For `Qwen3Desc`, `tokenize` performs configured HF chat rendering and tokenization and attaches Qwen3 parser policy. For `SenseNovaDesc` and `BagelDesc`, `tokenize` owns model-private framing, image placement, ingest binding, negative-prompt encoding, image-control normalization, runtime behavior declaration, and output-filter policy.

Model-private lowering MUST complete inside `tokenize`. The serving stack MUST NOT expose staged prepared-request products, prompt-recipe algebras, dialect registries, or a parallel omni request hierarchy.

## Sampling surface

The canonical sampling order and execution contract are defined by [`decode-runtime.md`](decode-runtime.md). Public admission MAY expose a processor only when the configured route capability declares it and the worker provides the same semantics for ordinary sampling and every configured verification path.

The target configured sampler surface includes greedy selection, temperature, top-k, top-p, min-p, typical sampling, repetition penalty, frequency penalty, presence penalty, logit bias, allowed-token masks, bad-word state, forced-token state, minimum-token floors, requested logprobs, stop-token IDs, EOS, and stop strings. Unsupported combinations MUST fail deterministically before operation registration.

Stop strings and custom CPU processors use the shared bounded CPU-continuation facility. They do not define a second request path.

## Configured runtime capabilities

Capabilities are declared by the resolved model route and validated against worker lowering at startup.

| Capability | Configured state |
| --- | --- |
| Ordinary Token Extend and Decode | Configured for every text-producing route. |
| Vision or latent Encode | Configured only for descriptions that consume those products. |
| Gen Transition, Gen Flow, and Materialize | Configured only for image-producing routes. |
| Product, KV Publish, and KV Install transfer | Configured for colocated tensor-parallel and model-stage routes that declare them. |
| Device-continuous stochastic and processor-bearing generation | Configured when the route declares exact sampler coverage and an unresolved window greater than one. |
| Repeated text/image interleave | Configured for SenseNova routes that declare every transition edge and resource bound. |
| Tensorized mixed Und/Gen execution | Configured only for exact row combinations that pass the mixed-execution oracle. |
| Draft and Token Verify | Closed runtime leaves with worker depth-one behavior; not configured or advertised without a production device drafter. |
| Request preemption and replay | Not configured or advertised by the current model routes. |
| Cross-node disaggregated execution | Not configured or advertised. |

Protocol leaves that are not configured MUST remain absent from public model capabilities and admission decisions. Internal tests of a closed leaf do not change the serving surface.

## Model-description ownership

Each description owns its identity, context limits, stop tokens, modality support, tokenizer binding, prompt framing, generation defaults, ingest rules, resolution policy, and model-specific output filtering. Qwen3 reasoning and tool parser construction is owned by `Qwen3Desc` and uses fixed configured types.

Shared ownership remains outside descriptions:

- OpenAI wire lowering and response assembly.
- The HF tokenizer implementation used by bound descriptions.
- Admission, scheduling, sampling, cache, transfer, and worker execution.
- Generic stop-string and CPU-processor continuations.
- Runtime lifecycle traces and public metrics.

The codebase MUST contain no shared model-name recipe enum, string-keyed renderer or parser factory, or tokenizer-backend selection for configured request paths.

## Capability closure

A route capability declaration MUST include all fields required by [`decode-runtime.md`](decode-runtime.md): supported work leaves, unresolved-window depth, operation and point bounds, sampler processor coverage, RNG layouts, graph and attention predicates, actual-length and append-offset ownership, sampling rank, topology, Gen conditioning, mixed-row combinations, transport kinds, snapshot scope, and credit maxima.

Startup MUST prove that every advertised work leaf lowers to one depth-one physical route and that every public sampling field is covered by the exact processor bitset. Request admission MUST reject fields or modality combinations outside that declaration. Runtime fallback MAY occur only within a declared mathematically and operationally equivalent provider set that satisfies the same numerical, zero-blocking, and resource contract.

## Transport and topology

Configured transport is bounded, event-driven, and colocated with the serving topology. Tensor-parallel product exchange and declared local model-stage transfer use runtime tickets and exact product identity. Administrative snapshot export and restore are separate operations outside steady-state request execution.

One public request never selects a transport implementation, topology, model stage, sampling rank, or recovery mode. Those choices are fixed by the resolved server route and its capability digest.

## Output assembly

The engine emits ordered `ServeEvent` values with exact semantic roots and public-commit timestamps. OpenAI response assembly preserves event order, streaming usage, finish reason, text and reasoning deltas, image artifacts, and configured Qwen3 parser semantics.

SenseNova and Bagel output filters are description-owned and operate on committed events. They MUST NOT create a second semantic cursor or alter the runtime's operation identity, commit order, or resource accounting.

## Repository ownership

| Area | Canonical responsibility |
| --- | --- |
| `crates/frontend/protocol-adapters` | OpenAI wire types, fallible request lowering, and response assembly. |
| `crates/frontend/serving` | Generate values, resolved descriptions, runtime facade, tokenizer binding, and event stream. |
| `crates/frontend/model-profile` | Closed model-profile data and fallible resolution into the three descriptions. |
| `crates/frontend/engine-gateway` | One concrete submit boundary into the configured engine. |
| `crates/engine` and `uniserve_worker` | Operation protocol, scheduling, execution, products, completion, controls, and metrics. |

An abstraction with one configured implementation MUST have a system responsibility beyond implementation selection. Conversion-only layers, single-entry registries, and ownership duplicates are non-conformant.

## Verification

Serving-surface conformance requires:

1. Schema and route tests for every configured endpoint, including streaming, cancellation, usage, text, image input, image output, and SenseNova interleave.
2. Request-funnel tests proving one `GenerateReqInput`, one description-owned `tokenize` call, one gateway submission, and ordered response assembly for each configured model behavior.
3. Capability tests proving deterministic admission or rejection from the exact route declaration.
4. Model-description fixtures derived from the configured tokenizer, framing, ingest, parser, image-control, and output-filter contracts.
5. Runtime tests for retained sampling processors, stop strings, CPU continuations, colocated transfer, tensor parallel execution, and every advertised modality edge.
6. Workspace ownership review showing that every serving type and dependency has a configured consumer and one canonical owner.
7. The candidate and production qualification defined by [`generation-runtime-qualification.md`](generation-runtime-qualification.md).

Tests MUST assert configured behavior and positive ownership properties. They MUST be replaced with the owning artifact when that artifact is redesigned.
