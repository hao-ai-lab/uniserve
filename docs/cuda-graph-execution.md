# CUDA Graph Code-Reading Roadmap

## Destination

Use this roadmap to learn UniServe's CUDA graph implementation by following one small execution path before studying its variants. The first target is a single-token text decode. Prefill, mixed text, model-owned text, packed multimodal execution, denoising, scheduling, and policy come later.

You are ready to leave the roadmap when you can explain these questions from the code:

- Which tensor addresses must remain stable after capture?
- Which values may change between replays, and where are they copied?
- What selects a physical graph state for a request?
- What work happens before capture, inside the captured closure, before every replay, and after replay?
- Why does a mixed extend-plus-decode text batch use the prefill graph runner?
- What is the difference between a logical `CapturedForwardGraph` and a physical `torch.cuda.CUDAGraph`?

## Keep this mental model beside the code

A CUDA graph records a fixed sequence of GPU work once and launches that sequence again with `replay()`. Replay does not rerun the Python that originally built the sequence. Tensor addresses, shapes, and GPU control flow therefore have to remain compatible with capture.

UniServe handles changing requests with this cycle:

1. Select a captured capacity bucket, or create one if it does not exist.
2. Copy current request values into graph-owned tensors whose addresses stay stable.
3. Refresh mutable attention planning that must not remain frozen at capture-time values.
4. Replay the captured numerical work.
5. Slice away bucket padding and commit request state outside the graph.

On the first pass, classify every object or operation as one of four things: stable graph storage, dynamic request data, captured GPU work, or out-of-graph state management. That classification is more useful than memorizing class names.

## Reading rules

- Read only the named symbols, not each file from top to bottom.
- Start at the physical graph runner and work outward. The scheduler is the last stop, not the first.
- Ignore metrics, logging, host-staging optimizations, warmup enumeration, and detailed error classification until the basic replay path is clear.
- Keep a one-page trace with columns for `value`, `owner`, `shape/address stability`, and `when refreshed`.
- After each pass, answer its checkpoint without looking at this document. If the answer is unclear, repeat that pass before adding another execution case.

## Pass 0: learn the data vocabulary

Read these definitions only far enough to recognize the values passed into the graph code:

1. [`ForwardMode`](../uniserve_worker/contracts/forward_mode.py#L16) distinguishes decode, extend, mixed, generation, and denoise work.
2. [`ForwardBatch`](../uniserve_worker/contracts/forward_batch.py#L183) is the model-facing batch and carries attention metadata.
3. [`TextAttentionMetadata`](../uniserve_worker/contracts/forward_context.py#L93) describes the current text attention plan.
4. [`PagedKVPool`](../uniserve_worker/runtime/kv_pool.py#L37) owns KV storage, while [`BatchedPagedRequestCache`](../uniserve_worker/runtime/paged_text_cache.py#L481) maps batch rows to pages and sequence lengths.

Do not study the complete planner or cache implementation yet. The only facts needed are that decode consumes one new token per row, prior keys and values live in a paged cache, and the page mapping and sequence lengths can change while the graph's tensor addresses cannot.

Checkpoint: Given `input_ids`, `positions`, a block table, and sequence lengths, state which values change on the next decode step even when the batch capacity is unchanged.

## Pass 1: understand the shared physical lifecycle

Open [`_GraphRunnerBase`](../uniserve_worker/execution/forward/graph/base.py#L176-L375) and read two methods in this order:

1. `_capture_or_replay` is the per-call state machine. Trace the cache lookup, missing-state capture, input copy, optional attention preparation, replay, event recording, and non-fatal miss path.
2. `_capture_graph_state` performs physical capture. Trace the warmup stream, warmup iterations, final input refresh, `torch.cuda.graph(...)` context, and captured output buffer.

Write the main path in your own words before moving on:

```text
key -> states[key]?
     -> capture stable state if absent
     -> copy live values into stable inputs
     -> prepare mutable backend state
     -> state.graph.replay()
     -> return a view of stable outputs
```

Ignore `_warmup_capture_buckets`, graph retirement, shared allocation details, metrics, and failure classification on this pass.

Checkpoint: Explain why `copy_inputs` is called during warmup, immediately before capture, and again before every replay.

## Pass 2: trace one physical decode graph

Use [`text_decode.py`](../uniserve_worker/execution/forward/graph/text_decode.py) as the concrete implementation of the lifecycle. Read its symbols in this order:

1. [`TextDecodeGraphState`](../uniserve_worker/execution/forward/graph/text_decode.py#L48) lists everything kept alive for a physical graph.
2. `DecodeCudaGraphRunner.bucket_batch_size` and `resolve_bucket` show how a live batch maps to a reusable capacity.
3. [`DecodeCudaGraphRunner.maybe_run`](../uniserve_worker/execution/forward/graph/text_decode.py#L352) supplies the concrete callbacks to `_capture_or_replay`.
4. `DecodeCudaGraphRunner.capture` and `_capture_decode_graph` allocate state and delegate physical capture to the base class.
5. [`copy_text_decode_graph_inputs`](../uniserve_worker/execution/forward/graph/text_decode.py#L653) updates current token IDs, positions, page tables, and lengths without replacing captured tensors.
6. [`_replay_decode_graph`](../uniserve_worker/execution/forward/graph/text_decode.py#L555) launches the graph and slices the bucket-sized logits to the live batch size.
7. [`resolve_paged_decode_graph_prepare`](../uniserve_worker/execution/forward/graph/text_decode.py#L1169) locates the backend hook that refreshes attention planning before replay.

For the first trace, assume a live batch of three rows reuses a bucket of four. Draw the shapes of the stable inputs, identify the padded row, and follow the returned `state.logits[:batch_size]` view.

Ignore `TextDecodeGraphHostInputs`, `maybe_run_host_inputs`, warmup, token replacement, shared buffer pooling, and backend-specific branches until this trace is complete.

Checkpoint: Point to the exact line where a live value is copied instead of a new tensor being installed, and the exact line where the physical graph launches.

## Pass 3: connect the physical runner to the model

Read [`TextGraphRunner`](../uniserve_worker/execution/forward/graph/text.py#L40-L231), concentrating on `maybe_run`, `_maybe_decode`, and the small forward closure passed into the decode runner.

Follow these questions through the code:

1. What eligibility checks return `None` before physical capture is attempted?
2. How are `model`, `input_ids`, `positions`, `ForwardBatch`, and `ForwardContext` converted into the callback accepted by `DecodeCudaGraphRunner`?
3. Why must a paged-decode backend provide a preparation hook?
4. Where does the model remain graph-unaware?

Then read [`TextDriver.forward_logits_graph`](../uniserve_worker/execution/text_driver.py#L262), [`TextDriver.forward_graph_result`](../uniserve_worker/execution/text_driver.py#L317), and `_forward_graph_with_optional_padding_reorder`. Stop when you can see how request records become model inputs and how a graph miss propagates as `None` rather than silently running eager code inside the graph-only call.

Checkpoint: Trace one decode request from `TextDriver.forward_graph_result` to `state.graph.replay()` and back to a `ForwardResult`. Write down every function boundary in order.

## Pass 4: add the outer logical graph layer

Only after the physical decode trace is clear, read the general execution framework in this order:

1. [`ModelRunner._build_text_execution`](../uniserve_worker/execution/runner.py#L396-L446) constructs the system-owned text graph runner and performs startup warmup.
2. [`ForwardExecutor.execute`](../uniserve_worker/execution/forward/executor.py#L38-L78) applies graph preference, strictness, capture permission, and eager fallback.
3. [`CudaGraphForwardRunner.run`](../uniserve_worker/execution/forward/graph/runner.py#L19-L123) selects a logical program and caches logical shape coverage.
4. [`CapturedForwardGraph` and the text programs](../uniserve_worker/execution/forward/graph/programs.py#L30-L230) adapt logical decode or prefill plans to the text driver.
5. [`ForwardStepExecutor._run_group`](../uniserve_worker/execution/forward/step.py#L176) shows the surrounding worker execution context.

There are now two caches to keep separate:

| Layer | Cached object | Main key | Owns `torch.cuda.CUDAGraph`? |
|---|---|---|---|
| Logical forward layer | `CapturedForwardGraph` | `ForwardGraphShapeKey` | No |
| Physical text layer | `TextDecodeGraphState` or `TextInitialPrefillGraphState` | Capacity bucket or physical topology key | Yes |

The word `capture` appears in both layers. In `CudaGraphForwardRunner`, it can mean that a logical program has established coverage. In `DecodeCudaGraphRunner`, it reaches an actual `torch.cuda.graph(...)` context.

Checkpoint: Explain how a new logical shape can be observed while the physical decode runner still reuses an existing bucket.

## Pass 5: add prefill, then mixed text

Read [`TextInitialPrefillGraphState` and `PrefillCudaGraphRunner`](../uniserve_worker/execution/forward/graph/text_prefill.py#L35-L393) by mapping each concept back to the decode runner. Focus on token-count and row-count buckets, flattened variable-length tokens, stable metadata, capture, input copying, backend preparation, replay, and removal of padding.

Next return to [`TextGraphRunner.maybe_run`](../uniserve_worker/execution/forward/graph/text.py#L81-L103). Its central routing rule is:

```text
DECODE          -> decode graph runner
EXTEND or MIXED -> prefill graph runner
```

A mixed text batch here means extend rows and one-token decode rows sharing one flattened variable-length attention layout. A decode row is equivalent to an extend row with one new token and a nonzero cached prefix, so the prefill topology can represent both row kinds. Read `reorder_mixed_for_padding` only after that statement makes sense.

Use [`test_text_graph_runner_routes_mixed_extend_decode_to_prefill_runner`](../tests/python/integration/runtime/test_cuda_graph_replay.py#L680) and the neighboring padding tests as executable examples. Only then inspect the scheduler's [`mixed_prefill_tokens` assembly path](../crates/engine/scheduler/src/scheduler.rs#L2763-L2897) to learn how those rows are admitted into one batch.

Checkpoint: Explain why mixed text does not alternate between a decode graph and a prefill graph, and why bucket padding must preserve per-row context and sampling positions.

## Pass 6: study the remaining execution families

Treat each family as a variation of the same questions: What is the stable state? What is the key? What is copied? What is captured? What is prepared before replay? What is committed afterward?

| Case | Start here | Read after that | Main difference from system-owned text |
|---|---|---|---|
| Model-owned interleaved text | [`InterleavedTextPrefillGraphRunner`](../uniserve_worker/execution/forward/graph/interleaved_text.py#L190) and [`InterleavedTextDecodeGraphRunner`](../uniserve_worker/execution/forward/graph/interleaved_text.py#L421) | [`interleaved_text_stepper.py`](../uniserve_worker/execution/interleaved_text_stepper.py) | The model-facing adapter owns extra cache and commit behavior; learn homogeneous prefill and decode paths before considering combinations. |
| Packed text plus denoise | [`PackedMixedForward`](../uniserve_worker/execution/forward/programs/packed_visible.py#L215) | [`PackedMixedGraphRunner`](../uniserve_worker/execution/forward/graph/packed_visible.py#L73) and [`stream.py`](../uniserve_worker/execution/forward/stream.py) | Text and denoise segments share one packed decoder launch, so row layout, modality visibility, cache promotion, and output scatter are part of the topology. |
| Pure denoise | [`DenoiseStepGraphRunner`](../uniserve_worker/execution/forward/graph/denoise_step.py#L118) | Its state, row partitioning, replay, and post-replay update helpers in the same file | The changing inputs and authoritative updates are latent and denoise state rather than token and text KV state. |

Do not use the packed path to learn CUDA graphs for the first time. It combines the physical lifecycle with the most complicated batch geometry and state-commit rules in the repository.

Checkpoint: For each family, make a five-row table containing physical owner, key, stable inputs, dynamic copies, and post-replay commit.

## Pass 7: return for policy, warmup, and failures

Once all normal paths are recognizable, revisit these cross-cutting mechanisms:

1. [`ForwardGraphPolicy`](../uniserve_worker/execution/forward/fallback.py#L35) and [`ForwardExecutor`](../uniserve_worker/execution/forward/executor.py) for graph preference, strict mode, capture permission, and fallback.
2. [`_GraphRunnerBase._warmup_capture_buckets`](../uniserve_worker/execution/forward/graph/base.py#L344) plus each runner's `warmup` method for startup capture ordering and synthetic inputs.
3. [`fallback.py`](../uniserve_worker/execution/forward/fallback.py) and `disable` methods for the distinction between a recoverable graph miss and a fatal CUDA error.
4. [`stats.py`](../uniserve_worker/execution/forward/graph/stats.py) and the graph events in [`base.py`](../uniserve_worker/execution/forward/graph/base.py#L59) for observability.
5. [`runtime_config.py`](../uniserve_worker/foundation/runtime_config.py) for the feature switches and bucket configuration that instantiate these policies.

Checkpoint: Predict the result of a missing graph state under each combination of capture allowed or disallowed and strict mode enabled or disabled.

## Tests as the final reading pass

Read tests after their implementation, not before it. Use this order:

1. [`test_cuda_graph_replay.py`](../tests/python/integration/runtime/test_cuda_graph_replay.py) for physical text capture, replay, padding, mixed routing, and miss behavior.
2. [`test_unified_forward.py`](../tests/python/unit/execution/forward/test_unified_forward.py) for logical program selection, shape keys, strictness, and fallback.
3. [`test_interleaved_text_graph_runner.py`](../tests/python/unit/execution/test_interleaved_text_graph_runner.py) for model-owned text adapters.
4. [`test_packed_mixed_graph.py`](../tests/python/unit/execution/test_packed_mixed_graph.py) for packed multimodal topology and promotion rules.
5. [`test_denoise_step_graph.py`](../tests/python/integration/runtime/test_denoise_step_graph.py) for denoise partitioning and replay.

For each test, first predict whether the expected event is capture, replay, miss, fallback, or failure. Then run only that test and compare the assertion with the implementation branch you traced.

## A practical stopping exercise

Choose one decode test and annotate its runtime path with these eight labels:

```text
request state
logical plan and key
physical bucket
stable graph state
dynamic input copy
attention preparation
physical replay
post-replay result or state commit
```

Then repeat the annotation for the mixed-text routing test. If both traces are complete and you can explain why their physical runners differ, you have enough context to investigate any of the specialized graph families directly from code.
