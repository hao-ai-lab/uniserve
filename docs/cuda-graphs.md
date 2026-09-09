# CUDA Graph execution and memory

UniServe's [`CudaGraphRunner`](../uniserve_worker/execution/cuda_graph.py) owns a bounded set of complete captured forwards for one execution lane. [`ModelRunner`](../uniserve_worker/execution/model_runner.py) also owns fixed-input module captures. Both use `GraphEntry` to own the executable, stable input storage and borrowed outputs. Capture eligibility depends on the model entry, selected attention provider, shape, device and CUDA context.

## Execution coverage

| Path | Capture boundary | Live preparation |
| --- | --- | --- |
| Paged decode | One complete forward for a captured batch-size bucket, including eligible greedy selection | Tokens, positions, page mappings, sequence lengths and native attention plans |
| Paged prefill | One complete forward for a padded token/request geometry bucket | Host row metadata, padding, page mappings and native attention plans |
| Packed flow or mixed decode/flow | Complete forward only when the selected provider supports packed capture | Exact flow/mixed geometry and supported attention metadata |
| H3 video decoder | Complete fixed-shape decoder module on each owning rank | Copy the current temporal unit into the captured input |
| H3 denoiser, text encoder and audio decoder | Eager module execution | Component-specific inputs, communication and execution state |

The FlashInfer packed provider currently uses eager flow/mixed execution. Paged decode/prefill capability is independent of packed capability; successful warmup of a flow path alone does not imply capture. H3 captures video decoder input shape `(1, 24, 7, 48, 84)` in FP32 storage. Longer clips distribute temporal units across the configured decoder owners and reuse those module captures. This boundary excludes the HTTP request, denoising loop, temporal assembly and media encoding.

Under the terminology in [Advanced CUDA Graph Techniques in SGLang](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/), a complete captured forward with preparation before replay is a full CUDA Graph. BCG instead executes an ordered sequence of captured segments and explicit eager regions. UniServe shares executable ownership through `GraphEntry`, but currently has no BCG backend or unified runner/backend strategy interface. Unsupported provider/shape domains follow the declared eager execution path.

## Mutable metadata and ordering

Before capture, the runner binds an attention wrapper to the static batch and prepares its plan. A captured decode forward consumes that bound plan; it must not capture a second scheduling upload containing capture-time lengths. Before every replay, the runner refreshes live page mappings, lengths and native scheduling data, then enqueues replay on the owning stream.

Host row lengths already held by the execution producer supply FlashInfer planning inputs. Native scheduling uploads own immutable pinned-host generations until DMA completion through PyTorch allocator stream tracking. Device workspace addresses remain stable. These ownership rules permit asynchronous planning without reading device lengths back merely to reconstruct host metadata. See [KV cache management](kv-cache-management.md#forward-execution).

Captured tensors are borrowed storage. A consumer must finish using them before another replay or shared-pool entry can overwrite their addresses. Packed output publication copies live rows into caller-owned storage; tensors with a common device and dtype share that allocation. Module callers finish consuming borrowed output before reusing the module capture. Cross-stream consumers retain their explicit event ordering and allocator lifetime registration.

## Memory accounting

Paged graph buckets in one lane share a private pool under serialized execution. Independent module captures own separate pools. The configured graph-memory budget and startup catalog constrain residency; request shape diversity does not add arbitrary captures after startup.

Private-pool reserved bytes include blocks retained for addresses used by captured kernels. An inactive block in an allocator snapshot can therefore remain necessary for replay. Removing that reservation requires retiring or recapturing its owner, or changing capture lifetimes; `empty_cache()` is not a replacement for that ownership change.

Persistent products backed by a component's provisioned storage are counted once. Publication/read tickets retain the source storage through consumer completion without reserving a second physical copy of the same payload. KV capacity, activation scratch, peer mappings, private Graph pools and external CUDA allocations remain separate memory consumers.

Comparisons with another engine must distinguish live tensor allocation, allocator reservation and device-level NVML usage, under matching model weights, precision, parallelism, request/KV capacities, graph shapes and sampling. Each H3 video decoder rank owns weights because it executes assigned temporal units. The presence of these weights on two ranks alone does not establish redundant execution or removable storage. Current local accounting establishes ownership; a matched SGLang/vLLM/vLLM-Omni memory ratio and overall Speed of Light qualification remain open.
