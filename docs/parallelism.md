# Parallel serving configuration

`--workers` accepts a JSON array of Worker configurations. Each configuration names an `id`, an ordered `ranks` list, named computation `entries`, and a positive `queue_depth`. Each rank declares `node` and `device`; entry membership indexes the ordered ranks of its Worker. Each entry has its own `parallel_config`, and several entries can share a rank. When `--workers` is omitted, `--worker-ranks` determines the default instance size and the model supplies its entry layout. Local process launch uses `node: "localhost"`.

## Product transfer

`--transfer` accepts comma-separated `source[:rank]->destination[:rank]=backend` bindings. Names identify statically configured Workers. Omitting a rank selects every member of that instance. Overlapping bindings are rejected. The two directions are independent, and different physical pairs on one logical edge may use different mechanisms:

```text
--transfer 'encoder:0->denoiser:0=shm,encoder:0->denoiser:1=cuda_ipc'
```

Each rank initializes its incident mechanisms and publishes through its required outbound mechanisms. Worker discovery reports the loaded endpoint incarnation, primary device and initialized backends. Binding validates rank membership, node reachability and backend availability before serving. `local` requires the same address space; `shm` requires one node; `cuda_ipc` requires CUDA devices on one node. A process-local device ordinal is interpreted within its endpoint. Cross-node process launch and transfer are unavailable.

Within one Worker, startup fills unconfigured directed rank pairs from their declared node and device coordinates: `local` for the same rank, `cuda_ipc` between CUDA ranks on one node, and `shm` for other same-node pairs. Explicit bindings take precedence and undergo the same startup validation. Transfers between different Workers require explicit bindings. A selected backend failure is an error; it does not select another mechanism.

CUDA IPC publication requires native, nonexpandable CUDA allocations. The launcher selects this allocation policy for publishing ranks when no allocator configuration is supplied. Explicit `expandable_segments:True` or `backend:cudaMallocAsync` settings are rejected for those ranks. Other ranks retain their normal allocation policy.

Products retain one logical representation and may publish several physical locations. Each consuming rank receives the locations selected by its directed bindings; tensor delivery matches their actual coverage to its reserved destination. A failed selected read reports failure. All backends of one rank share its transfer-byte and read-ticket budgets, and source storage remains retained until every publication and reader retires.

## Tensor parallel models

Qwen3, SenseNova, and BAGEL use a `model` entry. The following configuration runs tensor parallelism on visible CUDA devices 3 and 1, with model rank order `[1,0]`:

```bash
uniserve serve /models/Qwen3-32B \
  --model-description qwen3 \
  --workers '[{"id":"model","ranks":[{"node":"localhost","device":"cuda:3"},{"node":"localhost","device":"cuda:1"}],"entries":{"model":{"ranks":[1,0],"parallel_config":{"tensor_parallel_size":2}}},"queue_depth":2}]'
```

The same component schema applies to `--model-description sensenova` and `--model-description bagel`. Tensor degree must satisfy the checkpoint's head and projection geometry. These models currently require local sequence execution and one pipeline stage. Unsupported sequence, pipeline, and component bindings fail at startup.

## H3 components

Install the H3 dependencies with `uv sync --extra h3`. H3 requires CUDA compute capability 9.0 or later. Sparse attention selects the SM100 native provider on supported SM100 devices and the FlashInfer or Triton providers according to their device capabilities. Native providers require the CUDA 13 toolkit (including `nvcc`) and a C++20 compiler. Set `CUDA_HOME` when the toolkit is outside the standard CUDA installation path. Worker initialization builds and caches these providers before dependent CUDA graph capture; compiler failures are startup errors. The kernel package includes the native sources and their Apache-2.0 licensing.

Checkpoints with `fastvideo_inference.json` supply their trained DMD jump points through that inference contract. The four-step VSA DataFree checkpoint declares `[999, 749, 500, 250]`; the loader applies video/audio shifts of 12 and 3 with the checkpoint scheduler's FP32 arithmetic before preparing timestep modulation. Checkpoints without the sidecar use the uniform `[1000, 750, 500, 250]` grid. The contract's task, guidance and sparse-attention geometry must match the supported H3 computation.

For source checkouts, build the Rust executable and native Python extension against the same IPC schema and worker interpreter. For example, `PYO3_PYTHON=.venv/bin/python cargo build --release -p uniserve -p uniserve-ipc-py --features pyo3/extension-module` builds both artifacts; `uv pip install --python .venv/bin/python --no-build-isolation -e .` installs the checkout and its extension into the worker environment. Rebuild both after protocol changes. An extension linked against another Python ABI cannot be loaded by the worker interpreter.

Sparse ring, hybrid, and Attention2D currently require mutually accessible CUDA peers on one host. The runtime maps ordered peer-owned K/V allocations into one virtual key tensor, allowing attention to consume the selected keys without replicating their physical storage. Attention2D gathers columns and reads peer-owned row segments. Page alignment is physical capacity only; it does not change model padding or sparse selection. Compute providers are selected from the key extent and prefix geometry. Cross-host context transport is not implemented.

H3 declares five components: `denoiser`, `text_encoder`, `video_decoder`, `audio_decoder`, and `output`. Denoiser and encoder membership are independent. Video decoders distribute temporal units in the declared rank order with `distribution: "temporal_units"` and `units_per_rank: 1`. Audio and output each have one owner.

This four-device example places two-stage pipeline denoising on ranks 3 and 1, encoding on rank 0, temporal decoding on ranks 2 and 0, audio decoding on rank 1, and output assembly on rank 2:

```bash
uniserve serve /models/FastVideo-Minimax-FastH3-Preview-v0.2 \
  --model-description minimax-h3 \
  --served-model-name MiniMax-H3 \
  --workers '[{"id":"model","ranks":[{"node":"localhost","device":"cuda:0"},{"node":"localhost","device":"cuda:1"},{"node":"localhost","device":"cuda:2"},{"node":"localhost","device":"cuda:3"}],"entries":{"denoiser":{"ranks":[3,1],"parallel_config":{"pipeline_parallel_size":2}},"text_encoder":{"ranks":[0]},"video_decoder":{"ranks":[2,0],"distribution":"temporal_units","units_per_rank":1},"audio_decoder":{"ranks":[1]},"output":{"ranks":[2]}},"queue_depth":6}]' \
  --dtype bfloat16 --quantization-config '{"mode":"quality"}' \
  --pipeline-depth 6 --max-batch 2 --max-running-requests 2 \
  --max-num-batched-tokens 2 --chunked-prefill-size 1 \
  --max-model-len 16384 --max-video-seconds 15
```

On four GB200s, this binding has completed 5-second/1K-token and 15-second/16,384-token requests, including subsequent requests after three client cancellations. The stated precision resolves to BF16 denoiser attention, BF16 denoiser MLP, BF16 text encoder, and FP16 video VAE. Performance remains unranked.

Omitting configuration for H3 assigns all workers to projected-head Ulysses denoising, direct encoder TP, and temporal video decoding, with audio/output on rank zero. Four ranks therefore select U4, encoder TP4, and four temporal decoders.

## KV head delivery

KV products describe global token, layer and head coordinates. Each rank publishes the head interval written by its attention projections; TP member order determines the interval, and a KV group smaller than the TP size remains fully replicated. Consumers read their required head interval directly into reserved cache pages, allowing different TP head partitions without a host gather.

FP8 publications include per-source-page, per-layer scales with an explicit head-group axis. Resharding across scale groups or page sizes performs an explicit conversion using bounded startup storage and the source compute precision. An incremental import preserves the scale of an already installed destination page. Worker discovery reports both local storage geometry and global head coverage; physical cache capacity remains rank-local.

## Logical degrees and groups

`tensor_parallel_size` and `pipeline_parallel_size` default to one. `sequence_parallel` is a tagged choice whose degree is derived from its fields:

| Choice | Sequence configuration |
| --- | --- |
| Local | `{"kind":"local"}` |
| Ulysses | `{"kind":"ulysses","ulysses_degree":4}` |
| K/V gather | `{"kind":"allgather","allgather_degree":2}` |
| Ring | `{"kind":"ring","ring_degree":2}` |
| Ulysses plus ring | `{"kind":"hybrid","ulysses_degree":2,"ring_degree":2}` |
| Attention2D | `{"kind":"attention2d","attn2d_row_size":2,"attn2d_col_size":2,"ulysses_degree":1}` |

This table defines configuration geometry. An executing model/backend binding and sufficient memory are also required. Distributed operator comparisons cover two/four GPUs, reversed membership, and CUDA graph replay. Complete-media and cancellation/reuse checks exercise public serving. Supported layouts do not imply identical seeded outputs across tensor or sequence degrees, and performance recommendations require matching measurements.

On one four-GB200 host, the `quality`, `balanced`, `performance` and `maximum` presets have each passed complete HTTP serving with Local, Ulysses2/4, AllGather2/4, Ring2/4, Hybrid U2×R2, Attention2D 2×2, TP2×Ulysses2, TP2/4 and PP2/4. All 56 configurations complete 5-second/1K-token and 15-second/16,384-token video/audio requests and three cancellation/reuse cycles. Local uses one GPU; the other points use the component bindings required by their qualification. This is functional evidence, not a perceptual-quality comparison, latency ranking or qualification of arbitrary degree combinations.

H3 uses column-parallel QKVG and gate/up projections and row-parallel attention-output and MLP-down projections. Row projections sum local outputs in their selected output dtype and apply bias once. BF16 denoiser MLP and attention-output projections require FP32 partial reductions and retain BF16 output storage, including when communication releases rows in separate intervals. Other BF16 projections use framework GEMM. FP8, MXFP8 and NVFP4 use their declared quantization methods without a topology-specific precision promotion or fixed accumulation order. FP32 normalization/softmax statistics and solver state retain their model-defined roles. Numerical validation checks the operator's mathematical and quantization contracts and finite, usable generation results; cross-topology SNR, cosine distance and bitwise equality are not acceptance gates.

Projected-head Ulysses execution exchanges bounded row intervals through registered storage. Dense output and feed-forward projections consume each complete head interval while later transfers remain in flight. The FlashInfer sparse provider also produces queries in paired sequence-owner intervals: each interval attends to the same complete selected K/V domain and composes directly into its exchange source views. K/V packing is shared across intervals, while geometry-specific BSR plans retain device-resident mutable block maps. The communication layer owns rank ordering, transport submission and stream dependencies. Tensor-wide quantized projection domains consume all rows together before applying their declared scales.

Within a row-production interval, globally visible prefix queries use FlashInfer dense FA2 prefill over the complete valid key domain. Video queries retain their selected BSR blocks. Prefix/video boundaries follow global logical rows independently of rank order and physical device identity. The fused output composition accepts each provider's output and log-sum-exp strides, removes padded-key softmax mass, and applies trained compression into the original communication destinations. This provider decomposition preserves BF16 operands, FP32 attention statistics, sparse selection and transport granularity; alternate attention kernels can produce different BF16 rounding and seeded media bytes.

When the attention provider releases its compute scratch before row consumption, each completed dense feed-forward interval can also produce the next layer's normalized input. The dense sequence projection uses that registered scratch as a two-slot AllGather pipeline, computes local rows independently, and completes each remote projection before reusing its slot. The full QKVG output is allocated only when its first input interval is published. This cross-layer pipeline preserves the complete selected attention domain and leaves tensor-wide quantization scales under complete-domain execution.

Streamed dense projection publishes each completed interval with global logical row coordinates. H3 applies its existing Q/K RMSNorm and RoPE to those rows, then prepares their pooled tiles and provider input layout while later peer intervals remain in flight. Sparse selection consumes the complete pooled domain. The provider allocates its packed layout at first publication and retains it through fine-attention production; projection completion transfers ownership and releases the row consumer. This moves transient input storage earlier in the execution lifetime, so memory requirements include its overlap with remaining feed-forward and projection work. Whole-input callers use the same layout-packing implementation with a complete row interval.

For model-parallel components, membership size must equal tensor degree × sequence degree × pipeline degree. Every degree is positive and every membership list contains unique ranks within the configuration. H3 currently binds tensor and total sequence degrees of 1, 2, or 4; tensor degree × Ulysses degree must divide 56 heads. Encoder TP independently divides 64 query heads, eight KV heads, and MLP width 25,600. Sparse context ownership does not further divide heads.

Meshes order dimensions as pipeline, context, tensor, and Ulysses, with Ulysses varying fastest. Attention2D replaces context with row and column dimensions. A T2×U2 denoiser on `[0,1,2,3]` consequently has tensor groups `[0,2]` and `[1,3]`, and Ulysses groups `[0,1]` and `[2,3]`. Group roots and peers use positions within these ordered groups.

Workers finalize membership, numerical formats, geometry, and capacity before accepting requests. Changing membership or precision requires a fresh worker initialization. Queue depth controls outstanding commands; admitted request capacity comes from the loaded components and provisioned storage.

## H3 layer pipeline

Set `pipeline_parallel_size` on the denoiser to divide its 50 transformer blocks into balanced contiguous ranges. Stage order follows denoiser rank order, with tensor and sequence coordinates varying inside each stage. The first stage owns input projections and text refinement; the final stage owns output projections and the solver. Updated latents return to the first stage before the next denoising step. Decoder transfers consume unique final-stage row owners.

The configuration above assigns blocks 0–24 to rank 3 and 25–49 to rank 1. Each stage loads only its assigned block weights and timestep products. Pipeline stages cannot exceed the model's layer count, and the full denoiser membership must still match the product of its parallel degrees. PP2 and PP4 have complete-serving evidence across all four precision presets on four GB200s with independently placed components. Other factorizations, including PP×TP/SP compositions, require matching capacity and qualification.
