# Parallel execution

Qwen3, SenseNova and BAGEL use the same tensor, sequence and pipeline parallel execution boundaries. Model definitions supply their layer mathematics, routing, positional embeddings and checkpoint names. The runtime owns device groups, communication, cache regions, output publication and physical retirement.

## Configure model parallelism

`--workers` names a deployment configuration: the participating devices and the parallel configuration of each component placed on them. This four-device example combines TP2 with Ulysses SP2.

Save it as `workers.json`:

```json
[
  {
    "id": "model",
    "ranks": [
      {"node": "localhost", "device": "cuda:0"},
      {"node": "localhost", "device": "cuda:1"},
      {"node": "localhost", "device": "cuda:2"},
      {"node": "localhost", "device": "cuda:3"}
    ],
    "components": {
      "model": {
        "ranks": [0, 1, 2, 3],
        "parallel_config": {
          "tensor_parallel_size": 2,
          "pipeline_parallel_size": 1,
          "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2}
        }
      }
    },
    "queue_depth": 2
  }
]
```

```bash
uniserve serve /workspace/models/Qwen3-32B \
  --workers workers.json
```

The same `model` component configuration applies to SenseNova and BAGEL with their checkpoint paths and model descriptions. The product of tensor, pipeline and sequence degrees must match the component's rank count. Attention heads must admit the declared partition. Pipeline stages own nonempty layer ranges; layer counts need not divide evenly between stages.

A deployment may list several workers, each with its own ranks and queue depth. Repeating a component name in several workers declares interchangeable replicas of the same numerical component; their output contracts must match. The scheduler binds a request to one replica for its lifetime, preserves co-location with other components in that worker when possible, and owns request rows, product buffers, queue capacity, and execution lanes per physical worker. A component named by only one worker remains a shared stage through which every route passes.

A rank's `device` is a CUDA device for a worker that computes, or `"cpu"` for a host worker: a worker whose ranks run on the host and hold only host components, such as FastH3's `video_encoder` and `muxer`. Several `"cpu"` ranks may share a node. Products move between workers over the mechanisms their coordinates admit: device products between CUDA ranks over VMM handles, host products over shared memory on one node and over the rank channel across nodes; `--transfer` overrides an edge explicitly. A component with `"distribution": "temporal_units"` divides its work by media unit over its ranks, `units_per_rank` units each per round.

`memory_fraction` optionally gives one worker process its per-rank share of device memory. Set it when independent workers share a GPU; their fractions should leave aggregate headroom for CUDA contexts and communication workspaces. If it is absent, the process uses the global `--mem-fraction-static` value.

For PP2 on two devices, use `pipeline_parallel_size: 2`, `tensor_parallel_size: 1`, and omit `sequence_parallel`. Each stage loads its assigned global checkpoint layers. Embedding and prediction modules follow their stage ownership, and KV storage records each stage's logical layer and head region. An incompatible layout is rejected before serving.

Sequence parallelism partitions packed input rows and exchanges attention heads. It replicates a stage's decoder weights unless combined with tensor or pipeline parallelism. Short decode requests can incur more communication without enough computation to offset it; select topology using the intended workload.

MiniMax H3 has multiple components for conditioning, denoising and media decoding. Its default placement and explicit component configuration are described in the [FastH3 guide](fast_h3/fast_h3.md).

## Shared execution capabilities

| Capability | Applicability |
| --- | --- |
| Distributed vocabulary selection | Text outputs from Qwen3, SenseNova and BAGEL; hidden and continuous outputs retain their own semantics. Sampling features that require full logits materialize them through the same output interface. |
| Incremental row execution | Layer calls whose numerical dependencies permit independent row intervals. Completed attention exchanges can feed output equations and the next layer's projection. Small payloads use a complete exchange. |
| Quantized row projection | Splits respect row, block or tensor scale requirements. Tensor-wide dependencies use complete inputs and the relevant parallel reduction groups. |
| Attention and collective selection | Providers are selected from the device, dtype, layout, mask, topology and execution domain. Model names and the presence of KV storage are not selection criteria. |
| Product storage | Physical capacity follows output geometry, placement, consumers and outstanding execution. A product remains live until its physical readers finish, including cancellation and cross-device transfer. |

Attention transfer capacity is allocated before graph capture and variable memory pools. Serialized layers and shape buckets reuse the same buffers; independent execution lanes own disjoint storage. The current attention output exchange and the following layer's projected inputs use separate regions. Registered memory enables supported NCCL collective algorithms without changing the numerical provider or logical row ownership.

Graph execution requires a provider that supports the selected attention mode. FlashInfer performs its paged and segmented planning before capture and binds fixed-address numerical inputs for replay; FA4 uses the same explicit sequence and visibility contracts on supported devices. Cached prefix lengths and the lengths after adding current tokens are separate domains; each attention segment is planned from its own boundaries.

Product release and request retirement have different scopes. Releasing a completed image or latent publication does not wait for unrelated KV computation belonging to the same request. Reusing physical pages still waits for every computation or transfer that can access those pages.

The physical verification matrix covers GB200 with two and four participating devices, including combined TP/SP/PP layouts and graph replay. Larger rank counts and Hopper or SM120 devices require separate physical verification; capability-based selection alone does not establish performance on those systems.

Floating-point equivalence is evaluated with dtype-appropriate error bounds and model quality. Implementations may fuse calls and choose different reduction orders; no model requires bitwise reproduction of a particular provider or GPU topology. Data transport and integer control metadata retain their exact contracts.

Startup prepares homogeneous text and diffusion calls on their configured execution lanes. Numerical conformance uses independent references and appropriate numerical tolerances; small score changes near a tie may change greedy selection.

## Worker computation resources

A worker's logical domains (`decode`, `prefill`, and `flow`) resolve to ComponentBinding values during construction. Without `--lane`, components use full-device execution streams; independent components can progress concurrently. An explicit lane configuration creates a CUDAStream with a Green Context and SM quota. Domains assigned to the same stream share compatible InputBuffers and graph storage; separate streams have independent mutable computation storage and NCCL communicators. The explicit-stream NCCL provider keeps communication kernels inside the assigned Green Context during eager execution and capture. ModelRunner owns numerical grouping and synchronization for each actual forward call while preserving result alignment and completion boundaries.

The following server options configure a shared 152-SM binding or independent 64/88-SM bindings on a device supporting those quotas:

```bash
--lane '{"lane_id":"compute","sm_budget":152,"domains":["decode","prefill","flow"]}'

--lane '{"lane_id":"decode","sm_budget":64,"domains":["decode"]}' \
--lane '{"lane_id":"compute","sm_budget":88,"domains":["prefill","flow"]}'
```

The quotas are explicit configuration, validated against the target device at startup. Unsupported domains, overlapping assignments, or invalid quotas fail construction. CUDAStream resources remain fixed until the worker closes. Warmup, capture, replay, and eager execution all use the configured bindings; graph policy does not change SM allocation. Use `--graph-policy full --prefill-cuda-graph true` for required capture, or `--graph-policy off --prefill-cuda-graph false` for eager execution. A required graph cannot silently fall back to eager execution.

The [Worker lifecycle](worker-lifecycle.md) describes endpoint ownership, optional direct-execution warmup, and the scope that releases graphs and staging before CUDAStream resources.
