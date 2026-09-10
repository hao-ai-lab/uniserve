# Parallel execution

Qwen3, SenseNova and BAGEL use the same tensor, sequence and pipeline parallel execution boundaries. Model definitions supply their layer mathematics, routing, positional embeddings and checkpoint names. The runtime owns device groups, communication, cache regions, output publication and physical retirement.

## Configure a packed model

Use `--workers` to declare participating devices and the parallel configuration of the `model` entry. This four-device example combines TP2 with Ulysses SP2:

Save the placement as `workers.json`:

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
    "entries": {
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
  --model-description qwen3 \
  --workers "$(cat workers.json)"
```

The same `model` entry configuration applies to SenseNova and BAGEL with their checkpoint paths and model descriptions. The product of tensor, pipeline and sequence degrees must match the entry's rank count. Attention heads must admit the declared partition. Pipeline stages own nonempty layer ranges; layer counts need not divide evenly between stages.

For PP2 on two devices, use `pipeline_parallel_size: 2`, `tensor_parallel_size: 1`, and omit `sequence_parallel`. Each stage loads its assigned global checkpoint layers. Embedding and prediction modules follow their stage ownership, and KV storage records each stage's logical layer and head region. An incompatible layout is rejected before serving.

Sequence parallelism partitions packed input rows and exchanges attention heads. It replicates a stage's decoder weights unless combined with tensor or pipeline parallelism. Short decode requests can incur more communication without enough computation to offset it; select topology using the intended workload.

MiniMax H3 has multiple component entries for conditioning, denoising and media decoding. Its default placement and explicit component configuration are described in the [FastH3 guide](fast_h3/fast_h3.md).

## Shared execution capabilities

| Capability | Applicability |
| --- | --- |
| Distributed vocabulary selection | Text outputs from Qwen3, SenseNova and BAGEL; hidden and continuous outputs retain their own semantics. Sampling features that require full logits materialize them through the same output interface. |
| Incremental row execution | Layer operations whose numerical dependencies permit independent row intervals. Completed attention exchanges can feed output equations and the next layer's projection. Small payloads use a complete exchange. |
| Quantized row projection | Splits respect row, block or tensor scale requirements. Tensor-wide dependencies use complete inputs and the relevant parallel reduction groups. |
| Attention and collective selection | Providers are selected from the device, dtype, layout, mask, topology and execution domain. Model names and the presence of KV storage are not selection criteria. |
| Product storage | Physical capacity follows output geometry, placement, consumers and outstanding execution. A product remains live until its physical readers finish, including cancellation and cross-device transfer. |

Attention transfer capacity is allocated before graph capture and variable memory pools. Serialized layers and shape buckets reuse the same buffers; independent execution lanes own disjoint storage. The current attention output exchange and the following layer's projected inputs use separate regions. Registered memory enables supported NCCL collective algorithms without changing the numerical provider or logical row ownership.

Graph execution requires a provider that supports the selected attention mode. A backend's eager support does not imply graph support: FlashInfer segmented attention performs host planning, while FA4 provides a capturable packed path on supported devices. Cached prefix lengths and the lengths after adding current tokens are separate domains; each attention segment is planned from its own boundaries.

Product release and request retirement have different scopes. Releasing a completed image or latent publication does not wait for unrelated KV computation belonging to the same request. Reusing physical pages still waits for every computation or transfer that can access those pages.

The physical verification matrix covers GB200 with two and four participating devices, including combined TP/SP/PP layouts and graph replay. Larger rank counts and Hopper or SM120 devices require separate physical verification; capability-based selection alone does not establish performance on those systems.

Floating-point equivalence is evaluated with dtype-appropriate error bounds and model quality. Implementations may fuse operations and choose different reduction orders; no model requires bitwise reproduction of a particular provider or GPU topology. Data transport and integer control metadata retain their exact contracts.

Mixed-batch startup measures service time to select useful execution geometries. It does not require identical greedy tokens or compare full-model outputs using a single operator's dtype tolerance. Numerical conformance is verified separately against independent references; small score changes near a tie may change greedy selection.
