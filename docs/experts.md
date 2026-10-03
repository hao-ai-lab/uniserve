# Routed expert computation

`uniserve.nn.moe.FusedMoE` evaluates selected expert projections over stacked weights. `up_gate` holds `[experts, 2 × intermediate, hidden]` rows, with up channels before gate channels; `down` holds `[experts, hidden, intermediate]`. The caller supplies int32 expert ids and FP32 route weights. `TopK` computes those values from router scores with deterministic tie ordering, optional probability renormalization and optional per-expert scaling.

Expert projections and gating are numerical model operations. `ExecutionContext` binds their prepared kernels, transient workspace and borrowed exchange or weight-prefetch resources. Models remain ordinary `torch.nn.Module` compositions and retain no runtime owner. Portable execution uses the same public expert equation; CUDA preparation selects a supported native provider and reports incompatible representations.

## Loading and partitioning

The public model loader maps individual checkpoint expert tensors into the stacked representation. Tensor parallelism partitions each expert's intermediate channels and reduces the resulting output once. `partition_experts` distributes contiguous expert intervals over a `Communicator`; the expert count must divide its membership. Binding mathematical partitions precedes loading and quantization.

`load_model(..., experts=group)` loads only the rank's expert interval. `modules` selects numerical subtrees, and `exclude_modules` leaves named subtrees unmaterialized, including shared aliases. Unselected parameters remain on meta; callers must bind resources and load all numerical components they intend to execute. Excluded expert subtrees do not inherit an attention module's tensor-parallel reduction.

Calibrated NVFP4 checkpoints own their packed values, per-block and per-expert scales, and activation calibration. A prepared provider can reorder physical rows and scales without changing logical weights. Owners retain one resident representation and must not replace it while another prepared operator or graph borrows it.

## Token exchange

`ExpertExchange` owns the resources for one expert group. Each invocation agrees its capability and token capacity, opens a step, evaluates expert layers in the same order on every rank, and closes the step. FlashInfer NVLink all-to-all dispatches tokens to resident experts and combines the weighted results. The NVFP4 MegaMoE provider fuses dispatch, expert computation and combination under the same step protocol.

A rank with no input tokens still participates. `ExecutionContext.join_expert_layers()` completes the forward's skipped tail with empty source rows; `JoinGraphs` captures this participation for configured capacities. Graph padding contributes no route weight. Drain all participating invocations before closing graphs, contexts and the exchange, in that order.

The colocated MegaMoE provider requires directly mapped NVLink peer memory. Its NVSHMEM owner defaults `NVSHMEM_REMOTE_TRANSPORT` to `none`, preserving any explicit operator setting. This avoids initializing unused RDMA endpoints; peer memory remains the kernel's data path. NVIDIA documents the transport setting in its [NVSHMEM environment reference](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html).

## Independent weight prefetch

`WeightPrefetch` exposes immutable distributed expert weights through contiguous CUDA virtual views. Resident pages alias published storage; remote pages use two local prefetch slots. Setup is collective, while inference dependencies use local copy and compute events, so replicas can advance at different rates or remain idle. Contexts borrow this owner through `weights=` and serialize their calls within its execution domain.

The CUDA graph owner captures computation around copy submissions that cannot be placed inside a graph. External events preserve dependencies between graph segments and copies. Published peer pages, local slots and their mappings must outlive all contexts and graph replays that read them. Release the owner only after those readers have retired.

Install the locked GPU extra to use native expert providers and NVSHMEM. Distributed correctness tests cover weighted outputs, empty participation, repeated graph replay, unequal token counts and independent prefetch progress; they do not establish a preferred deployment ratio or throughput.

## Asymmetric expert exchange

An `ExpertExchange` can separate source-only attention ranks from expert-only ranks in one ordered communicator. `attention_ranks=N` assigns the first N members to sources; the remaining M members own equal contiguous expert partitions. Source parameters may remain on meta, while expert ranks materialize the same `FusedMoE` modules used by colocated execution. `source_group` identifies the global ranks that execute one source forward together, such as a tensor-parallel group. Every member must declare consistent, disjoint source groups, and a source group starts only after all its members submit the same capability.

The runtime retains the model's logical expert ids, hidden widths and numerical representation. A transport provider owns wire ids, padding, communication streams and staging buffers. Every rank agrees the capability and capacity and visits the same layer sequence, including empty sources. Dispatch and combine alternate before reusing a buffer; unused capacity has invalid expert ids and zero route weights. The expert equation applies each route weight exactly once, after its output projection, before returning the completed logical rows to the source.

| Transport | Source/expert placement | Numerical representation |
| --- | --- | --- |
| `deepep` | Unequal N/M groups or colocated expert parallelism | BF16 or calibrated NVFP4; packed values and scales travel without requantization |
| `megamoe` | Unequal N/M groups | Split MegaMoE on supported Blackwell devices, with MXFP8 or calibrated NVFP4 SiLU experts |

DeepEP uses the NCCL GIN API; the locked environment pins NCCL 2.30.4 for both PyTorch and the native transport. Split MegaMoE pads physical hidden and intermediate tiles to multiples of 512 and removes padding from returned rows. Padding preserves encoded values and scales. Native sources from the pinned FastAFD reference and their dependency licenses are packaged with `uniserve-kernels`; the package build compiles their host bindings, and their device kernels compile at run time with NVRTC from the packaged headers. Kernel preparation completes before peers enter device communication waits.

## Numerical microbatches

`Microbatches` borrows an ordered sequence of `ExecutionContext` instances with distinct CUDA streams, scratch and expert exchanges on one device. Immutable model weights may be shared. Each invocation receives one ordinary numerical callable per context and returns results in the same order:

```python
from contextlib import closing

from uniserve.runtime import Microbatches

# Contexts, inputs and model weights have already been loaded and bound.
with closing(Microbatches(contexts)) as run:
    outputs = run((lambda: model(inputs[0]), lambda: model(inputs[1])))
```

The runtime starts host turns in index order and yields after posting an expert dispatch. Model forwards keep their local variables and ordinary sequential composition. The caller's CUDA stream forks into the context streams and joins them before returning, so `CUDAGraph` captures the complete dependency graph. Warm every numerical specialization through this owner before capture; CUDA library handles belong to the host threads that execute them.

A host exception wakes suspended calls, waits for their host turns to retire and propagates the original failure. One owner cannot run overlapping invocations. Distributed process failure remains the deployment owner's responsibility. Normal retirement drains device readers before closing graphs, microbatch owners, contexts and collective buffers. Split MegaMoE uses one persistent expert launch over the complete layer-major, microbatch-minor sequence, avoiding competing persistent grids that could wait on different peers.
