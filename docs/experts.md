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
