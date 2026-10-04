# Expert placement and execution

UniServe separates a model's numerical composition from the placement of its routed experts. The same model, checkpoint loader and `FusedMoE` layers serve colocated replicas and attention–FFN disaggregation (AFD). Request admission, sampling, KV storage and cancellation remain with attention replicas; dedicated expert workers execute their resident expert shards without request components or KV caches.

## Deployment

The [Qwen3-MoE deployment](../configs/afd/qwen3_moe.json) places a tensor-parallel attention replica on two devices of `rank-0` and its experts on two devices of `rank-1`. Both hosts need the same locked GPU environment, compatible CUDA drivers, peer-accessible NVLink devices, and the same checkpoint at the configured path. Install the environment on each host:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra dev --extra test --extra bench --extra gpu
```

Start the head on `rank-0`:

```bash
.venv/bin/uniserve serve /models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen \
  --workers configs/afd/qwen3_moe.json \
  --host-identity rank-0 \
  --expert-parallel \
  --expert-exchange deepep \
  --expert-microbatches 2
```

The head prints the launcher registration address. On `rank-1`, start the host launcher using that address:

```bash
.venv/bin/uniserve-host --head <head-address> --host-identity rank-1
```

Wait for HTTP readiness after both hosts load their assigned weights and prepare numerical kernels and graphs. The ordinary chat, streaming, cancellation and metrics interfaces apply. Keep the launcher alive until the head retires its workers.

In a deployment file, model groups declare their request components as usual. Expert groups have `"role": "experts"` and an empty `"components"` object. All expert ranks share contiguous partitions of every routed expert layer; the expert count must divide their rank count. Attention ranks load the remaining numerical modules. Tensor parallelism follows the model's existing mathematical constraints. `--data-parallel-size` counts attention replicas; each replica contains one model group, and expert groups are shared across them. All attention replicas use the same component placement and numerical configuration.

## Transports and microbatches

| Placement | Transport | Representation and execution |
| --- | --- | --- |
| Colocated expert shards | `alltoall` | FlashInfer NVLink token exchange with shared grouped-expert kernels |
| Dedicated expert ranks | `deepep` | Elastic asymmetric dispatch/combine with BF16 or calibrated NVFP4 input fields and shared grouped-expert kernels |
| Dedicated expert ranks | `megamoe` | Blackwell split MegaMoE with MXFP8 or calibrated NVFP4 SiLU experts |
| Colocated expert shards | `megamoe` | Fused NVFP4 expert dispatch, computation and combine |
| Colocated immutable expert shards | `dwdp` | Asynchronous expert-weight prefetch; replicas advance independently |

Transport selection preserves checkpoint quantization and activation calibration. It does not select a different model precision. DeepEP's native transport uses NCCL's GIN API; the locked environment includes its required NCCL runtime. Split MegaMoE requires supported Blackwell devices and aligned physical tiles; the provider owns padding and returns the model's logical dimensions.

For a dense Qwen3-MoE checkpoint, `--quantization-config '{"mode":"mxfp8-experts"}' --expert-exchange megamoe` selects K32 MXFP8 weights and activations for the routed experts while attention and routing remain BF16. This explicit precision choice is also available as `precision="mxfp8-experts"` through the public model loader. Calibrated NVFP4 checkpoints supply their own numerical configuration and do not accept precision overrides.

`--expert-microbatches` accepts one to four microbatches, with one as the default. Values above one require dedicated experts. The worker divides whole request rows before staging, so every microbatch owns independent fixed input buffers, numerical context, stream, graph storage and communication buffer. They share model weights and disjoint rows of request storage. More microbatches increase persistent staging and graph storage; startup reserves that storage before fitting the KV pool. The worker's batch limits still bound the complete call.

Each expert step agrees its numerical capability and transfer capacity across ranks. A tensor-parallel source starts only when all its members have the same forward ready. Idle sources join with no tokens, and graph padding has no route weight or request progress. Cooperative yields inside the bound expert operation overlap dispatch and computation while ordinary model methods retain their sequential numerical meaning. Split MegaMoE uses one persistent expert launch for the complete layer and microbatch sequence.

With graphs enabled, an empty microbatch replays its captured expert participation at the agreed capacity. Its graph uses the same stream and private allocation pool as that microbatch's populated forwards. Replicas may have different empty microbatches, including during their first prefill; every microbatch still participates in the complete expert-layer sequence.

## Lifetime and verification

Queued decode successors use the existing request and device-continuation machinery. Cancellation retires request storage after its readers finish. Normal shutdown asks every participating group to leave before retiring collective resources. A terminal loss in a shared expert communicator fails all dependent worker groups; they cannot restart one member into the existing communicator.

Correctness checks compare public prefill/decode logits and selected tokens, encoded transport values, expert route sums, repeated graph replays, empty participation, cancellation and resource reuse. Fix the checkpoint, precision, placements, capacities, seeds and tolerances before a run, and execute measurement points serially. Two-host correctness evidence does not determine the optimal attention/expert ratio or large-scale throughput.
