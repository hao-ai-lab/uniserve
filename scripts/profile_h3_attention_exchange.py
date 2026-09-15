from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from uniserve.distributed import DeviceMesh
from uniserve.ops.video_sparse import (
    compose_to_head_shards,
    unpack_add_compression,
)
from uniserve.runtime import TensorBuffers, initialize_process_groups
from uniserve.tensors import BufferConfig
from uniserve_worker.bootstrap.config import ParallelConfig, SequenceConfig


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    options = dist.ProcessGroupNCCL.Options()
    options.use_pg_for_symm_mem_rendezvous = True
    dist.init_process_group("nccl", pg_options=options, device_id=device)
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(
            "the H3 attention exchange profile requires four ranks"
        )

    local_rows = 10_944
    global_rows = local_rows * world
    local_heads = 14
    global_heads = local_heads * world
    width = 128
    tiles = global_rows // 64
    generator = torch.Generator(device=device).manual_seed(100 + rank)
    attended = torch.randn(
        (1, local_heads, global_rows, width),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    gate = torch.randn(
        (global_rows, local_heads, width),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    alternate_gate = torch.randn(
        (global_rows, local_heads, width),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    compressed = torch.randn(
        (local_heads, tiles, width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    local_output = torch.empty_like(gate)
    receive = torch.empty(
        (world, local_rows, local_heads, width),
        device=device,
        dtype=torch.bfloat16,
    )
    reference = torch.empty(
        (local_rows, global_heads, width),
        device=device,
        dtype=torch.bfloat16,
    )
    candidate_output = torch.empty_like(reference)
    sync_input = torch.full((1,), rank, device=device, dtype=torch.int32)
    sync_output = torch.empty((world,), device=device, dtype=torch.int32)
    environment = initialize_process_groups(
        rank=rank, local_rank=rank, world_size=world, device=str(device)
    )
    bound_meshes = {}
    for mesh_name, (mesh_ranks, mesh_parallel) in sorted(
        (
            {
                "denoiser": (
                    tuple(range(world)),
                    ParallelConfig(
                        sequence_parallel=SequenceConfig("ulysses", (world,))
                    ),
                )
            }
        ).items()
    ):
        topology = DeviceMesh(
            ranks=mesh_ranks,
            shape=tuple(size for _, size in mesh_parallel.dimensions),
            axes=tuple(axis for axis, _ in mesh_parallel.dimensions),
            rank=environment.rank,
        )
        bound_mesh = environment.bind(topology, device=environment.device)
        if environment.rank in mesh_ranks:
            bound_meshes[mesh_name] = bound_mesh
    mesh = bound_meshes["denoiser"]
    group = mesh.get_group("ulysses")
    schema = {
        "output": BufferConfig(
            (local_rows, global_heads, width), torch.bfloat16
        )
    }
    workspace = TensorBuffers.allocate(
        schema, device=device, symmetric={"output": group}
    )
    local = workspace.view(schema)["output"]

    def collective_fence() -> None:
        group.all_gather(sync_input, out=sync_output)

    def baseline_once(layer_gate: torch.Tensor) -> None:
        unpack_add_compression(attended, layer_gate, compressed, local_output)
        work = dist.all_to_all_single(receive, local_output, async_op=True)
        work.block_current_stream()
        reference.copy_(receive.permute(1, 0, 2, 3).reshape_as(reference))

    def candidate_once(layer_gate: torch.Tensor) -> None:
        compose_to_head_shards(
            attended,
            layer_gate,
            compressed,
            workspace.peers("output"),
            rank,
        )
        collective_fence()
        candidate_output.copy_(local)
        collective_fence()

    def baseline() -> None:
        for layer in range(48):
            baseline_once(gate if layer % 2 == 0 else alternate_gate)

    def candidate() -> None:
        for layer in range(48):
            candidate_once(gate if layer % 2 == 0 else alternate_gate)

    def capture(function: object) -> torch.cuda.CUDAGraph:
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            function()  # type: ignore[operator]
        stream.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph, stream=stream):
            function()  # type: ignore[operator]
        return graph

    def time_graph(
        graph: torch.cuda.CUDAGraph, iterations: int = 20
    ) -> list[float]:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        local = torch.tensor(
            [start.elapsed_time(end) / iterations / 48], device=device
        )
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        return [float(value.item()) for value in gathered]

    baseline_graph = capture(baseline)
    baseline_times = time_graph(baseline_graph)
    candidate_graph = capture(candidate)
    candidate_graph_2 = capture(candidate)
    candidate_times = time_graph(candidate_graph)
    correct_first = bool(torch.equal(candidate_output, reference))
    candidate_times_2 = time_graph(candidate_graph_2)
    correct_second = bool(torch.equal(candidate_output, reference))
    candidate_times_3 = time_graph(candidate_graph)
    correct_third = bool(torch.equal(candidate_output, reference))
    correct = (correct_first, correct_second, correct_third)
    correctness = [None] * world
    dist.all_gather_object(correctness, correct)
    if rank == 0:
        print(
            json.dumps(
                {
                    "baseline_ms": baseline_times,
                    "candidate_ms": candidate_times,
                    "candidate_second_graph_ms": candidate_times_2,
                    "candidate_first_graph_replay_ms": candidate_times_3,
                    "slowest_speedup": max(baseline_times)
                    / max(candidate_times),
                    "correct": correctness,
                }
            ),
            flush=True,
        )
    dist.barrier()
    torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
