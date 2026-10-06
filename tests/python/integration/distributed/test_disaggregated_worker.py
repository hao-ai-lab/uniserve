"""Worker microbatches preserve packed rows, cache state and graph outputs."""

import argparse
import os
import socket
from contextlib import contextmanager

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.fixtures.checkpoints import qwen_moe_checkpoint
from tests.python.integration.runtime.test_prefill_graphs import _call
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.nn.moe import FusedMoE
from uniserve.runtime import PrefixCache
from uniserve.runtime.process_groups import (
    Rendezvous,
    initialize_process_groups,
)
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import (
    graph_table_widths,
    input_buffer_config,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.kv_cache import KVCacheManager

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@contextmanager
def _worker(model, config, *, group=None, attention_ranks=1):
    expert = config.role == "experts"
    runner = ModelExecutor(
        model,
        config,
        expert_group=group,
        attention_ranks=0 if group is None else attention_ranks,
        bindings={} if expert else None,
        entry_points={} if expert else None,
    )
    manager = None
    try:
        if expert:
            runner.configure_experts()
        else:
            cache = PrefixCache(
                model.cache_config,
                num_units=32,
                block_size=16,
                device=config.device,
            )
            manager = KVCacheManager(
                cache,
                info=cache_info(model, config, num_units=32),
                request_pool_size=8,
                table_width=4,
            )
            runner.configure_inputs(
                input_config=input_buffer_config(model, config),
                kv_cache=manager,
                latent_pool=None,
                decode_predicates=torch.ones(
                    9, dtype=torch.bool, device=config.device
                ),
                max_calls=4,
                request_slots=8,
                latent_capacity_units=0,
                table_widths=graph_table_widths(model, config, manager),
                max_inflight=1,
            )
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        if manager is not None:
            manager.block_tables.install(
                tuple(
                    (
                        slot,
                        0,
                        0,
                        tuple(range(1 + (slot - 1) * 4, 1 + slot * 4)),
                        64,
                    )
                    for slot in range(1, 4)
                )
            )
        yield runner, manager
    except BaseException:
        runner.close(aborted=True)
        raise
    else:
        runner.close()
    finally:
        if manager is not None:
            manager.close()


def _inputs(replica=0):
    # Reuse resident buckets with different row lengths and counts, including
    # a one-row call that leaves one of two microbatches empty.
    steps = (
        ((1, (2, 3, 4, 5)), (2, (7, 8, 9, 10, 11, 12, 13)), (3, (15, 16))),
        ((1, (17,)), (2, (18,)), (3, (19,))),
        ((2, (20, 21, 22)), (3, (23, 24, 25, 26, 27, 28))),
        ((1, (29,)),),
        ((1, (30,)), (2, (31,)), (3, (32,))),
    )
    if replica:
        # Independent replicas can prefill while their peers decode, and
        # need not leave the same microbatch empty in a shared expert step.
        steps = (
            ((1, (2, 3, 4)),),
            ((1, (9,)), (2, (7, 8))),
            ((1, (11,)), (2, (12,))),
            ((2, (13, 14, 15)), (3, (19, 20, 21))),
            ((1, (16,)), (2, (17,)), (3, (18,))),
        )
    lengths = {1: 0, 2: 0, 3: 0}
    result = []
    for values in steps:
        decode = all(len(tokens) == 1 for _, tokens in values)
        rows = []
        for slot, tokens in values:
            start = lengths[slot]
            rows.append(
                TokenRow(
                    forward_mode=ForwardMode.DECODE
                    if decode
                    else ForwardMode.PREFILL,
                    token_ids=torch.tensor(tokens),
                    positions=torch.arange(start, start + len(tokens)),
                    selection=TokenSelection.LAST_LOGITS
                    if decode
                    else TokenSelection.ALL_LOGITS,
                    request_pool_idx=slot,
                    seq_len=start,
                    write_kv=True,
                )
            )
            lengths[slot] += len(tokens)
        result.append(tuple(rows))
    return result


@torch.inference_mode()
def _run(rank, local_rank, rendezvous, root, graphs, attention_tp, replicas):
    device = torch.device("cuda", local_rank)
    attention_ranks = attention_tp * replicas
    source = rank < attention_ranks
    with initialize_process_groups(
        rank=rank % attention_tp,
        local_rank=local_rank,
        world_size=attention_tp,
        device=device,
        experts=(rank, 2 * attention_ranks, rendezvous),
    ) as groups:
        metadata = models.read_config(root)
        config = WorkerConfig(
            rank=rank % attention_tp,
            world_size=attention_tp,
            device=str(device),
            model_dtype="bfloat16",
            block_size=16,
            max_batch_calls=4,
            max_request_pool_size=8,
            max_batch_tokens=32,
            max_sequence_tokens=64,
            prefill_graph_token_sizes=(16, 32),
            decode_graph_batch_sizes=(1, 2, 4),
            graph_policy="off",
        )
        expected = []
        inputs = _inputs(rank // attention_tp if source else 0)
        if source:
            reference = models.load_model(metadata, device=device).model
            with _worker(reference, config) as (runner, manager):
                for rows in inputs:
                    output = _call(runner, manager, rows).materialize()
                    expected.append(
                        tuple(value.cpu() for value in output.values)
                    )
            del reference, runner, manager, output
        with torch.device("meta"):
            description = metadata.model_class(metadata.model)
        paths = frozenset(
            path
            for path, module in description.named_modules()
            if isinstance(module, FusedMoE)
        )
        mesh = None
        if source:
            mesh = groups.bind(
                DeviceMesh(
                    ranks=tuple(range(attention_tp)),
                    shape=(attention_tp,),
                    axes=("tp",),
                    rank=rank % attention_tp,
                ),
                device=device,
            )
        model = models.load_model(
            metadata,
            device=device,
            modules=None if source else paths,
            exclude_modules=paths if source else frozenset(),
            meshes={"": mesh} if source else None,
            experts=None
            if source
            else Communicator(
                tuple(range(attention_ranks, 2 * attention_ranks)),
                rank - attention_ranks,
                "experts",
                device,
            ),
        ).model
        if not source:
            model = torch.nn.ModuleList(
                module
                for module in model.modules()
                if isinstance(module, FusedMoE)
            )
        config = config.replace(
            role="model" if source else "experts",
            expert_exchange="deepep",
            expert_microbatches=2,
            graph_policy="full" if graphs else "off",
        )
        with _worker(
            model, config, group=groups.experts, attention_ranks=attention_ranks
        ) as (runner, manager):
            if source:
                # Retain borrowed decode outputs before their backing is
                # reused, without synchronizing each forward on the host.
                observed = [
                    _call(runner, manager, rows).materialize().clone()
                    for rows in inputs
                ]
                for step, (output, wanted) in enumerate(
                    zip(observed, expected, strict=True)
                ):
                    for actual, reference in zip(
                        output.values, wanted, strict=True
                    ):
                        torch.testing.assert_close(
                            actual.cpu(),
                            reference,
                            rtol=2e-2,
                            atol=2e-2,
                            msg=lambda detail: (
                                f"rank {rank}, step {step}: {detail}"
                            ),
                        )
                        assert torch.equal(
                            actual.argmax(-1).cpu(), reference.argmax(-1)
                        )
                    if output.greedy is not None:
                        assert torch.equal(
                            output.greedy.tokens.cpu(),
                            torch.cat([value.argmax(-1) for value in wanted]),
                        )
            while not runner.experts.released:
                runner.join_expert_step(leaving=True)


def _local(rank, port, root, graphs, attention_tp, replicas):
    _run(
        rank,
        rank,
        Rendezvous("127.0.0.1", port),
        root,
        graphs,
        attention_tp,
        replicas,
    )


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize(
    ("attention_tp", "replicas"),
    [(1, 1), (2, 1), (1, 2)],
    ids=["tp1", "tp2", "replicas"],
)
def test_microbatch_worker_matches_colocated_execution(
    tmp_path, graphs, attention_tp, replicas
):
    ranks = 2 * attention_tp * replicas
    if torch.cuda.device_count() < ranks:
        pytest.fail(f"disaggregated worker correctness requires {ranks} GPUs")
    qwen_moe_checkpoint(
        tmp_path,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _local,
        args=(port, str(tmp_path), graphs, attention_tp, replicas),
        nprocs=ranks,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-tp", type=int, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument(
        "--graph-policy", choices=("off", "full"), required=True
    )
    # The worker owns its union TCPStore; torchrun already owns MASTER_PORT.
    parser.add_argument("--expert-port", type=int, required=True)
    arguments = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != (
        2 * arguments.attention_tp * arguments.replicas
    ):
        parser.error("the world must contain equal attention and expert ranks")
    _run(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        Rendezvous(os.environ["MASTER_ADDR"], arguments.expert_port),
        arguments.model,
        arguments.graph_policy == "full",
        arguments.attention_tp,
        arguments.replicas,
    )
