"""Startup reports the kernel serving every prepared call site."""

from __future__ import annotations

import json
import logging

import pytest
import torch
from torch import nn

from tests.python.fixtures.execution_worker import execution_worker
from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.kernel_table import KERNEL_TABLE_TAG

pytestmark = pytest.mark.integration

# Providers that are not native CUDA kernels: the portable reference, the
# FlashAttention-2 library and FlashInfer's own attention templates.
NON_NATIVE = {"torch", "flash_attn", "flashinfer"}


def _tables(caplog) -> list[dict]:
    prefix = f"{KERNEL_TABLE_TAG} "
    return [
        json.loads(record.getMessage().removeprefix(prefix))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_worker_startup_logs_one_table_of_every_call_site(device, caplog):
    """Warmup ends with one tagged JSON table of the selected kernels.

    The stub model's single cache-writing attention layer is served by the
    portable provider by name; on CUDA the table also lists it among the
    portable call sites so that no such call site is silent.
    """
    policy = WorkerConfig(
        graph_policy="off",
        prefill_cuda_graph=False,
        flow_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
    )
    with (
        execution_worker(device=device, execution=policy) as worker,
        caplog.at_level(logging.INFO, logger="uniserve_worker"),
    ):
        worker.warmup()

    (table,) = _tables(caplog)
    assert (table["rank"], table["device"]) == (0, device)
    attention = [
        (runner["runner"], site)
        for runner in table["runners"]
        for site in runner["call_sites"]
        if site["op"] == "attention"
    ]
    assert attention
    assert all(
        site["provider"] == "torch" and site["layers"] for _, site in attention
    )
    portable = [
        (entry["runner"], entry["layers"])
        for entry in table["portable"]
        if entry["op"] == "attention"
    ]
    assert portable == (
        [(runner, site["layers"]) for runner, site in attention]
        if device != "cpu"
        else []
    )


@pytest.mark.gpu
@torch.inference_mode()
def test_attention_records_name_the_provider_of_each_input_class():
    """Each attention call site reports the provider of every input class.

    A half-precision dense call is served natively; the FP32 single-head
    spatial attention of an image autoencoder is the one class the portable
    provider serves on CUDA, and its record says so.
    """

    class Layers(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = Attention(1, 1, 512)
            self.text = Attention(8, 2, 128)

    device = torch.device("cuda", 0)
    layers = Layers()
    spatial = [torch.randn((1, 1, 64, 512), device=device) for _ in range(3)]
    text = [
        torch.randn((1, heads, 64, 128), device=device, dtype=torch.bfloat16)
        for heads in (8, 2, 2)
    ]

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    with (
        stream,
        ExecutionContext(layers, attention="auto", stream=stream) as context,
    ):
        context.prepare(None)
        layers.spatial(*spatial, AttentionBatch.single(DenseInput(False, None)))
        layers.text(*text, AttentionBatch.single(DenseInput(True, None)))
        records = {record["path"]: record for record in context.kernels()}

    assert records["spatial"]["op"] == "attention"
    assert records["spatial"]["dtype"] == "float32"
    assert records["spatial"]["inputs"] == {
        "dense attention: non-causal": "torch"
    }
    assert records["text"]["dtype"] == "bfloat16"
    assert list(records["text"]["inputs"]) == ["dense attention: causal"]
    assert records["text"]["inputs"]["dense attention: causal"] not in (
        NON_NATIVE
    )
