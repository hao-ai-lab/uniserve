"""CPU worker fixture: real H3 preparation with synthetic weights, stopping at transformer inputs.

The native endpoint and Worker.execute are the same public boundaries exercised by
worker IPC integration tests. Meta denoising/decoding weights are deliberately not
executed: this fixture provides input-path evidence, not video generation evidence.
"""

import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from tests.python.unit.models.test_h3_image_conditioning import make_encoder
from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.ipc import WorkerIpcEndpoint
from uniserve_worker.execution.output import finalize_run_result
from uniserve_worker.models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
from uniserve_worker.models.minimax_h3.encoder import H3TextEncoderConfig
from uniserve_worker.models.minimax_h3.layout import H3Layout
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.packing import audio_latent_frames
from uniserve_worker.models.minimax_h3.transformer import MiniMaxH3Transformer
from uniserve_worker.models.minimax_h3.video_vae import (
    H3ImagePosterior,
    MiniMaxH3VideoDecoder,
    MiniMaxH3VideoVAE,
)
from uniserve_worker.models.minimax_h3.weights import H3Components
from uniserve_worker.nn.diffusion.modulation import ModulationPlan
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.runtime.distributed import DistributedEnvironment
from uniserve_worker.worker import Worker


class ProjectedTextEncoder(nn.Module):
    """Small synthetic Qwen weights with the checkpoint's public channel width."""

    def __init__(self):
        super().__init__()
        self.encoder = make_encoder()
        self.processor = self.encoder.processor
        self.projection = nn.Linear(
            16, H3TextEncoderConfig().hidden_size, dtype=torch.bfloat16, bias=False
        )
        with torch.no_grad():
            self.projection.weight.zero_()
            rows = torch.arange(self.projection.out_features)
            self.projection.weight[rows, rows % 16] = 1

    def numerical_entry(self, tokens, pixels=None):
        result = (
            self.encoder.numerical_entry(tokens)
            if pixels is None
            else self.encoder.numerical_entry(tokens, pixels)
        )
        if isinstance(result, tuple):
            states, tags = result
            return self.projection(states), tags
        return self.projection(result)


@torch.inference_mode()
def main():
    from diffusers import AutoencoderKLMiniMaxH3Audio

    logging.basicConfig(level=logging.INFO)
    args = parse_worker_args()
    config = replace(args.execution, device="cpu", graph_policy="off", max_request_pool_size=2)
    entries = dict(args.components)
    bindings = EntryBindings(
        entries,
        {name: DeviceMesh((0,), 0, entry.parallel_config) for name, entry in entries.items()},
        Communicator(),
    )
    encoder = ProjectedTextEncoder()
    transformer = MiniMaxH3Transformer(
        bindings.meshes["denoiser"],
        parameter_device="meta",
        attention="dense",
        attention_linear_precision="bf16",
        mlp_linear_precision="bf16",
    )
    transformer.modulation_plan = ModulationPlan(
        torch.empty(1, 50, 3, 6 * 5376, device="meta"),
        torch.empty(1, 3, 2 * 5376, device="meta"),
    )
    posterior = H3ImagePosterior(parameter_device="cpu")
    for parameter in posterior.parameters():
        parameter.zero_()
    image_vae = MiniMaxH3VideoVAE(posterior, linear_precision="fp32")
    video = MiniMaxH3VideoVAE(
        MiniMaxH3VideoDecoder(
            layer_config=LayerConfig(Communicator(), None),
            parameter_device="meta",
            buffer_device="cpu",
        ),
        linear_precision="fp32",
    )
    root = Path(args.model.path)
    with torch.device("meta"):
        audio = MiniMaxH3AudioVAE(
            AutoencoderKLMiniMaxH3Audio.from_config(
                json.loads((root / "audio_vae/config.json").read_text())
            )
        )
    conditioner = nn.Sequential(
        nn.Linear(H3TextEncoderConfig().hidden_size, 5376, dtype=torch.bfloat16, bias=False)
    )
    # Tiled identity weights make the full-channel transformer conditioning
    # independently predictable from the small real Qwen encoder's output.
    conditioner[0].weight.zero_()
    rows = torch.arange(5376)
    conditioner[0].weight[rows, rows % 16] = 1
    layout = H3Layout.build(
        bindings,
        frames=39,
        text_rows=1024,
        audio_frames=audio_latent_frames(39),
        height=480,
        width=832,
        attention="dense",
        reference_shape=(64, 64),
        presentation_tags=torch.ones(1024, dtype=torch.long),
    )
    model = MiniMaxH3Model(
        bindings,
        H3Components(transformer, conditioner, encoder, video, audio, image_vae),
        layout,
        denoise_steps=1,
        presentation_processor=encoder.processor,
    )
    output = Path(os.environ["UNISERVE_INPUT_EVIDENCE"])
    count = 0

    def capture(_module, inputs):
        nonlocal count
        tensors, scratch, metadata, _step = inputs
        packed = metadata.layout.packed
        evidence = {
            "text": tensors.text_condition.clone(),
            "video": tensors.video_rows.clone(),
            "audio": tensors.audio_rows.clone(),
            "reference": tensors.reference_rows.clone(),
            "reference_indices": packed.reference_indices.clone(),
            "text_indices": packed.text_indices.clone(),
            "audio_indices": packed.audio_indices.clone(),
            "positions": scratch.rotary_positions.clone(),
            "tags": packed.presentation_tags,
        }
        temporary = output / f"{count}.part"
        torch.save(evidence, temporary)
        temporary.rename(output / f"{count}.pt")
        count += 1
        # A test observation boundary, not a replacement denoiser implementation.
        # No output media can be mistaken for a generated result from meta weights.
        raise RuntimeError("CPU evidence ends at transformer inputs")

    transformer.register_forward_pre_hook(capture)
    environment = DistributedEnvironment(0, 1, torch.device("cpu"), "gloo")
    with (
        environment,
        WorkerIpcEndpoint(
            args.ipc.service_name, args.ipc.max_payload_bytes, args.ipc.max_inflight
        ) as endpoint,
        Worker(
            model,
            worker_id=args.worker_id,
            worker_config=config,
            sampling_group=Communicator(),
            tokenizer=None,
            allowed_work_variants=model.supported_work,
            pipeline_depth=args.ipc.pipeline_depth,
            completion_payload_bytes=args.ipc.max_payload_bytes,
            schedule=DiffusionSchedule.uniform_grid(2, (12.0, 3.0), device="cpu"),
            components=args.components,
            distributed_environment=environment,
        ) as worker,
    ):
        # Exercise the synchronous public execution API without numerical warmup:
        # the contract under test ends before denoising and decoding weights run.
        while True:
            request = endpoint.recv()
            kind = request["kind"]
            call_id = request.get("call_id")
            if kind == "info":
                endpoint.respond(
                    {"kind": "info", "call_id": call_id, "info": worker.info.to_mapping()}
                )
            elif kind == "submit":
                logging.info(
                    "CPU input fixture operations: %s",
                    [op.kind for op in request["run"].operations],
                )
                prepared = worker.prepare_execute(worker.plan_run(request["run"]))
                deadline = time.monotonic() + 30
                while not prepared.advance():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CPU fixture inputs did not become ready")
                    time.sleep(0.001)
                result = finalize_run_result(worker.execute_prepared(prepared))
                endpoint.respond(
                    {"kind": "result", "call_id": call_id, "result": result.to_mapping()}
                )
            elif kind == "close":
                endpoint.respond({"kind": "ok", "call_id": call_id})
                break
            else:
                raise ValueError(f"unexpected CPU fixture request: {kind}")


if __name__ == "__main__":
    main()
