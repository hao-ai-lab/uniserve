"""Formal FastH3 activation calibration over production inference forwards."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_DENOISER = re.compile(r"^transformer_blocks\.(\d+)\.ff\.fc_(in|out)$")
_TEXT = re.compile(
    r"^language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$"
)
_VAE = re.compile(
    r"^decoder\.transformer_blocks\.(\d+)\."
    r"(attn\.to_[qkv]|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)$"
)


@dataclass
class _Collector:
    calibrator: object
    calls: int = 0
    values: int = 0

    def __call__(self, module, inputs):
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("calibrated Linear input must be a Tensor")
        self.calibrator.collect(value)
        self.calls += 1
        self.values += value.numel()


def _module(module):
    """Return an eager module through supported distributed wrappers."""
    while hasattr(module, "module") and isinstance(
        module.module, torch.nn.Module
    ):
        module = module.module
    return module


def _targets(worker, phase: str) -> dict[str, torch.nn.Module]:
    modules = worker.pipeline.modules
    requested = {
        "phase-a": ("text_encoder", "transformer"),
        "phase-b": ("vae",),
        "smoke": ("text_encoder", "transformer", "vae"),
    }[phase]
    missing = [name for name in requested if name not in modules]
    if missing:
        raise RuntimeError(f"calibration pipeline is missing modules {missing}")
    targets = {}
    for component in requested:
        root = _module(modules[component])
        pattern = {
            "text_encoder": _TEXT,
            "transformer": _DENOISER,
            "vae": _VAE,
        }[component]
        prefix = {
            "text_encoder": "text_encoder",
            "transformer": "denoiser",
            "vae": "video_vae",
        }[component]
        for name, child in root.named_modules():
            match = pattern.fullmatch(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if component == "text_encoder" and layer >= 50:
                continue
            targets[f"{prefix}.{name}"] = child
    expected = {
        "phase-a": 100 + 150,
        "phase-b": 36 * 6,
        "smoke": 100 + 150 + 36 * 6,
    }[phase]
    # VAE gate/up is one fused Linear, hence six source Linears per block.
    if len(targets) != expected:
        raise RuntimeError(
            f"{phase} architecture match expected {expected} Linears, found "
            f"{len(targets)}: {sorted(targets)[:8]}"
        )
    return targets


def verify_modelopt_support(worker):
    """Verify ModelOpt owns the real FastVideo Linear types used by H3."""
    from fastvideo.layers.linear import ColumnParallelLinear, RowParallelLinear
    from modelopt.torch.quantization.nn import QuantModuleRegistry

    # Importing the integration module performs ModelOpt's supported registry
    # extension for the actual FastVideo classes, not a substitute module.
    from uniserve_eval import h3_modelopt_plugin  # noqa: F401

    registered = {
        cls.__name__: QuantModuleRegistry.get(cls) is not None
        for cls in (ColumnParallelLinear, RowParallelLinear)
    }
    if not all(registered.values()):
        raise RuntimeError(
            f"ModelOpt FastVideo Linear registration failed: {registered}"
        )
    text_targets = _targets(worker, "phase-a")
    text_types = {
        type(module).__name__
        for name, module in text_targets.items()
        if name.startswith("text_encoder.")
    }
    if not text_types.issubset({"ColumnParallelLinear", "RowParallelLinear"}):
        raise RuntimeError(
            f"unexpected H3 text Linear types: {sorted(text_types)}"
        )
    return {
        "rank": worker.rank,
        "registered": registered,
        "text_types": sorted(text_types),
    }


def _modelopt_config(names: list[str]) -> dict:
    weight = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
    }
    activation = {
        "num_bits": (2, 1),
        "effective_bits": 4.5,
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    }
    quantizers = [{"quantizer_name": "*", "enable": False}]
    for name in names:
        quantizers.extend(
            (
                {"quantizer_name": name + "*weight_quantizer", "cfg": weight},
                {
                    "quantizer_name": name + "*input_quantizer",
                    "cfg": activation,
                },
            )
        )
    return {
        "quant_cfg": quantizers,
        "algorithm": {"method": "mse", "fp8_scale_sweep": True},
    }


def setup_deployment_quant(
    worker,
    *,
    activation_scales: dict,
    components: tuple[str, ...] = ("transformer", "text_encoder"),
):
    """Install calibrated ModelOpt fake quant for Phase-B deployment inputs."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn.modules.quant_module import (
        QuantLinearConvBase,
    )

    from uniserve_eval import h3_modelopt_plugin  # noqa: F401

    if hasattr(worker, "_h3_modelopt_text_quantizers"):
        raise RuntimeError("H3 deployment quantization is already configured")
    modules = worker.pipeline.modules
    selected = {}
    specifications = {
        "transformer": (_DENOISER, "denoiser", 100),
        "text_encoder": (_TEXT, "text_encoder", 150),
    }
    unknown = set(components).difference(specifications)
    if unknown:
        raise ValueError(
            f"unknown deployment quantization components: {sorted(unknown)}"
        )
    for component in components:
        pattern, prefix, expected = specifications[component]
        root = _module(modules[component])
        names = []
        for name, child in root.named_modules():
            match = pattern.fullmatch(name)
            if match is None or (
                component == "text_encoder" and int(match.group(1)) >= 50
            ):
                continue
            names.append(name)
        if len(names) != expected:
            raise RuntimeError(
                f"Phase-B {component} expected {expected} quantized Linears, "
                f"found {len(names)}"
            )
        mtq.quantize(root, _modelopt_config(sorted(names)), forward_loop=None)
        root_modules = dict(root.named_modules())
        for name in names:
            module = root_modules[name]
            if not isinstance(module, QuantLinearConvBase):
                raise TypeError(
                    f"ModelOpt did not convert real module {component}.{name}"
                )
            key = f"{prefix}.{name}"
            calibrated = activation_scales.get(key)
            if calibrated is None:
                raise KeyError(f"Phase-A activation scale is missing {key}")
            amax = torch.tensor(
                calibrated["activation_amax"],
                dtype=torch.float32,
                device=module.weight.device,
            )
            module.input_quantizer.amax = amax
            selected[key] = module

    text = {
        name: module
        for name, module in selected.items()
        if name.startswith("text_encoder.")
    }
    worker._h3_modelopt_text_quantizers = text
    worker._h3_modelopt_deployment_modules = selected
    set_text_quant_enabled(worker, enabled=False)
    return {
        "rank": worker.rank,
        "denoiser_modules": len(selected) - len(text),
        "text_modules": len(text),
        "text_enabled": False,
    }


def _output_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        tensors = [value for value in output if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(
        f"expected one tensor output, found {type(output).__name__}"
    )


def _error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    difference = actual - expected
    signal = torch.linalg.vector_norm(expected)
    return {
        "nrmse": float(torch.linalg.vector_norm(difference) / signal),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual, expected, dim=0)
        ),
        "maximum_absolute": float(difference.abs().amax()),
    }


@torch.inference_mode()
def text_conditioning_conformance(
    worker, *, input_ids: list[int], activation_scales: dict
):
    """Compare BF16 and calibrated ModelOpt text on one held-out prompt."""
    if not input_ids:
        raise ValueError("text conformance requires at least one token")
    module = _module(worker.pipeline.modules["text_encoder"])
    selected = worker._h3_modelopt_text_quantizers
    representative = {
        name: child
        for name, child in selected.items()
        if any(
            name.startswith(f"text_encoder.language_model.layers.{layer}.mlp.")
            for layer in (0, 25, 49)
        )
    }
    if len(representative) != 9:
        raise RuntimeError(
            "expected 9 representative text projections, found "
            f"{len(representative)}"
        )

    reference_outputs: dict[str, torch.Tensor] = {}
    projection_metrics = {}
    saturation = {name: {"clipped": 0, "values": 0} for name in selected}
    mode = "reference"
    handles = []

    def output_hook(name):
        def hook(_module, _inputs, output):
            tensor = _output_tensor(output).detach()
            if mode == "reference":
                reference_outputs[name] = tensor.clone()
            else:
                projection_metrics[name] = _error_metrics(
                    tensor, reference_outputs[name]
                )

        return hook

    def input_hook(name):
        def hook(_module, inputs):
            if mode != "candidate":
                return
            tensor = inputs[0]
            amax = float(activation_scales[name]["activation_amax"])
            saturation[name]["clipped"] += int(
                (tensor.float().abs() > amax).sum()
            )
            saturation[name]["values"] += tensor.numel()

        return hook

    for name, child in representative.items():
        handles.append(child.register_forward_hook(output_hook(name)))
    for name, child in selected.items():
        handles.append(child.register_forward_pre_hook(input_hook(name)))

    tokens = torch.tensor(input_ids, dtype=torch.long, device=worker.device)
    try:
        set_text_quant_enabled(worker, enabled=False)
        reference = module(tokens).detach()
        mode = "candidate"
        set_text_quant_enabled(worker, enabled=True)
        candidate = module(tokens).detach()
    finally:
        set_text_quant_enabled(worker, enabled=False)
        for handle in handles:
            handle.remove()

    token_metrics = [
        _error_metrics(candidate[index], reference[index])
        for index in range(reference.shape[0])
    ]
    return {
        "rank": worker.rank,
        "tokens": len(input_ids),
        "final_hidden_state": _error_metrics(candidate, reference),
        "token_metrics": token_metrics,
        "representative_projections": projection_metrics,
        "activation_saturation": saturation,
    }


def set_text_quant_enabled(worker, *, enabled: bool):
    """Select the calibrated or BF16 text branch without changing weights."""
    for module in worker._h3_modelopt_text_quantizers.values():
        for quantizer in (module.input_quantizer, module.weight_quantizer):
            quantizer.enable() if enabled else quantizer.disable()
    worker._h3_modelopt_text_enabled = enabled
    return {"rank": worker.rank, "text_enabled": enabled}


def setup_collectors(worker, *, phase: str, restored: list[dict] | None = None):
    """Install ModelOpt headroom calibrators on exact architecture targets."""
    from modelopt.torch.quantization.calib.nvfp4_act_headroom import (
        NVFP4ActHeadroomCalibrator,
    )

    if hasattr(worker, "_h3_modelopt_collectors"):
        raise RuntimeError("H3 ModelOpt collectors are already installed")
    targets = _targets(worker, phase)
    state = None if restored is None else restored[worker.rank]
    if state is not None and state["rank"] != worker.rank:
        raise ValueError("calibration resume state rank mismatch")
    collectors, handles = {}, []
    for name, module in targets.items():
        calibrator = NVFP4ActHeadroomCalibrator(
            block_size=16,
            anchor_percentile=1.0,
            upper_percentile=99.99,
            rho=16384.0,
            num_bins=512,
        )
        collector = _Collector(calibrator)
        if state is not None:
            saved = state["statistics"].get(name)
            if saved is not None:
                calibrator._hist = (
                    None
                    if saved["histogram"] is None
                    else torch.tensor(
                        saved["histogram"],
                        dtype=torch.int64,
                        device=worker.device,
                    )
                )
                calibrator._running_max = (
                    None
                    if saved["running_max"] is None
                    else torch.tensor(
                        saved["running_max"],
                        dtype=torch.float32,
                        device=worker.device,
                    )
                )
                collector.calls = saved["calls"]
                collector.values = saved["values"]
        collectors[name] = collector
        handles.append(module.register_forward_pre_hook(collector))
    worker._h3_modelopt_collectors = collectors
    worker._h3_modelopt_collector_handles = handles
    return {
        "rank": worker.rank,
        "phase": phase,
        "matched": sorted(targets),
        "count": len(targets),
    }


def snapshot_collectors(worker):
    """Copy resumable per-rank calibrator state to ordinary host values."""
    collectors = worker._h3_modelopt_collectors
    statistics = {}
    for name, collector in collectors.items():
        calibrator = collector.calibrator
        histogram = calibrator._hist
        running_max = calibrator._running_max
        statistics[name] = {
            "histogram": None
            if histogram is None
            else histogram.cpu().tolist(),
            "running_max": None if running_max is None else float(running_max),
            "calls": collector.calls,
            "values": collector.values,
        }
    return {"rank": worker.rank, "statistics": statistics}


def remove_collectors(worker):
    for handle in worker._h3_modelopt_collector_handles:
        handle.remove()
    del worker._h3_modelopt_collector_handles
    del worker._h3_modelopt_collectors
    return {"rank": worker.rank, "removed": True}


def merge_statistics(states: list[dict]) -> dict[str, dict]:
    """Merge disjoint SP observations into ModelOpt's fixed histogram domain."""
    names = sorted({name for state in states for name in state["statistics"]})
    merged = {}
    for name in names:
        items = [state["statistics"].get(name) for state in states]
        items = [
            item
            for item in items
            if item is not None and item["histogram"] is not None
        ]
        if not items:
            raise RuntimeError(
                f"calibration target {name!r} was never executed"
            )
        histogram = torch.tensor(items[0]["histogram"], dtype=torch.int64)
        for item in items[1:]:
            histogram += torch.tensor(item["histogram"], dtype=torch.int64)
        running_max = max(item["running_max"] for item in items)

        from modelopt.torch.quantization.calib.nvfp4_act_headroom import (
            NVFP4ActHeadroomCalibrator,
        )

        calibrator = NVFP4ActHeadroomCalibrator(
            block_size=16,
            anchor_percentile=1.0,
            upper_percentile=99.99,
            rho=16384.0,
            num_bins=512,
        )
        calibrator._hist = histogram
        calibrator._running_max = torch.tensor(running_max, dtype=torch.float32)
        amax = calibrator.compute_amax()
        if amax is None or not torch.isfinite(amax) or amax <= 0:
            raise RuntimeError(
                f"calibration target {name!r} produced invalid amax"
            )
        merged[name] = {
            "activation_amax": float(amax),
            "observed_max": running_max,
            "histogram": histogram.tolist(),
            "calls_by_rank": [
                state["statistics"].get(name, {}).get("calls", 0)
                for state in states
            ],
            "values_by_rank": [
                state["statistics"].get(name, {}).get("values", 0)
                for state in states
            ],
        }
    return merged


def _configure_environment() -> None:
    values = {
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
        "FASTVIDEO_VSA_SM100A": "1",
        "FASTVIDEO_VSA_CUTEDSL": "0",
        "FASTVIDEO_FA4": "1",
        "FASTVIDEO_NVFP4_FA4": "0",
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE": "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE": "1",
        "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
        "FASTVIDEO_ULYSSES_A2A": "off",
        "FASTVIDEO_STAGE_LOGGING": "1",
    }
    os.environ.update(values)


def load_h3_inference_contract(model: Path) -> dict:
    """Load and cross-check the checkpoint-owned FastH3 inference schedule."""
    contract = json.loads((model / "fastvideo_inference.json").read_text())
    if contract.get("schema_version") != "fasth3-inference-contract-v1":
        raise ValueError("unsupported or missing FastH3 inference contract")
    num_inference_steps = int(contract["num_inference_steps"])
    if int(contract["transformer_forwards"]) != num_inference_steps - 1:
        raise ValueError(
            "FastH3 inference contract has inconsistent step counts"
        )
    for scheduler, key in (
        ("scheduler", "video_scheduler_shift"),
        ("audio_scheduler", "audio_scheduler_shift"),
    ):
        configured = float(
            json.loads(
                (model / scheduler / "scheduler_config.json").read_text()
            )["shift"]
        )
        declared = float(contract.get(key, configured))
        if configured != declared:
            raise ValueError(
                f"{scheduler} shift {configured} disagrees with "
                f"{key}={declared}"
            )
    return contract


def _generator(
    model: Path,
    *,
    output_type: str,
    num_gpus: int,
    text_encoder_offload: bool = True,
):
    from fastvideo import VideoGenerator
    from fastvideo.api import (
        CompileConfig,
        ComponentConfig,
        EngineConfig,
        GeneratorConfig,
        OffloadConfig,
        ParallelismConfig,
        PipelineSelection,
    )

    contract = load_h3_inference_contract(model)
    config = GeneratorConfig(
        model_path=str(model),
        pipeline=PipelineSelection(
            components=ComponentConfig(),
            experimental={
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "inference_torch_compile": False,
                "vae_parallel_decode": True,
                "vae_parallel_decode_strategy": "gather",
                "h3_sequential_load": False,
                "VSA_sparsity": float(contract["vsa_sparsity"]),
                "VSA_tile_size": int(contract["vsa_tile_size"]),
                "output_type": output_type,
            },
        ),
        engine=EngineConfig(
            num_gpus=num_gpus,
            execution_backend="mp",
            use_fsdp_inference=False,
            parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
            offload=OffloadConfig(
                dit=False,
                dit_layerwise=False,
                text_encoder=text_encoder_offload,
                vae=True,
                pin_cpu_memory=True,
                lazy_module_load=False,
            ),
            compile=CompileConfig(
                enabled=False,
                text_encoder_enabled=False,
                vae_enabled=False,
                audio_vae_enabled=False,
            ),
        ),
    )
    return VideoGenerator.from_config(config)


def _request(record: dict, *, num_inference_steps: int):
    from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

    runtime = record["runtime_config"]
    return GenerationRequest(
        prompt=record["prompt_compiled"],
        negative_prompt="",
        sampling=SamplingConfig(
            height=runtime["height"],
            width=runtime["width"],
            num_frames=runtime["num_frames"],
            fps=runtime["fps"],
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            batch_cfg=False,
            seed=record["sampling"]["seed"],
        ),
        output=OutputConfig(save_video=False, return_frames=False),
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_activation_calibration(
    *,
    model: Path,
    sample: Path,
    output: Path,
    phase: str,
    num_gpus: int = 4,
    limit: int | None = None,
    resume: bool = True,
    deployment_scales: Path | None = None,
) -> dict:
    """Execute records serially and persist resumable per-rank statistics."""
    _configure_environment()
    records = [json.loads(line) for line in sample.read_text().splitlines()]
    if limit is not None:
        records = records[:limit]
    output.mkdir(parents=True, exist_ok=True)
    contract = load_h3_inference_contract(model)
    num_inference_steps = int(contract["num_inference_steps"])
    state_path = output / "state.pt"
    progress_path = output / "progress.json"
    restored, start = None, 0
    if resume and state_path.is_file() and progress_path.is_file():
        restored = torch.load(state_path, weights_only=True)
        progress = json.loads(progress_path.read_text())
        start = progress["completed_records"]
        if progress["phase"] != phase or progress["total_records"] != len(
            records
        ):
            raise ValueError("calibration resume state disagrees with this run")

    generator = _generator(
        model,
        output_type="latent" if phase == "phase-a" else "pil",
        num_gpus=num_gpus,
    )
    started = time.time()
    try:
        support = generator.executor.collective_rpc(verify_modelopt_support)
        _atomic_json(output / "modelopt-support.json", support)
        if phase == "phase-b":
            if deployment_scales is None or not deployment_scales.is_file():
                raise FileNotFoundError(
                    "Phase B requires completed Phase-A activation scales"
                )
            activation_scales = json.loads(deployment_scales.read_text())
            deployment = generator.executor.collective_rpc(
                setup_deployment_quant,
                kwargs={"activation_scales": activation_scales},
            )
            _atomic_json(output / "deployment-quantization.json", deployment)
        matches = generator.executor.collective_rpc(
            setup_collectors,
            kwargs={"phase": phase, "restored": restored},
        )
        _atomic_json(output / "module-matches.json", matches)
        _atomic_json(
            output / "run-contract.json",
            {
                "model": str(model),
                "sample": str(sample),
                "phase": phase,
                "records": len(records),
                "num_inference_steps": num_inference_steps,
                "transformer_forwards": int(contract["transformer_forwards"]),
                "video_scheduler_shift": float(
                    contract.get("video_scheduler_shift", 12.0)
                ),
                "audio_scheduler_shift": float(
                    contract.get("audio_scheduler_shift", 3.0)
                ),
                "vsa_sparsity": float(contract["vsa_sparsity"]),
                "vsa_tile_size": int(contract["vsa_tile_size"]),
            },
        )
        if start == 0:
            _atomic_json(
                progress_path,
                {
                    "phase": phase,
                    "completed_records": 0,
                    "total_records": len(records),
                    "last_record_id": None,
                    "elapsed_seconds": 0.0,
                },
            )
        text_mode = None
        for index in range(start, len(records)):
            if phase == "phase-b":
                requested = records[index]["calibration"]["phase_b_text"]
                if requested not in {"bf16", "nvfp4"}:
                    raise ValueError(
                        f"invalid Phase-B text assignment {requested!r}"
                    )
                if requested != text_mode:
                    generator.executor.collective_rpc(
                        set_text_quant_enabled,
                        kwargs={"enabled": requested == "nvfp4"},
                    )
                    text_mode = requested
            generator.generate(
                _request(
                    records[index], num_inference_steps=num_inference_steps
                )
            )
            completed = index + 1
            if completed % 10 == 0 or completed == len(records):
                states = generator.executor.collective_rpc(snapshot_collectors)
                temporary = state_path.with_suffix(".pt.tmp")
                torch.save(states, temporary)
                temporary.replace(state_path)
                _atomic_json(
                    progress_path,
                    {
                        "phase": phase,
                        "completed_records": completed,
                        "total_records": len(records),
                        "last_record_id": records[index]["id"],
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"[{phase}] {completed}/{len(records)} records complete",
                    flush=True,
                )
        states = generator.executor.collective_rpc(snapshot_collectors)
        merged = merge_statistics(states)
        _atomic_json(output / "activation-scales.json", merged)
        return {
            "phase": phase,
            "records": len(records),
            "modules": len(merged),
            "output": str(output),
        }
    finally:
        try:
            generator.executor.collective_rpc(remove_collectors)
        finally:
            generator.shutdown()
