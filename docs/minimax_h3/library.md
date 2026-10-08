# MiniMax-H3 computation library

The model package loads base MiniMax-H3, FastH3 DMD and FastH3 OmniRef checkpoints through the public model catalog. Each checkpoint determines its denoising components, attention layout, schedule, output heads and elementwise rounding. Component exports resolve their missing text encoder, processor and VAE components at the base revision recorded in the export.

Models are ordinary PyTorch modules. Text and latent encoders, denoisers, video and audio decoders compose the shared numerical capabilities. Execution contexts own kernel workspaces and communication backing; callers own request state and advance the solver. Serving workers use these same modules and numerical calls.

## Generate from Python

The eager library entry accepts a loaded model and a presented text prompt. This example uses a local base checkpoint, generates a portrait video and returns RGB frames, stereo PCM and final latents. The checkpoint and its active components must fit on the selected GPU.

```python
from pathlib import Path

from transformers import AutoTokenizer

from uniserve_models import loading
from uniserve_models.minimax_h3 import processing, weight_config
from uniserve_models.minimax_h3.generation import generate

root = Path("/path/to/MiniMax-H3")
config = loading.read_config(
    root,
    modules=frozenset(
        {
            "text_encoder",
            "denoiser",
            "video_decoder",
            "video_postprocessor",
            "audio_decoder",
        }
    ),
)
model = loading.load_model(
    config, device="cuda", weights=weight_config(config.model)
).model
plan = processing.plan_request(
    processing.Task.T2VA,
    processing.Target(aspect_ratio="9:16", duration_seconds=4.0),
    [],
    processing.VisionConfig.from_processor(root / "processor"),
)
prompt = processing.present(
    AutoTokenizer.from_pretrained(root / "tokenizer"),
    plan,
    "A lighthouse keeper climbs a spiral staircase while waves crash below.",
)
result = generate(
    model,
    prompt_token_ids=prompt.token_ids,
    num_frames=plan.num_frames,
    canvas=plan.canvas,
    seed=7,
)
```

`result.frames` contains `[frames, height, width, 3]` uint8 RGB; `result.audio` contains stereo int16 PCM at 32 kHz. Frame counts follow the model's temporal windows, so this four-second request produces 107 frames. `generate` supports text conditioning; keyframe and reference computation uses the encoders and denoiser's public calls directly.

## Layouts and conditions

A denoiser's `make_size` describes the request's frame count, canvas, prompt and ordered conditions. `layout_size` derives the numerical layout; `holds` checks whether a capacity layout can evaluate a request. Dense layouts pack keyframes and references into their declared prefix. Regional sparse layouts retain each reference region's independent selection domain. Layout indices and latent buffers are borrowed tensors, without request identities or pool ownership.

Visual conditions are encoded in independent temporal units and assembled in the declared latent order. A still image may also be encoded in bands of whole latent patch rows: `VideoEncoder.row_bands` partitions it, `encode(..., rows=...)` encodes one band from the tiles it depends on, and the bands assemble the whole encoding exactly. An `ExecutionContext` created with `tiles={encoder: operator}` encodes that `SpatialEncoder`'s tiles through a borrowed operator equal to its `encode_tile`, such as the replay of a graph captured at `VideoEncoder.still_tile`. Audio conditions supply whole stereo tracks. The denoiser's `encode_conditions`, `prepare_latents`, `prepare_state` and solver calls preserve the checkpoint's condition-noise order, modality schedules and output-head mathematics. Encoded conditions remain fixed while generated video and audio evolve.

For a partitioned model, load and bind component meshes through the public loader and execution context interfaces. Every participating rank executes the same numerical entry; runtime communication owns collective resources. Passing the server's capacity layout to `generate(..., layout=...)` also preserves the prompt padding and conditioning layout used by serving.
