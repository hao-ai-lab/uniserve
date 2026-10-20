# Python library

`uniserve` supplies the numerical layers, loading and resource binding; `uniserve_models` composes them into PyTorch modules; and the serving workers run those same modules. This example generates a 5-second 16:9 video with FastH3 on one GPU:

```python
from transformers import AutoTokenizer

from uniserve_models import loading
from uniserve_models.minimax_h3 import config as h3_config
from uniserve_models.minimax_h3 import processing, weight_config
from uniserve_models.minimax_h3.generation import generate

config = loading.read_config(
    "models/FastH3",
    modules=frozenset(
        {"text_encoder", "transformer", "video_decoder", "video_postprocessor", "audio_decoder"}
    ),
)
model = loading.load_model(config, device="cuda", weights=weight_config(config.model)).model
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
prompt = tokenizer.encode(
    "A clear stream flows through a green forest while birds sing.",
    add_special_tokens=False,
)
result = generate(
    model,
    prompt_token_ids=tuple(prompt),
    num_frames=processing.frame_count(5.0),
    canvas=h3_config.canvas(16, 9),
    seed=42,
)
```

`result.frames` holds `[124, 768, 1344, 3]` uint8 RGB frames and `result.audio` stereo int16 PCM at 32 kHz. `generate` takes text prompts only; keyframe and reference conditions go through the denoiser's `encode_conditions`, `prepare_latents`, `prepare_state` and solver calls. To run a model across several GPUs, bind its component meshes through the loader and execution context; every rank then makes the same call.
