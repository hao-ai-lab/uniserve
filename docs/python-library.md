# Python computation library

UniServe's Python loader constructs the same numerical model used by serving. `resolve_model` normalizes checkpoint metadata into the concrete typed config and resolves weight files, tokenizer, image transforms and diffusion prompt assets. `load_model` receives that model class, config, sources and explicit numerical resource parameters. It returns the materialized model, loaded sources and validated weight reports. The caller owns placement, input processing, schedules and execution resources.

```python
from uniserve_models import resolve_model
from uniserve.loading import load_model

source = resolve_model(path, load=load_config)
layers = source.configure_layers(layers)
loaded = load_model(
    source.model_class, source.config, sources=source.weights, load=load_config,
    device=device, dtype=dtype, parallel=parallel, meshes=meshes, layers=layers,
    limits=limits,
)
model = loaded.model
```

`parallel` maps actual module paths to mathematical partitions. `meshes` and `layers` bind resident numerical resources; the empty path denotes the root module. `ModelLimits(text_tokens, video_frames)` states numerical input bounds. For component-only loading, `resolve_model(..., components=frozenset(module_paths))` selects the required checkpoint sources while still validating all architecture sidecars. `flow_device` optionally places explicitly declared flow branches on a second device; their calls return results to the input device.

Models compose ordinary `torch.nn.Module` layers with `TextMixin`, `EncoderMixin`, `DiffusionMixin`, `DecoderMixin`, and `VideoMixin`. Text uses `forward`, `compute_logits`, and `embed_input_ids`; diffusion uses `prepare_latents` and `forward_diffusion`, with component-owned latent/noise sizes and a numerical solver; encoders and decoders use `encode` and `decode`. A numerical call accepts one homogeneous computation batch. Independent text and diffusion calls use separate execution resources when they run concurrently.

## Compute text logits

The [text logits example](../examples/text_logits.py) loads a Qwen3, BAGEL, or SenseNova checkpoint, binds real KV tensors and the public PyTorch SDPA backend, and evaluates the checkpoint's text capability directly. It uses the shared metadata construction and numerical model implementation without a server or request pool.

```bash
python examples/text_logits.py \
  --checkpoint /path/to/Qwen3-32B \
  --text "Name a primary color." \
  --device cuda:0 \
  --output /path/to/logits.pt
```

The saved dictionary contains CPU `input_ids` shaped `[tokens]` and BF16 `logits` shaped `[tokens, vocabulary]`. Each row predicts the token following that prompt position using causal attention. The input is raw text, without chat-template framing or sampling. The example joins vocabulary shards and removes padding from the numerical output. Its complete CPU copy precedes KV and process-group retirement.

## Decode H3 video latents

The [video decoding example](../examples/decode_video.py) loads the video decoder and pixel postprocessing components from the supported full FastH3 VSA checkpoint. It uses the public loader and resource binding APIs, decodes the declared windows, preserves overlap between windows, and returns an independent CPU RGB tensor. It does not start a server or allocate serving requests.

Run it from an installed UniServe checkout with its CUDA dependencies and the full `FastVideo/FastH3-4-step-Preview-v1-VSA-DataFree` checkpoint available locally:

```bash
python examples/decode_video.py \
  --checkpoint /path/to/FastH3-4-step-Preview-v1-VSA-DataFree \
  --latents /path/to/video-latents.pt \
  --frames 22 \
  --device cuda:0 \
  --output /path/to/rgb.pt
```

The latent file must contain one FP32 tensor saved with `torch.save`. Its rows are the complete final video modality in H3 packed order, with shape `[((frames - 5) // 17 * 5 + 2) * 24 * 42, 96]`. Join sequence shards in logical order before calling the example. Frame counts must be at least 22 and equal `17 * n + 5`; the checkpoint raster is 768 × 1344 at 24 frames per second. These are checkpoint mathematics, not adjustable output-quality settings.

The saved result is a CPU `torch.uint8` tensor shaped `[frames, 768, 1344, 3]` in RGB order. It contains decoded pixels; an application can pass them to its own image or video encoder. The example does not generate latents or decode audio.

## Image diffusion mathematics

`ImageDiffusion` describes per-image latent bounds, framing markers, patch conversion, noise scaling, CFG semantics, and analytical schedule coordinates. The standard `DiffusionMixin` uses it to compute `latent_shape` and `noise_shape` and initialize caller-owned latent state from supplied noise. `normal_noise` fills ordered, caller-allocated views from explicit seeds without changing global RNG state. `EulerSolver` and `CleanSampleEulerSolver` preserve their respective prediction conversion and update arithmetic. BAGEL retains patch-token latent rows; SenseNova restores image-space RGB latents through `unpatchify`. Prediction kinds are `velocity` and `sample`.

The execution caller resolves prompt overrides and negative conditioning, generates noise according to the declared RNG contract, owns schedule tensors, and advances the solver. Sequence length includes numerical framing markers; it does not specify how many concurrent trajectories to allocate.

## Encoder inputs

`EncoderMixin.encode` consumes already preprocessed tensors in `EncodeBatch.values` and preserves their row order. Text and conditioning modules process each variable-length row independently and return `conditioning`. VAE encoding stacks images and calls the autoencoder's `encode`, returning `latents` with its posterior sampling semantics intact.

Vision returns `features`. Uniform CHW image rows are stacked into an NCHW batch for the vision encoder. Packed patch rows include aligned device `grids` and host-known `grid_shapes`; the shared implementation concatenates patches and splits results using the encoder's actual spatial downsampling factor. BAGEL composes `nn.vision.PatchEncoder` for patch packing, isolated attention, projection, and learned output positions. SenseNova composes its NEO patch encoder. All inputs must already reside on the participating component device, and shared attention layers require their public backend binding.

`source.image_processor` provides immutable resize, normalization, patch and feature-token settings for the input caller. The resolver binds declared token names to the checkpoint tokenizer and rejects undefined markers. `source.flow_prompt` supplies immutable classifier-free-guidance text framing where the architecture requires it; its `encode(source.tokenizer, text=..., conditioned=...)` method produces prefix tokens. Numerical models consume the prepared tensors and retain neither asset. Worker construction receives these assets separately from the loaded model.

## Resource lifetime

`initialize_process_groups` owns process groups; `initialize_model_parallel` creates numerical meshes from explicit rank sets and `ParallelConfig` values. Direct callers bind KV attention with `bind_attention_modules` and dense attention with `bind_dense_attention_modules`. Attention modules identify their local cache slice at construction; binding attaches that borrowed slice before the first numerical call, and the caller retains its backing through all computation and graph readers. The video example allocates decoder and RGB storage itself and invokes the numerical components directly. Serving separately creates entries, schedules, pools and graphs around the same model.

Components that require caller-owned storage expose `state_buffers(size)`, `workspace_buffers(size)` and `constant_buffers(size)` as separate queries. Each returns `BufferConfig` values with exact shape, dtype, optional backing capacity and required host representation. `TensorBuffers.allocate` creates the backing; `bind_state(configs, storage)` and `bind_scratch(configs, storage)` lend exact views. `prepare_constants(component, size, device=...)` allocates the component’s constants and asks it to fill borrowed views. Ordinary text, vision and image VAE calls need no resource declaration.

The text backbone owns its activation width, attention mode, input token limit and numerical KV configuration. Vocabulary size comes from the actual vocabulary partition. Independent text encoders expose their own token limits. These quantities are not required fields on every `Model`.

`TextSize(tokens, rows=1, selection=None)` describes a numerical token extent. `TextBatch` supplies actual query lengths and selected rows; inactive selected rows can have zero queries. Shared text projection checks activation width, dtype and live-row coverage while accepting trailing capture padding. `TextOutput` retains the actual vocabulary partition, and `materialize()` gathers complete vocabulary results when needed.

Image decoding reads dtype from the supplied numerical input or learned decoder. H3 components use `VideoSize(frames, prompt_tokens)` for resource queries and `VideoInfo` for raster and sample timing. `video_output.decode_windows(video)` gives the temporal reconstruction intervals. Audio reconstruction receives its target PCM sample count directly.

`VideoAttention.workspace_buffers(rows, query_rows, dtype=...)` declares projection exchange and attention output scratch. The shared backing accommodates both encoded projection inputs and activation-format return rows; callers allocate and retain the declared capacity regardless of weight precision. H3 consumes this same public layer declaration.

`TensorOutput` contains the actual tensors and their optional `OutputLayout`. Layouts retain global shape, dtype, local region and pixel range where consumers need that information. H3 RGB reconstruction returns a prefix of supplied pixel scratch; the caller must retain that backing through every reader.

Each independent request or trajectory needs its own mutable state. Scratch can be reused only after its final reader completes. Decoder graph outputs and pixel scratch can be overwritten by the next call, so the example completes a CPU copy of each window before reusing them. Constants and all captured tensor addresses remain alive until the corresponding graphs retire. Numerical objects and borrowed views leave scope before the process-group owner closes.

Output-producing components expose `output_layout` using their numerical input size. Worker startup translates these global layouts into persistent wire product bounds; local shard regions do not reduce assembly capacity on remote consumers. Direct Python callers can consume `TensorOutput` and choose their own serialization.

`ImageSize(height, width)` defines an image raster. `VideoSize(frames, prompt_tokens)` defines a video and its conditioning extent; the reconstruction modules retain their fixed raster and sample rates. Numerical `DiffusionBatch.sizes` and `DecodeBatch.sizes` follow latent row order. Audio decoding uses the requested PCM sample count directly, and video decoding receives its explicit `DecodeWindow` values. Tensor and layer dtypes determine the result representation.

`uniserve.model.denoising.DenoisingStep` composes the numerical prediction, solver step and pipeline feedback. Its tensors and schedule are borrowed; capture policy, request slots and accepted progress remain outside that computation. `DiffusionSchedule` stores complete sigma and timestep endpoints, including the terminal endpoint. H3 learned modulation uses the four evaluation timesteps before that terminal endpoint.

`DiffusionConfig` contains numerical request options: evaluation count, optional timestep shift, CFG scales/interval/renormalization and seed. Image dimensions use `ImageSize` separately. `ImageDiffusion.create_schedule(config, device=...)` materializes complete FP32 endpoints; `guidance(config, index)` selects and combines CFG branches using the analytical host coordinate before rounding. Omitting the shift selects the model's mathematical default.
