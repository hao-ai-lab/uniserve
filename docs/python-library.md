# Python computation library

UniServe's Python loader constructs the same numerical modules used by serving. `uniserve_models.loading.read_config` normalizes checkpoint metadata into immutable architecture configs and resolves the selected checkpoint sources and preprocessing assets. `load_model` materializes the requested modules and returns the model, sources and weight completeness reports. The application owns input preparation, placement, cache storage, execution resources and sampling.

```python
from uniserve_models import loading as models

config = models.read_config("/path/to/Qwen3-32B")
loaded = models.load_model(config, device="cuda:0")
model = loaded.model
```

`modules=frozenset(module_paths)` selects numerical submodules during configuration and loading. The empty path denotes the root module. Shared descendants retain their parameter identity. Unselected modules remain available for architecture inspection on the meta device. `devices` maps module paths to devices; shared aliases must agree on placement. `weights` selects compute dtypes and quantizers by module path, while `precision` selects one of the checkpoint's declared presets.

Models compose ordinary `torch.nn.Module` layers. Capabilities such as `CausalLM`, `TextEncoder`, `ImageDenoiser`, `ImageDecoder` and `VideoDecoder` provide shared numerical behavior. Text uses `forward`, `embed_input_ids` and `compute_logits`; encoders use `encode`; media decoders use `decode`; denoisers use `prepare_latents` and `forward`. Independent token decoding and diffusion use separate homogeneous calls. Concurrent callers share immutable model parameters and own independent execution contexts.

## Model packages

Each architecture has a package under `uniserve_models`: `qwen3`, `bagel`, `sensenova_u1` and `minimax_h3`. `siglip` supplies the reusable visual tower, and `stub` supplies the deterministic computation the engine's IPC and process tests run without model weights. Import public objects from the package, for example `from uniserve_models.bagel import Config, Model`; package initializers declare exports rather than implement networks.

| Module | Responsibility |
| --- | --- |
| `config.py` | Immutable architecture composition and checkpoint metadata normalization |
| `model.py` | Top-level module composition and numerical entry-point declarations |
| `transformer.py`, `encoder.py`, `denoiser.py` | The corresponding numerical submodule implementations |
| `inputs.py` | Model-specific borrowed numerical inputs |
| `weights.py` | Checkpoint sources, tensor assignments and precision presets; H3's component precision policies live in `precision.py` |
| `processing.py` | Architecture-specific image transforms and prompt framing |

Packages contain the modules their computations need. SigLIP exposes an encoder rather than a serving model, and the deterministic test model has no checkpoint loader. Image/audio/video codecs and other architecture-specific submodules retain their domain names. A submodule's own configuration stays with that submodule when it is independently composed.

Shared discovery and materialization remain in `uniserve_models.loading`; shared preprocessing value types and tokenizer utilities live in `uniserve.processing`, alongside the rest of the computation library. Loadable packages declare `config_sources` for checkpoint headers needed during architecture inspection, and provide their own image-processor factory and flow prompt. These are loading-time assets used by callers, not resources retained by numerical modules.

## Text logits and prefix storage

The [text logits example](../examples/text_logits.py) loads a Qwen3, BAGEL or SenseNova U1 text capability, allocates prefix storage and evaluates raw-text next-token logits through the public numerical interfaces.

```bash
python examples/text_logits.py \
  --checkpoint /path/to/Qwen3-32B \
  --text "Name a primary color." \
  --device cuda:0 \
  --output /path/to/logits.pt
```

The saved dictionary contains CPU `input_ids` shaped `[tokens]` and `logits` shaped `[tokens, vocabulary]`. Each row predicts the token following that prompt position. The input is raw text without chat-template framing. `Logits.gather()` joins vocabulary shards and removes vocabulary padding; sampling is a separate application call.

`TextInput` supplies token IDs, positions, attention inputs and optional embedding replacements or expert routes. `TextSize(num_tokens, batch_size)` bounds one numerical call. `PagedInput.from_blocks` describes query lengths, prefix lengths and logical-to-physical block IDs. Physical block zero is valid. A write index of `-1` skips that token's cache update; `write_indices=None` makes the entire call read-only.

`SequenceLengths(values, offsets, host=None)` borrows int32 device lengths and their prefix sum. Without an exact host mirror, `num_tokens` and `maximum` return `None`; `batch_size` comes from the device tensor's shape. Native attention can consume the device columns directly. `ExecutionContext.bind_attention` obtains a mirror only when backend planning or mathematical token partitioning requires one. Backend authors expose this requirement through `Operator.requires_host_lengths`. Dynamic rotary recipes still require their exact sequence-length domain before numerical evaluation.

`PrefixCache` allocates the numerical states declared by a model's `cache_config`. Its layer states borrow that backing. `cache.mha.State` interprets key/value layouts, partial blocks and encoded scales. Prefix matching, block allocation policy, request ownership and retirement belong to the application or worker.

`State.copy_blocks(source, target)` copies complete state fields, encoded scales and initialization bits from their values at the start of the call, including overlapping source and target blocks. Supply either two index tuples or two equal-length int64 vectors on the backing device. Paired `-1` vector entries skip a copy; valid targets must be distinct. Device vectors can change between CUDA graph replays without a host transfer. Out-of-range indices, unpaired sentinels and repeated targets are errors.

## Execution contexts and CUDA graphs

`ExecutionContext` owns prepared backend state, communication scratch and numerical workspace. Enter the context before calling its module, prepare the required size, and bind attention inputs before graph capture. `bind_attention` refreshes planning metadata; captured calls read numerical tensor contents from their fixed addresses.

The capability runners in `uniserve.execution` provide the ordinary direct-call boundary over an existing context. `TextRunner` binds attention before `forward` and keeps vocabulary projection separate; `EncoderRunner`, `LatentRunner`, `DenoisingRunner`, `ImageRunner`, `VideoRunner`, `VideoProcessor`, and `AudioRunner` expose the corresponding numerical methods. Runners borrow the model and context, connect the context stream back to the caller's current stream, and never own request state or scheduling policy.

```python
from uniserve.execution import TextRunner
from uniserve.runtime import ExecutionContext

with ExecutionContext(model, cache=cache) as execution:
    runner = TextRunner(model, context=execution)
    runner.warmup(size)
    hidden = runner.forward(inputs)
    logits = runner.compute_logits(hidden, token_indices=token_indices)
```

```python
from uniserve.runtime import CUDAGraph, ExecutionContext

with ExecutionContext(model, cache=cache) as execution:
    execution.prepare(size)
    execution.bind_attention(inputs.attention)
    model(inputs)  # Warm the numerical kernel specializations.
    with CUDAGraph(context=execution) as graph:
        graph.capture(lambda: model(inputs))
        result = graph.replay()
        retained = result.clone()
```

The caller completes readers before replaying a graph or replacing its backing. Captured outputs borrow graph storage; copy results that must survive replay. `capture(..., restore=...)` restores caller-owned mutable inputs after capture when the numerical call updates them. Cross-device graphs retain caller-provided CUDA memory pools for their additional devices. Graphs retire before their execution contexts and pools.

Components with numerical storage requirements expose `state_buffers(size)`, `constant_buffers(size)` and `workspace_buffers(size)`. These queries return `BufferConfig` values. `TensorBuffers.allocate` creates backing and `view` lends typed tensors. `ExecutionContext.prepare` prepares constants and workspace, or borrows explicitly supplied `TensorBuffers`; the caller allocates persistent state separately. Re-preparing replaces the context's previous resources after their final readers finish.

Distributed projection iterators borrow context-owned gather storage until their final consumer has enqueued its reads. Serialized calls reuse released storage; overlapping iterator lifetimes receive separate buffers. Text capacities are prepared in advance, while other numerical shapes acquire their backing during eager warmup. Warm every required shape before capture and retain the context until its graphs and readers retire.

Streamed projections and attention exchanges can overlap communication with local computation. The execution context owns their transport streams within the same CUDA or Green Context and joins transfers before their outputs are consumed. Exhaust or close an iterator before reusing its borrowed resources; closing joins its published transfers. Ordinary synchronous collectives preserve ordering with pending streamed transfers.

## Parallel computation and precision

`DeviceMesh` describes ordered ranks, named axes and axis sizes. `initialize_process_groups` creates the process-group owner; its `bind` method returns a mesh carrying borrowed communicators. Models retain numerical communication interfaces without owning process groups, streams or communication backing. `parallelize_` binds mathematical partitions before weight loading, or the public loader applies the supplied `meshes` directly.

`AttentionParallelConfig` selects head exchange with `Ulysses(axis)`, context gathering with `ContextParallelConfig(gather_axis=...)`, or both on independent axes. Axis names refer to the mesh and do not duplicate degree values. Column projections return output-channel shards; row projections sum contraction shards and add their bias once. Vocabulary heads return local logits until explicitly gathered.

`Quantizer("fp8", axis=...)`, `Quantizer("mxfp8")` and `Quantizer("nvfp4")` convert tensors into `QuantizedTensor` values. The logical dtype remains the computation dtype; `buffers()` borrows the encoded values and scales, and `dequantize()` reconstructs dense values. Scale layout is a physical representation choice. Quantization statistics retain their complete logical domain across tensor and sequence partitions. Logical merged branches keep independent weight scales.

Projection `forward_chunks` methods expose `(token_slice, tensor)` or named-projection mappings. Ordered input iterators can connect ready residual chunks to the following projection. Shared layers own exchange and publication ordering; a full-tensor quantizer waits for its complete statistical domain before publishing encoded chunks. VSA consumes explicit Q/K/V/gate chunks, while row projections apply their required reduction and bias.

## Image and video computation

`VisionInput` carries preprocessed images or patches and their grids. Vision encoders return ordered feature tensors; text encoders accept tuples of token sequences; latent encoders retain their posterior sampling semantics. The resolved configuration's `image_processor`, tokenizer path and `flow_prompt` provide caller-owned preprocessing information. Numerical models consume tensors without retaining tokenizers or image-processing runtimes.

Image denoisers declare `latent_shape`, `noise_shape`, `make_schedules` and `make_guidance`. `normal_noise` fills ordered caller-provided views from explicit seeds. `DenoiserInput` contains per-modality `LatentInput` values, sizes and a numerical step index. A forward call predicts values without committing request progress. BAGEL preserves patch-token latent rows and image framing; SenseNova U1 preserves image conditioning, axial positions and its velocity conversion.

`Schedule` stores complete FP32 timestep and sigma endpoints with analytical host coordinates. `Guidance` selects and combines the branches for a supplied schedule coordinate. `EulerSolver` and `CleanSampleEulerSolver` implement their declared prediction conversion and update equations. `DenoisingStep` composes the denoiser, solver and mathematical pipeline feedback using borrowed state; the worker decides which request step to accept.

The [video decoding example](../examples/decode_video.py) loads H3's video decoder and postprocessor, decodes complete latent windows, preserves overlap and returns independent CPU RGB pixels.

```bash
python examples/decode_video.py \
  --checkpoint /path/to/FastH3-4-step-Preview-v1-VSA-DataFree \
  --latents /path/to/video-latents.pt \
  --frames 22 \
  --device cuda:0 \
  --output /path/to/rgb.pt
```

The latent file contains one FP32 tensor saved with `torch.save`, with shape `[((frames - 5) // 17 * 5 + 2) * 24 * 42, 96]` in H3's canonical video order. Join sequence shards in logical order before calling the example. Frame counts have the form `17 * n + 5`, with `n >= 1`. The checkpoint raster is 768 × 1344 at 24 frames per second. The result has CPU `torch.uint8` shape `[frames, 768, 1344, 3]` in RGB order.

`VideoDecoder.frame_slices` identifies legal temporal windows. Its `decode` receives explicit frame slices and complete frame counts; `TensorOutput.layout` describes each window's logical location. `VideoPostprocessor` consumes the supplied overlap state and produces RGB frames in borrowed workspace. Audio decoding receives an explicit sample count and returns sample-major tensors. Encoding a media container is an application responsibility.

## Loading custom modules

`uniserve.loading.load_model` accepts a model constructor, typed configuration, resolved checkpoint sources and a mapping function. `load_weights` applies the same assignment path to an existing module. Mapping functions return `weights.ModuleMapping` values containing `Assignment` records, required and optional parameter names, declared nonresident source names and any checkpoint-derived postprocessing. Shared parameters materialize once, complete-source assignments follow the bound partition, and missing, incomplete or unexpected weights reject loading.

`loading.Config` selects file format, read mode, snapshot revision, file filtering, read concurrency, memory mapping and optional checksums. `checkpoint.Reader` owns scoped file access; `checkpoint.Weight.read` reads a complete tensor or an explicit rectangle. Model constructors consume already-normalized configuration fields. Loading resources close before the materialized numerical model is returned.

Dummy mode initializes deterministic synthetic weights. Parameter-only models do not require checkpoint payloads. When a loading callback derives constants from auxiliary checkpoint tensors, the reader uses checkpoint metadata for their shapes and generates synthetic source values; the callback runs normally and all source assignments retain completeness checks.
