# Python computation library

UniServe's Python loader constructs the same numerical model used by serving. `load_model(LoadRequest(...))` returns its model, tokenizer, resolved execution configuration, schedule, and component bindings. The caller supplies physical placement and owns execution resources; the model receives numerical configuration, borrowed mesh/layer interfaces, and tensor views.

Models compose ordinary `torch.nn.Module` layers with `TextMixin`, `EncoderMixin`, `DiffusionMixin`, `DecoderMixin`, and `VideoMixin`. Text uses `forward`, `compute_logits`, and `embed_input_ids`; diffusion uses `diffusion_spec`, `prepare_latents`, and `forward_diffusion`; encoders and decoders use `encode` and `decode`. A numerical call accepts one homogeneous computation batch. Independent text and diffusion calls use separate execution resources when they run concurrently.

## Compute text logits

The [text logits example](../examples/text_logits.py) loads a Qwen3, BAGEL, or SenseNova checkpoint, binds real KV tensors and the public PyTorch SDPA backend, and evaluates the checkpoint's text capability directly. It uses the shared metadata construction and numerical model implementation without a server or request pool.

```bash
python examples/text_logits.py \
  --checkpoint /path/to/Qwen3-32B \
  --text "Name a primary color." \
  --device cuda:0 \
  --output /path/to/logits.pt
```

The saved dictionary contains CPU `input_ids` shaped `[tokens]` and BF16 `logits` shaped `[tokens, vocabulary]`. Each row predicts the token following that prompt position using causal attention. The input is raw text, without chat-template framing or sampling. The example validates numerical result schemas before joining vocabulary shards and removing padding. Its complete CPU copy precedes KV and process-group retirement.

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

`ImageDiffusion` describes per-image latent bounds, framing markers, patch conversion, noise scaling, CFG semantics, and analytical schedule coordinates. The standard `DiffusionMixin` uses it to produce a `DiffusionSpec` and initialize caller-owned latent state from supplied noise. BAGEL retains patch-token latent rows; SenseNova restores image-space RGB latents through `unpatchify`. Prediction kinds are `velocity` and `sample`.

The execution caller resolves prompt overrides and negative conditioning, generates noise according to the declared RNG contract, owns schedule tensors, and advances the solver. Sequence length includes numerical framing markers; it does not specify how many concurrent trajectories to allocate.

## Encoder inputs

`EncoderMixin.encode` consumes already preprocessed tensors in `EncodeBatch.values` and preserves their row order. Text and conditioning modules process each variable-length row independently and return `conditioning`. VAE encoding stacks images and calls the autoencoder's `encode`, returning `latents` with its posterior sampling semantics intact.

Vision returns `features`. An `ImageProcessor` with `TowerTransform` uses uniform CHW image rows; its `vision_encoder` consumes the stacked NCHW batch. A `PatchTransform` uses packed patch rows plus aligned device `grids` and host-known `grid_shapes`; the shared implementation concatenates patches and splits the result using the declared spatial downsampling ratio. BAGEL composes `nn.vision.PatchEncoder` for patch packing, isolated attention, projection, and learned output positions. SenseNova composes its NEO patch encoder. All inputs must already reside on the participating component device, and shared attention layers require their public backend binding.

## Resource lifetime

`initialize_process_groups` owns process groups, while `initialize_entries` binds logical components to their local meshes. The loading result retains these physical bindings outside the model. `ModelRunner` supplies attention and native decoder execution resources and closes its graphs after outstanding work completes.

`model.tensor_specs(call, shape)` returns numerical requirements for constants, state, scratch, and outputs. `prepare_constants` allocates constants and asks the model to fill borrowed views. `resolve_resources` derives runtime state and scratch backing from the caller's chosen shapes; `TensorBuffers.allocate` creates that backing. `bind_state` and `bind_scratch` lend exact views to a numerical call. A stateless call can use `None` for state backing; a stateful call must receive allocated storage.

`TextShape(tokens, rows=1, selection=None)` describes backbone activation extent. Supply a `TokenSelection` to declare selected hidden or vocabulary rows; selected inactive rows can have zero query tokens. `TextOutput.validate` takes one requirement per sequence before vocabulary materialization. Non-final pipeline activations declare their local sequence region; projected logits declare padded vocabulary storage and its local shard. Text arithmetic dtype comes from the composed decoder's activation representation. H3 encoding and conditioning use positive unselected token extents and preserve unpadded sequence lengths.

`MediaShape` describes spatial and temporal geometry; supply its `dtype` when a computation preserves its input representation, as RGB patch decoding does. Learned VAE reconstruction declares its parameter dtype, and image diffusion declares its prediction dtype independently of solver storage. Vision and latent encoders declare their feature width, spatial downsampling, and output representation through their composed numerical modules. Calls without a numerical declaration raise an error instead of supplying an empty resource contract.

`VideoAttention.tensor_specs(rows, query_rows, dtype=...)` declares projection exchange and attention output scratch. The shared backing accommodates both encoded projection inputs and activation-format return rows; callers allocate and retain the declared capacity regardless of weight precision. H3 consumes this same public layer declaration.

`TensorOutput.validate(needs, state=state, scratch=scratch)` checks names, dtypes, shapes, local regions, and declared storage borrowing without reading device values. Supply one `TensorNeeds` for a common shape or an aligned sequence for different row shapes. An output with `TensorAlias("scratch", "rgb_frames")` borrows that explicit scratch tensor's storage; H3 postprocessing uses this contract for each returned RGB prefix. Validation does not extend the backing lifetime or synchronize readers.

Each independent request or trajectory needs its own mutable state. Scratch can be reused only after its final reader completes. Decoder graph outputs and pixel scratch can be overwritten by the next call, so the example completes a CPU copy of each window before reusing them. Constants and all captured tensor addresses remain alive until the corresponding graphs retire. Numerical objects and borrowed views leave scope before the process-group owner closes.

`Model.output_shapes` declares maximum numerical input shapes for named result components. Runtime derives bounded wire products from the corresponding `TensorNeeds.outputs`, preserving global output extent independently of local shard regions. Applications performing only numerical computation can consume `TensorOutput` directly and choose their own serialization.
