# DiffusionGemma computation

The public loader constructs DiffusionGemma from immutable typed configuration and checkpoint mappings. Its causal language model and `TokenDenoiser` share one transformer backbone and vocabulary head. The vision tower uses the shared patch encoder and image-processing descriptors. BF16 and per-expert NVFP4 checkpoints use the same model composition and shared routed-expert layers.

A causal `forward` writes prompt KV and returns hidden rows. `fill_cache` writes the same KV when no output rows are needed. A `CanvasInput` supplies packed token IDs, absolute positions, table-indexed attention and optional soft embeddings from the preceding pass. The token denoiser reads the retained prefix and every current token of its own canvas without appending canvas KV. `compute_logits` projects selected hidden rows through the shared, softcapped vocabulary head. A readout may split `attend` and `finish` to complete only selected rows after the final attention operation.

`uniserve.diffusion.tokens` defines the temperature schedule, entropy-bound acceptance, Philox draws, re-noising, stopping, EOS truncation and self-conditioning equations. Random draws are keyed by request seed and indexed by block, step, position and stream, so batch composition does not change a request's random bits. `uniserve.diffusion.canvas` applies one step to borrowed resident tensors and provides the same numerical contract through portable formulas and SM100 kernels. A zero confidence threshold runs all configured steps. The caller selects steps, commits accepted blocks and owns request progress.

Cache allocation, native layer preparation, streams and graph lifetimes belong to `PrefixCache` and `ExecutionContext`, as in ordinary causal inference. Image slots can use `PatchEncoder.encode_packed`: each slot's image features equal encoding that image separately, and device grid values can change between graph replays.

## Python example

The [numerical readout example](../../examples/diffusion_gemma/readout.py) loads the text and denoising components, fills a fresh prompt cache and evaluates a fixed yes/no canvas:

```bash
.venv/bin/python examples/diffusion_gemma/readout.py \
  /models/diffusiongemma-26B-A4B-it \
  --state 'The door is open.' \
  --question 'Is the door open?'
```

The output contains probabilities normalized over the two candidates and their probability mass in the full vocabulary. Candidate mass distinguishes a strong preference between the named answers from a model distribution that primarily favors other tokens. This example performs one numerical denoising pass and owns the lifetime of its cache and execution context.
