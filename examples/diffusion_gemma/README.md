# DiffusionGemma examples

These examples use the same public numerical modules and HTTP endpoint as UniServe serving. Install the repository's GPU environment; the HTTP adapter also uses the `bench` extra for aiohttp.

## Python numerical readout

```bash
.venv/bin/python examples/diffusion_gemma/readout.py \
  /workspace/models/diffusiongemma-26B-A4B-it \
  --state 'The door is open.' \
  --question 'Is the door open?'
```

The example loads the text and token-denoising components, owns a fresh KV cache and execution context, fills the prompt cache, and reads one yes/no position in a fixed canvas. It prints the normalized candidate probabilities and their mass in the full vocabulary. It performs no autoregressive sampling. The prompt and scaffold are a small standalone demonstration; use `/v1/systemone` for the full question, candidate, and image contract.

## DJev HTTP envelope

Start UniServe with the desired checkpoint and numerical readout configuration, then start the adapter:

```bash
.venv/bin/uniserve serve /workspace/models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma --host 127.0.0.1 --port 8000
.venv/bin/python examples/diffusion_gemma/djev_adapter.py \
  --upstream http://127.0.0.1:8000 --model diffusiongemma --port 8765
```

```bash
curl http://127.0.0.1:8765/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"states":[{"id":"door","state":"The door is open.","questions":{"open":{"type":"boolean","instructions":"Is the door open?","criteria":{"true":"Open","false":"Closed"}}}}]}'
```

Each state becomes one System One request. The adapter preserves state order and IDs, maps boolean/choice/score responses, and forwards optional `images` as `x_images`. `GET /api/health` checks the upstream service. Unsupported nonempty `options` are rejected, and upstream validation or capacity errors remain visible. Numerical settings, batching, caching and model execution belong to the upstream server. The adapter's elapsed time includes HTTP transport; its network call count is the number of states, not a GPU forward count.
