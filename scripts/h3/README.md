# Single-model H3 deployment

`serve-h3.sbatch` serves exactly one checkpoint contract on one rack-3 four-GPU tray. Copy `serve-h3.sbatch`, `h3-endpoint.py`, and `smoke-h3.py` together to `/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy/`. The source checkout is `/mnt/lustre/vlm-wlsaidhi/src/uniserve-wt/omniref`; launch from a clean, recorded commit.

| ENTRY | Checkpoint root | Model ID | Port | Recipe |
| --- | --- | --- | --- | --- |
| base | `/mnt/lustre/vlm-k1kong/models/MiniMax-H3` | `minimax-h3-base` | 8100 | Dense, 50-point grid / 49 forwards, quality precision |
| 8step | `/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10` | `minimax-h3-8step` | 8101 | VSA-H3 0.8, eight forwards, balanced precision |
| ref | `/mnt/lustre/vlm-k1kong/models/MiniMax-H3` | `minimax-h3-ref` | 8102 | Dense reference, 50-point grid / 49 forwards, quality precision |

Base weights are `hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`; the matching FastVideo source is `a943220c115228ade5d57b3bab9a6a87fd600a10`. The eight-step manifest identifies FastVideo source `24bbe7fddd05ca6f2c34b3dbed06ac1c75b72086`, ladder `[999,874,749,624,500,375,250,125]`, shifts 10/3, and terminal zero. This is not the uniform nine-point scheduler.

The launcher sets `UNISERVE_H3_VARIANT=base` to select the base denoiser from a repository that also contains `transformer_ref`. The shared resolver accepts explicit `base` or `ref`, rejects other values and rejects overrides of distilled manifests. Without an explicit selection it retains checkpoint discovery. A served-model alias alone does not select a recipe.

## Reference entry

Copy `serve-h3-ref.sbatch` and `smoke-h3-ref.py` to `/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy/`. The reference launcher uses the endpoint helper in the source checkout and accepts the checkpoint root positionally. The orchestrator, not the launcher, owns submission and run registration:

```bash
sbatch --export=NIL serve-h3-ref.sbatch ref /mnt/lustre/vlm-k1kong/models/MiniMax-H3
python3 smoke-h3-ref.py /absolute/path/to/reference.png
```

The reference entry uses 1024 prompt tokens, a fixed 480×832×124 target, and one inline PNG/JPEG reference at most. Input axes must be divisible by 32 and no larger than 4096; compressed source bytes must fit 32 MiB. The smoke sends text-only and single-image requests serially, saving two MP4s under `/mnt/lustre/vlm-wlsaidhi/fastvideo/eval/uniserve-smoke/ref/`. CPU input validation does not submit the launcher or run media generation. A root containing `transformer_ref` also selects this contract through `uniserve serve <root> --model-description minimax-h3` without an explicit variant override.

CPU input-path evidence is `OMP_NUM_THREADS=2 .venv/bin/pytest -q tests/python/e2e/test_h3_reference_cpu.py`, after `cargo build -p uniserve-server --example h3_reference_cpu` and building the current Python IPC extension. This test uses synthetic weights and the production HTTP/engine/IPC path, stopping at transformer inputs rather than generating MP4s.

## Allocation setup

Hermes registers the serving run and submits the job; these scripts never create records or submit/cancel jobs. Pass the registered identity as an additional `RUN_ID` export for runtime/finalization and endpoint linkage. Omitting it permits the minimal launcher command but leaves registry linkage to the submitting orchestrator.

```bash
cd /mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy
sbatch --export=NIL,ENTRY=base,CHECKPOINT_ROOT=/mnt/lustre/vlm-k1kong/models/MiniMax-H3,RUN_ID=REGISTERED_ID serve-h3.sbatch
sbatch --export=NIL,ENTRY=8step,CHECKPOINT_ROOT=/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10,RUN_ID=REGISTERED_ID serve-h3.sbatch
```

The allocation uses the existing ARM64 CUDA container and API-key file conventions. Rust resides under `uniserve-deploy/toolchain/{cargo,rustup}` and uv under `uniserve-deploy/toolchain/uv/uv`. Inside the writable container, build prerequisites are installed and `uv sync --locked --extra h3` builds/updates this checkout's `.venv-gpu`, following the existing worker environment convention without importing another checkout's extension. The default existing server checkout has no provisioned `.venv-gpu`; a successful allocation build remains necessary. Package repositories must be reachable. No dependencies are installed on the login node by the launcher.

The base and eight-step entries use eager execution, one resident request, a 512-token text capacity, and five-second video capacity. Base quality precision retains FP32 video VAE; eight-step balanced precision follows the existing FastH3 launcher. These are launch configurations, not performance or parity claims.

An observer publishes `endpoints/h3-<entry>-rack3.json` atomically only after health succeeds and `/v1/models` returns exactly the selected ID. It uses the compute IP and scheduler deadline, stores only the key-file path, and removes its own allocation's record on normal exit or catchable termination. SIGKILL/node loss cannot run shell cleanup; consumers must honor expiry. The server's lifetime is not controlled by endpoint diagnostics.

## Smoke and CPU checks

```bash
python3 smoke-h3.py --entry base
python3 smoke-h3.py --entry 8step
```

Each smoke reads its ready endpoint, checks expiry, `/v1/models`, and health, then makes exactly one text-only `POST /v1/videos/sync` request (seconds 5, seed 0; explicit steps 50 for base). H3 admission maps this duration to 124 frames at 1344×768 / 24 FPS. MP4 and timing/request JSON are saved under `/mnt/lustre/vlm-wlsaidhi/fastvideo/eval/uniserve-smoke/<entry>/`. The client does not assert decoded frame count or numerical parity.

`bash -n serve-h3.sbatch` and `python3 -m py_compile smoke-h3.py h3-endpoint.py` check syntax. `python3 test_endpoint.py` exercises publication and allocation-owned cleanup against an isolated HTTP service. The Rust CLI has no dry-run/config-only command; `uniserve_worker.bootstrap.inspect_model` validates checkpoint provenance, component metadata, and resolved contracts without GPU weight allocation. Neither this inspection nor endpoint readiness establishes GPU correctness or media quality.
