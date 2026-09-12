# H3 reference checkpoint contract

`resolve_catalog_entry(("minimax-h3-ref",), root=root)` validates the top-level pinned MiniMax-H3 repository and selects `transformer_ref` for the denoiser, with shared `text_encoder`, `vae`, and `audio_vae` components. `resolve_base_h3_contract(root, reference=True)` exposes its recipe and capabilities. The immutable model identity is `hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`; FastVideo reference code is `a943220c115228ade5d57b3bab9a6a87fd600a10`.

Local roots require revision receipts for every consumed configuration, index and shard. The denoiser requires 14 indexed shards; nested task exports, missing components, unsafe shard paths, other revisions and distilled manifests fail validation. Revision receipts establish provenance, not recomputed weight hashes.

The contract fixes output to 480×832×124 at 24 FPS, with 32 kHz audio. Sampling uses the base CPU-FP32 50-point uniform grid, shifts 12/3, 49 transformer forwards and guidance 1.0. Attention is dense for both reference and target spans; reference-aware VSA is not implemented. The capability is `references: {max: 1, kinds: [image]}`.

By default, checkpoint inspection and worker catalog resolution automatically select the reference recipe when a local pinned root contains a top-level `transformer_ref`. Its configuration and weight sources are validated instead of the base denoiser. Explicit distilled manifests and operator sidecars retain precedence; a root without `transformer_ref` selects the base recipe.

## Execution boundary

Catalog selection is not HTTP admission authorization. The serving and worker execution paths must propagate expanded presentation geometry and tags, decoded pixels, and the image-VAE product before the capability can be advertised by a live endpoint. Target layout, packed latent rows, request storage, decoder scratch, output schemas, warmup inputs, and RGB assembly use the checkpoint contract's spatial geometry. Reference workers reserve 124 frames at 480×832 and reject configured temporal capacities smaller than that recipe; base and distilled workers retain their configured frame capacity at 768×1344. CPU tests cover contract-driven worker products, exact no-reference noise bytes, decoder latent windows, and exact RGB overlap/normalization products at both resolutions. Neural decoder forwards and full-request reference execution require separate GPU evidence. Admission remains closed until that connected path is implemented and tested.
