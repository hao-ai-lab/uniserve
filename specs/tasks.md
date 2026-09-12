# Reference-conditioned H3 serving

## Current boundary

The HTTP video schema and Rust lowering preserve ordered reference descriptors and validate modality/task/role combinations, source encoding, soundtrack selection, and bundle limits. The runtime rejects nonempty bundles before engine submission because its media execution plan is text-only. Descriptor parsing is not reference serving. The existing Python semantic layout and audio posterior encoder are not wired into admission or denoising.

## Request contract

`references` defaults to `[]`. Each item contains `type: image|video|audio`, `task: reference|first_frame|first_last_frame|continue_scene|continue_shot`, `role: reference|first_frame|last_frame|preceding`, and `source: {"type": "url"|"base64", "value": "..."}`. Video sources must specify boolean `include_audio`; other modalities must omit it. The generic task uses role `reference`; frame tasks use matching image roles; continuation tasks use video role `preceding`. Task labels do not request FL2VA exact endpoint anchoring. Source order is semantic and must survive all lowering and projection stages.

At most 12 sources are admitted, with at most 9 images, 3 videos and 3 standalone audio sources, and at least one visual source. Standard base64 payloads have a 32-MiB decoded per-source bound and a 96-MiB bundle bound. The server's independent 64-MiB HTTP body limit also applies. URLs have an 8192-byte descriptor bound and require HTTP(S) syntax; this is not network destination authorization. Debug formatting redacts both sources and URLs.

## Required implementation

1. Finish shared media admission and IPC: an SSRF-safe bounded fetcher must validate resolved addresses, bind connections to validated destinations, disallow unauthorized redirect destinations, bound streamed bytes and decode resources, and transport decoded geometry and products through request-owned IPC lifetimes. Encoded source limits do not bound decoded dimensions or duration. Cache keys must include the complete reference geometry, not only target/prompt dimensions.
2. Prepare ordered Qwen image/video presentations with the vision tower and DeepStack weights. Add causal video-VAE encoding with the FastVideo sampling, tiling, clip assembly and FP16-round-trip contract; prepare audio with the fixed resampling/window policy before calling the existing audio encoder. Encoding must execute once per reference and preserve caller order.
3. Resolve a separate Ref2VA checkpoint identity and load only its top-level `transformer_ref` denoiser. The base first entry is `MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`. Distilled reference exports require a separate explicit reference-task contract rather than relaxing the existing T2AV identity.
4. Lower semantic reference layouts into bounded resident tensors, preserving projection order across sequence ownership. Implement fixed reference modulation clocks, one-time visual augmentation, and target-only solver/decode state. Qwen visual boundary tokens remain video-tagged.
5. Transfer `p2_multi_region` reference-aware VSA from FastVideo commit `85d0955098ad3fa024290cf56a41f251fa99d2f3`, including per-region tile spans, target and reference selection rates, prefix visibility, padding exclusion, and pooled-attention behavior. The source is the approved `omniref-pdd8` checkout, not main's single-region implementation.
6. Exercise each public boundary with behavioral CPU tests and real external-media fixtures or external codec doubles. Existing no-reference lowering equality does not establish generated-output byte identity; preserve the original checkpoint/protocol for that regression.

## Completion evidence

A nonempty `/v1/videos` request must execute genuine `transformer_ref` conditioning and return a decodable MP4 with the declared audio policy. Require ordered modality/task cases, explicit soundtrack behavior, unchanged reference rows across every solver interval, target-only decoding, strict checkpoint partition identity, reference-aware attention parity, and unchanged no-reference output behavior. No T2VA recipe or semantic-layout-only test substitutes for these conditions.

GPU execution belongs to separately registered orchestrator runs, with every scheduler attempt attached. Fix source commits, model identities, media receipts, shapes, sampling, precision, attention policy and tolerances before measuring. Execute points serially: base reference loading; ordered modality/task cases; injected-noise transformer comparison; fixed-reference/target-only trajectories; SP1/SP4 parity; reference VSA mask/output parity; end-to-end MP4/audio comparisons; original no-reference regression. Numerical tolerances and media comparisons must be independently justified before these runs; no GPU measurements or accepted reference-serving command exist here.
