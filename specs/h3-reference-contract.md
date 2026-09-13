# H3 ordered reference contract

## Scope

The reference checkpoint entry uses transformer_ref from hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da. Base and eight-step entries remain unchanged. The intended admission contract supports first_frame with one image, first_last_frame with two images in first/last order, storyboard with two through five images in supplied order, and continue_scene or continue_shot with one video and its embedded soundtrack when present. Images and video are not mixed in a request. Capability publication must follow implementation across admission, transport, conditioning, and packed execution, not precede it.

## Numerical source

FastVideo aa11247f7d9ba73fbf69ddd4f80709348c1d82c2 defines media preparation in fastvideo/pipelines/basic/minimax_h3/reference.py, presentation in stages/minimax_h3_conditioning.py, and posterior encoding in stages/minimax_h3_latent_preparation.py. The delivery contract at scripts/preprocess/minimax_h3_omniref_delivery/contract.py resolves ordered image labels and preceding-video inputs.

Images use EXIF-corrected RGB and a 2048-short-edge, multiple-of-32 canvas with aspect ratio within 1:4 through 4:1. Videos use a 24-fps nearest-resampled timeline and the released 768-short-edge canvas with its area cap. Qwen sees 2-fps samples with temporal-patch padding; the visual VAE sees complete 17n+5 chunks. These are distinct frame selections. Embedded audio uses the selected untrimmed presentation window at 32 kHz, not the shorter visual-VAE chunk window. Mono is duplicated, stereo is preserved, and multichannel audio requires stereo downmix before audio encoding. Presentation labels are <Picture 1> through <Picture K>, or <Video 1> with the upstream soundtrack presentation.

## Ownership boundaries

- HTTP admission owns ordered task/kind/role validation, inline source size limits, and actionable 400 responses. Remote URLs are outside this contract.
- Media preparation owns bounded decode, EXIF/rotation handling, canvas normalization, timeline selection, and embedded-audio decode. Bounds must apply during decode, before accumulating an unbounded raster or waveform.
- Request transport must retain every source and its audio product in order. DecodedReference already declares pixel/audio products and rational fps; the engine currently populates only a single image product through DiffusionRequest.image_reference.
- H3 conditioning owns Qwen presentation and tags. Presentation token count is checked before language execution.
- H3 geometry owns visual/audio row counts and the complete packed budget, including alignment and target transport padding. A presentation-only bound is insufficient.
- H3 preparation owns seeded visual posterior sampling and audio posterior modes through the resident VAEs, with fixed conditions separate from target solver state.
- Semantic reference geometry already supports multiple visual/audio media. Resident packing, scratch capacity, input staging, and preparation currently specialize it to one image and must be generalized together.

## Completion evidence

Required evidence comprises observable role/count rejection, exact budget boundaries, deterministic CPU HTTP-to-transformer tests for two images, three storyboard images, and a video with audio, plus the workspace Rust tests and Python suite. CPU tests cannot establish GPU numerical parity or model quality; no such claim follows from fixture encoders.
