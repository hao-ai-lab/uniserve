//! UniServe protocol extensions for native generation.
//!
//! This is the one deliberate fork from the upstream vllm-rs wire protocol
//!: the upstream `EngineCoreRequest`/`EngineCoreOutput`
//! cannot express image generation, so UniServe appends one optional extension
//! field to each (`native`) and one to the handshake INIT message
//! (`native_controls`). Plain text traffic leaves them all `None`, keeping the
//! upstream encoding shape; native traffic is impossible to confuse
//! with upstream messages because the extension is a dedicated trailing slot.
//!
//! Everything here is a descriptor or small result (token ids, dimensions,
//! finished PNG bytes).

use serde::{Deserialize, Serialize};
use uniserve_core::{GenerationConstraint, ImageParams};

/// A staged multimodal input item on the wire: an input image referenced by
/// content hash that occupies `num_tokens` positions in the AR sequence once
/// its encoder embeddings are spliced in. The base64 bytes travel southbound as
/// model *input*; no embedding ever returns to the host.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WireMmItem {
    /// Content hash (encoder-cache key).
    pub hash: u64,
    /// Start position of this item's span within the flattened prompt ids.
    pub position: u32,
    /// Number of AR positions the encoder output occupies (0 = worker decides).
    pub num_tokens: u32,
    /// Input-image bytes (base64 PNG/JPEG).
    #[serde(default)]
    pub b64: String,
}

/// UniServe extension to [`crate::EngineCoreRequest`]: the native generation
/// parameters the upstream text protocol cannot express.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeRequestExt {
    /// Output constraint applied to the default generation paradigm.
    #[serde(default)]
    pub constraint: GenerationConstraint,
    /// Image diffusion parameters.
    pub image: ImageParams,
    /// CFG text-unconditional / image precontext prompt (may be empty).
    #[serde(default)]
    pub neg_prompt_ids: Vec<u32>,
    /// Staged multimodal input items (encoded before prefill).
    #[serde(default)]
    pub mm_items: Vec<WireMmItem>,
}

/// One typed image event riding on an [`crate::EngineCoreOutput`].

/// Encoded with serde's default *externally tagged* representation: the variant
/// is carried as the (compact) map key chosen by the binary codec rather than a
/// stringly-typed internal `kind` field, which both avoids serde's buffering
/// deserialize path and keeps the hot output wire small. The discriminant never
/// crosses a language boundary — the only consumers are the same-crate
/// translators in `translate.rs`, which match on the variants directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WireImageEvent {
    /// Diffusion for image `image_id` started.
    Begin {
        image_id: u32,
        height: u32,
        width: u32,
        steps: u16,
    },
    /// One denoise step completed.
    Step { image_id: u32, step: u16 },
    /// The image finished and committed; `png_b64` carries the encoded pixels
    /// (a small *result*, not KV). Dimensions, byte count, and
    /// checksum describe the encoded PNG bytes, not merely the requested
    /// control-plane size.
    Done {
        image_id: u32,
        #[serde(default)]
        height: u32,
        #[serde(default)]
        width: u32,
        #[serde(default)]
        bytes: u64,
        #[serde(default)]
        sha256: String,
        png_b64: String,
    },
}

/// Native finish statistics attached to the terminal output of an
/// native generation request, so the frontend can reconstruct the typed
/// `Finished` event without engine-side state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NativeFinishExt {
    /// Native finish reason (`eos`, `max_tokens`, `stop`, `image_done`,
    /// `cancelled`, `aborted`, `error`).
    pub reason: String,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub images: u64,
    /// Human-readable detail for `rejected` / `error` terminations.
    #[serde(default)]
    pub message: Option<String>,
}

/// UniServe extension to [`crate::EngineCoreOutput`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NativeOutputExt {
    /// A typed image event, when this output carries one.
    #[serde(default)]
    pub image: Option<WireImageEvent>,
    /// Native finish statistics, set on the terminal output.
    #[serde(default)]
    pub finish: Option<NativeFinishExt>,
}

/// Model control-token ids resolved from the tokenizer by the frontend and
/// shipped to the engine in the handshake INIT message (a UniServe extension to
/// `HandshakeInitMessage`): the engine's scheduler lifecycle decisions need them, but the
/// tokenizer lives with the frontend.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeControlTokens {
    /// Turn-open token (`<|im_start|>` in the BAGEL/ThinkMorph convention).
    pub bos: u32,
    /// EOS / turn-close token ids.
    pub eos: Vec<u32>,
    /// Image span start (`<|vision_start|>`).
    pub start_of_image: u32,
    /// Image span end (`<|vision_end|>`).
    pub end_of_image: u32,
    /// Token-id subsequence of the literal `<image_start>` visual-thinking
    /// trigger (ThinkMorph). Empty disables the literal trigger.
    #[serde(default)]
    pub image_start_ids: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_msgpack, decode_value, encode_msgpack};

    /// Guards the externally-tagged wire shape: each `WireImageEvent` must encode
    /// as a single-entry map keyed by its variant name, with no stringly-typed
    /// internal `kind` tag riding on the hot output wire.
    #[test]
    fn image_event_externally_tagged() {
        let cases = [
            (
                "Begin",
                WireImageEvent::Begin {
                    image_id: 1,
                    height: 512,
                    width: 512,
                    steps: 50,
                },
            ),
            (
                "Step",
                WireImageEvent::Step {
                    image_id: 1,
                    step: 7,
                },
            ),
            (
                "Done",
                WireImageEvent::Done {
                    image_id: 1,
                    height: 512,
                    width: 512,
                    bytes: 3,
                    sha256: String::new(),
                    png_b64: "QUJD".into(),
                },
            ),
        ];
        for (variant, ev) in cases {
            let bytes = encode_msgpack(&ev).unwrap();
            let value = decode_value(&bytes).unwrap();
            let map = value
                .as_map()
                .unwrap_or_else(|| panic!("{variant} should encode as a map, got {value}"));
            assert_eq!(
                map.len(),
                1,
                "{variant} should be a single-entry tagged map"
            );
            let key = map[0].0.as_str().expect("variant key must be a string");
            assert_eq!(
                key, variant,
                "variant must be the map key, not an inner field"
            );
            assert_ne!(key, "kind", "no internal `kind` tag on the wire");
        }
    }

    #[test]
    fn image_event_roundtrip() {
        for ev in [
            WireImageEvent::Begin {
                image_id: 1,
                height: 512,
                width: 512,
                steps: 50,
            },
            WireImageEvent::Step {
                image_id: 1,
                step: 7,
            },
            WireImageEvent::Done {
                image_id: 1,
                height: 512,
                width: 512,
                bytes: 3,
                sha256: "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78".into(),
                png_b64: "QUJD".into(),
            },
        ] {
            let bytes = encode_msgpack(&ev).unwrap();
            let back: WireImageEvent = decode_msgpack(&bytes).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn native_request_ext_roundtrip() {
        let ext = NativeRequestExt {
            constraint: GenerationConstraint::Default,
            image: ImageParams::default(),
            neg_prompt_ids: vec![1, 2],
            mm_items: vec![WireMmItem {
                hash: 7,
                position: 3,
                num_tokens: 0,
                b64: "QQ==".into(),
            }],
        };
        let bytes = encode_msgpack(&ext).unwrap();
        let back: NativeRequestExt = decode_msgpack(&bytes).unwrap();
        assert_eq!(back, ext);
    }
}
