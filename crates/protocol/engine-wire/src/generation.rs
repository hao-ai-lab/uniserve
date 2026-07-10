//! Typed generation results and model control tokens carried by the
//! engine protocol.

use serde::{Deserialize, Serialize};
/// One typed image event riding on an [`crate::EngineCoreOutput`].
///
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
    /// The generated image was committed by the worker.
    Commit { image_id: u32 },
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

/// Generation statistics attached to terminal output so the frontend can
/// reconstruct a typed `Finished` event without engine-side state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationFinish {
    /// Finish reason (`eos`, `max_tokens`, `stop`, `image_done`,
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

/// Typed generation payload carried by [`crate::EngineCoreOutput`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// A typed image event, when this output carries one.
    #[serde(default)]
    pub image: Option<WireImageEvent>,
    /// Finish statistics, set on terminal output.
    #[serde(default)]
    pub finish: Option<GenerationFinish>,
}

/// Model control-token ids resolved from the tokenizer by the frontend and
/// shipped to the engine in the handshake INIT message for token feedback and
/// terminal matching while the tokenizer remains frontend-owned.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationControlTokens {
    /// Turn-open token (`<|im_start|>` in the BAGEL/ThinkMorph convention).
    pub bos: u32,
    /// EOS / turn-close token ids.
    pub eos: Vec<u32>,
    /// Image span end (`<|vision_end|>`).
    pub end_of_image: u32,
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
            ("Commit", WireImageEvent::Commit { image_id: 1 }),
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
            WireImageEvent::Commit { image_id: 1 },
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
}
