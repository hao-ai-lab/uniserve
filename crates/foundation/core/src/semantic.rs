//! Dormant canonical semantic request contracts.
//!
//! One ordered semantic context plus typed output intents replaces top-level request variants: raw prompts,
//! pre-tokenized input, chat messages, tool results, and media are context
//! entries, never runtime request shapes, and every requested output is an
//! explicit typed [`OutputIntent`] with a finite bound.
//!
//! Text, Image, Audio, and Video are peer content and output kinds here;
//! interaction semantics (roles, turns, tools) are data on context entries.
//! Media travels as immutable digest-addressed [`ArtifactRef`] values — raw
//! bytes never appear in canonical values. Nothing in production consumes this
//! module yet; the current `crates/frontend/serving` request shapes remain
//! authoritative until the whole-slice cutover.

use serde::{Deserialize, Serialize};

/// One externally consumable immutable blob (image, audio, video, manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: u64,
    /// Content digest (SHA-256) binding the reference to exact bytes.
    pub digest: [u8; 32],
    pub bytes: u64,
    pub descriptor: MediaDescriptor,
}

/// Typed media descriptors; bounds are declared, never inferred at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaDescriptor {
    Image(ImageDescriptor),
    Audio(AudioDescriptor),
    Video(VideoDescriptor),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageDescriptor {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDescriptor {
    pub sample_rate: u32,
    pub channels: u16,
    /// Exact or declared upper-bound sample count; unbounded audio is not
    /// compilable context.
    pub max_samples: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoDescriptor {
    pub width: u32,
    pub height: u32,
    /// Frames per second expressed as an exact rational timebase.
    pub frame_rate: (u32, u32),
    /// Exact or declared upper-bound frame count.
    pub max_frames: u64,
}

/// Who supplied one context entry; interaction structure, not a modality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextOrigin {
    Prompt,
    Message { turn: u32, role: ContextRole },
    ToolResult { turn: u32, call_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// One typed semantic input value. Order inside an entry is preserved
/// exactly; media never flattens into placeholder text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Pre-tokenized content pinned to an exact tokenizer fingerprint; a
    /// representation of input, not a separate execution path.
    Tokens {
        tokenizer_fingerprint: [u8; 32],
        token_ids: Vec<u32>,
    },
    Media {
        artifact: ArtifactRef,
    },
    ToolCall {
        call_id: u64,
        name: String,
        arguments: String,
    },
    ToolResult {
        call_id: u64,
        content: String,
    },
}

/// One ordered context entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextEntry {
    pub origin: ContextOrigin,
    pub parts: Vec<ContentPart>,
}

/// The ordered semantic input truth. Required input capabilities derive from
/// this content; no separate modality boolean matrix exists to contradict it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticContext {
    pub primary: Vec<ContextEntry>,
    pub negative: Vec<ContextEntry>,
}

/// One requested output: a stable id, typed kind, and finite bounds. Output
/// behavior is never inferred from endpoint names or task labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputIntent {
    pub output_id: u16,
    pub cardinality: u16,
    pub kind: OutputKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Text {
        max_tokens: u32,
    },
    Image {
        width: u32,
        height: u32,
    },
    Audio {
        max_samples: u64,
        sample_rate: u32,
    },
    Video {
        max_frames: u64,
        width: u32,
        height: u32,
    },
}

/// A semantic request shape that violates the closed contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SemanticValidationError {
    #[error("semantic context has no primary entries")]
    EmptyContext,
    #[error("context entry {0} has no content parts")]
    EmptyEntry(usize),
    #[error("request declares no output intents")]
    NoOutputs,
    #[error("duplicate output id {0}")]
    DuplicateOutput(u16),
    #[error("output {0} declares zero cardinality")]
    ZeroCardinality(u16),
    #[error("output {0} declares a zero bound")]
    ZeroBound(u16),
    #[error("media artifact {0} declares zero extent")]
    EmptyArtifact(u64),
}

/// Validate the canonical-shape obligations that are decidable on the value
/// alone: nonempty ordered context, nonempty typed outputs with unique ids and
/// finite nonzero bounds, and finite media extents. Capability intersection
/// and profile validation happen at compilation with runtime context.
pub fn validate_semantic_shape(
    context: &SemanticContext,
    outputs: &[OutputIntent],
) -> Result<(), SemanticValidationError> {
    if context.primary.is_empty() {
        return Err(SemanticValidationError::EmptyContext);
    }
    for (index, entry) in context
        .primary
        .iter()
        .chain(context.negative.iter())
        .enumerate()
    {
        if entry.parts.is_empty() {
            return Err(SemanticValidationError::EmptyEntry(index));
        }
        for part in &entry.parts {
            if let ContentPart::Media { artifact } = part {
                let empty = artifact.bytes == 0
                    || match &artifact.descriptor {
                        MediaDescriptor::Image(image) => image.width == 0 || image.height == 0,
                        MediaDescriptor::Audio(audio) => audio.max_samples == 0,
                        MediaDescriptor::Video(video) => video.max_frames == 0,
                    };
                if empty {
                    return Err(SemanticValidationError::EmptyArtifact(artifact.artifact_id));
                }
            }
        }
    }
    if outputs.is_empty() {
        return Err(SemanticValidationError::NoOutputs);
    }
    let mut seen = std::collections::BTreeSet::new();
    for intent in outputs {
        if !seen.insert(intent.output_id) {
            return Err(SemanticValidationError::DuplicateOutput(intent.output_id));
        }
        if intent.cardinality == 0 {
            return Err(SemanticValidationError::ZeroCardinality(intent.output_id));
        }
        let bound = match &intent.kind {
            OutputKind::Text { max_tokens } => u64::from(*max_tokens),
            OutputKind::Image { width, height } => u64::from(*width) * u64::from(*height),
            OutputKind::Audio { max_samples, .. } => *max_samples,
            OutputKind::Video {
                max_frames,
                width,
                height,
            } => *max_frames * u64::from(*width) * u64::from(*height),
        };
        if bound == 0 {
            return Err(SemanticValidationError::ZeroBound(intent.output_id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_entry(text: &str) -> ContextEntry {
        ContextEntry {
            origin: ContextOrigin::Message {
                turn: 0,
                role: ContextRole::User,
            },
            parts: vec![ContentPart::Text {
                text: text.to_owned(),
            }],
        }
    }

    fn image_part() -> ContentPart {
        ContentPart::Media {
            artifact: ArtifactRef {
                artifact_id: 7,
                digest: [1; 32],
                bytes: 1024,
                descriptor: MediaDescriptor::Image(ImageDescriptor {
                    width: 2048,
                    height: 1152,
                }),
            },
        }
    }

    #[test]
    fn ordered_mixed_content_uses_one_context_shape() {
        let context = SemanticContext {
            primary: vec![
                text_entry("describe"),
                ContextEntry {
                    origin: ContextOrigin::Message {
                        turn: 0,
                        role: ContextRole::User,
                    },
                    parts: vec![
                        image_part(),
                        ContentPart::Text {
                            text: "then draw".into(),
                        },
                    ],
                },
            ],
            negative: vec![text_entry("blurry")],
        };
        let outputs = [
            OutputIntent {
                output_id: 0,
                cardinality: 1,
                kind: OutputKind::Text { max_tokens: 256 },
            },
            OutputIntent {
                output_id: 1,
                cardinality: 1,
                kind: OutputKind::Image {
                    width: 2048,
                    height: 1152,
                },
            },
        ];
        validate_semantic_shape(&context, &outputs).expect("mixed context is one shape");
    }

    #[test]
    fn empty_context_outputs_and_bounds_fail_closed() {
        let empty = SemanticContext {
            primary: vec![],
            negative: vec![],
        };
        assert_eq!(
            validate_semantic_shape(&empty, &[]),
            Err(SemanticValidationError::EmptyContext)
        );
        let context = SemanticContext {
            primary: vec![text_entry("hi")],
            negative: vec![],
        };
        assert_eq!(
            validate_semantic_shape(&context, &[]),
            Err(SemanticValidationError::NoOutputs)
        );
        let zero_bound = [OutputIntent {
            output_id: 0,
            cardinality: 1,
            kind: OutputKind::Text { max_tokens: 0 },
        }];
        assert_eq!(
            validate_semantic_shape(&context, &zero_bound),
            Err(SemanticValidationError::ZeroBound(0))
        );
        let duplicate = [
            OutputIntent {
                output_id: 3,
                cardinality: 1,
                kind: OutputKind::Text { max_tokens: 1 },
            },
            OutputIntent {
                output_id: 3,
                cardinality: 1,
                kind: OutputKind::Audio {
                    max_samples: 48_000,
                    sample_rate: 24_000,
                },
            },
        ];
        assert_eq!(
            validate_semantic_shape(&context, &duplicate),
            Err(SemanticValidationError::DuplicateOutput(3))
        );
    }

    #[test]
    fn unbounded_media_is_rejected() {
        let context = SemanticContext {
            primary: vec![ContextEntry {
                origin: ContextOrigin::Prompt,
                parts: vec![ContentPart::Media {
                    artifact: ArtifactRef {
                        artifact_id: 9,
                        digest: [0; 32],
                        bytes: 512,
                        descriptor: MediaDescriptor::Video(VideoDescriptor {
                            width: 640,
                            height: 480,
                            frame_rate: (30, 1),
                            max_frames: 0,
                        }),
                    },
                }],
            }],
            negative: vec![],
        };
        let outputs = [OutputIntent {
            output_id: 0,
            cardinality: 1,
            kind: OutputKind::Text { max_tokens: 8 },
        }];
        assert_eq!(
            validate_semantic_shape(&context, &outputs),
            Err(SemanticValidationError::EmptyArtifact(9))
        );
    }
}
