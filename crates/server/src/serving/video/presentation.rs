//! The conditioner presentation: the token ids the Qwen3-VL conditioner
//! reads and the AdaLN tag of every one of them.
//!
//! No chat template is applied. `fl2va` labels each keyframe `"<Picture i>: "`
//! before its vision block. `ref2va` labels its references in request order,
//! numbered per modality: a soundtrack `"<Audio j>: "` (a video's soundtrack
//! before its `"<Video k>: "` label), an image `"<Picture i>: "` before its
//! vision block, and a video one `"<{t:.1f} seconds>"` label before each of
//! its vision blocks. Audio contributes labels only and `ref2va` keyframes
//! contribute nothing. The prompt follows verbatim. Every text segment is
//! tokenized on its own without special tokens and the pieces are
//! concatenated.
//!
//! A vision block is `<|vision_start|>`, one `<|image_pad|>` or
//! `<|video_pad|>` per vision token, and `<|vision_end|>`. Its tokens,
//! markers included, carry the video tag ([`VIDEO_TAG`]); every other token
//! carries the text tag ([`TEXT_TAG`]).

use std::borrow::Cow;

use thiserror_ext::AsReport as _;

use super::plan::{Prepared, RequestPlan, Vision};
use super::{RequestField, VideoInputError};
use crate::profile::tokenizer::HuggingFaceTokenizer;

/// The AdaLN tag of a vision token.
pub const VIDEO_TAG: u8 = 0;
/// The AdaLN tag of a text token.
pub const TEXT_TAG: u8 = 1;

const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";

/// The placeholder a vision block's tokens are written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionPad {
    /// `<|image_pad|>`.
    Image,
    /// `<|video_pad|>`.
    Video,
}

impl VisionPad {
    /// The placeholder token.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Image => "<|image_pad|>",
            Self::Video => "<|video_pad|>",
        }
    }
}

/// One piece of the presentation, before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment<'a> {
    /// Text, tokenized on its own.
    Text(Cow<'a, str>),
    /// A vision block of `tokens` placeholders between the markers.
    Vision {
        /// The placeholder.
        pad: VisionPad,
        /// Vision tokens of the block.
        tokens: u32,
    },
}

/// Lists the presentation of a planned request, prompt last.
pub fn segments<'a>(plan: &RequestPlan, prompt: &'a str) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    let (mut pictures, mut videos, mut audios) = (0, 0, 0);
    let text = |value: String| Segment::Text(Cow::Owned(value));
    for condition in &plan.conditions {
        match (&condition.prepared, &condition.vision) {
            (Prepared::Audio(_), _) => {
                audios += 1;
                segments.push(text(format!("<Audio {audios}>: ")));
            }
            (Prepared::Video(clip), Some(Vision::Video(seen))) => {
                if clip.soundtrack.is_some() {
                    audios += 1;
                    segments.push(text(format!("<Audio {audios}>: ")));
                }
                videos += 1;
                segments.push(text(format!("<Video {videos}>: ")));
                for timestamp in &seen.block_timestamps {
                    // Like Python's `{:.1f}`, this rounds the exact value
                    // half to even: a 0.25-second mean renders as "0.2".
                    segments.push(text(format!("<{timestamp:.1} seconds>")));
                    segments.push(Segment::Vision {
                        pad: VisionPad::Video,
                        tokens: seen.block_tokens,
                    });
                }
            }
            (_, Some(Vision::Image(seen))) => {
                pictures += 1;
                segments.push(text(format!("<Picture {pictures}>: ")));
                segments.push(Segment::Vision {
                    pad: VisionPad::Image,
                    tokens: seen.tokens,
                });
            }
            // Keyframes of a ref2va request are not presented.
            _ => {}
        }
    }
    segments.push(Segment::Text(Cow::Borrowed(prompt)));
    segments
}

/// The tokenized presentation and the AdaLN tag of every token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Presentation {
    /// Token ids, labels and vision blocks first, prompt last.
    pub token_ids: Vec<u32>,
    /// [`VIDEO_TAG`] or [`TEXT_TAG`] per token.
    pub tags: Vec<u8>,
}

/// Tokenizes the presentation of a planned request.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] for `prompt` when the prompt is blank
/// or contains a vision placeholder or marker, which would misalign the
/// conditioner's vision inputs, and [`VideoInputError::Internal`] when the
/// tokenizer fails or lacks the vision tokens a vision block needs.
pub fn present(
    tokenizer: &HuggingFaceTokenizer,
    plan: &RequestPlan,
    prompt: &str,
) -> Result<Presentation, VideoInputError> {
    if prompt.trim().is_empty() {
        return Err(VideoInputError::invalid(
            RequestField::Prompt,
            "must not be blank",
        ));
    }
    // A vision block needs the vision vocabulary; a prompt can only produce
    // the vision tokens the tokenizer has, which it must not contain.
    let token = |name: &'static str| (name, tokenizer.token_to_id(name));
    let required = |(name, id): (&str, Option<u32>)| {
        id.ok_or_else(|| VideoInputError::internal(format!("the tokenizer has no {name} token")))
    };
    let start = token(VISION_START);
    let end = token(VISION_END);
    let image_pad = token(VisionPad::Image.token());
    let video_pad = token(VisionPad::Video.token());
    let placeholders: Vec<u32> = [start, end, image_pad, video_pad]
        .iter()
        .filter_map(|(_, id)| *id)
        .collect();

    let segments = segments(plan, prompt);
    let last = segments.len() - 1;
    let mut token_ids = Vec::new();
    let mut tags = Vec::new();
    for (position, segment) in segments.iter().enumerate() {
        match segment {
            Segment::Vision { pad, tokens } => {
                let pad = required(match pad {
                    VisionPad::Image => image_pad,
                    VisionPad::Video => video_pad,
                })?;
                let block = *tokens as usize + 2;
                token_ids.push(required(start)?);
                token_ids.extend(std::iter::repeat_n(pad, *tokens as usize));
                token_ids.push(required(end)?);
                tags.extend(std::iter::repeat_n(VIDEO_TAG, block));
            }
            Segment::Text(text) => {
                let ids = tokenizer.encode(text, false).map_err(|error| {
                    VideoInputError::internal(format!(
                        "tokenizing the presentation failed: {}",
                        error.as_report()
                    ))
                })?;
                if position == last && ids.iter().any(|id| placeholders.contains(id)) {
                    return Err(VideoInputError::invalid(
                        RequestField::Prompt,
                        "must not contain vision placeholder tokens",
                    ));
                }
                tags.extend(std::iter::repeat_n(TEXT_TAG, ids.len()));
                token_ids.extend(ids);
            }
        }
    }
    Ok(Presentation { token_ids, tags })
}

#[cfg(test)]
pub(super) mod tests {
    use std::borrow::Cow;

    use serde_json::Value;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::super::RequestField;
    use super::super::plan::tests::{Request, fixture, vision};
    use super::{Segment, TEXT_TAG, VIDEO_TAG, VisionPad, present, segments};
    use crate::profile::assets::PipelineCheckpoint;
    use crate::profile::tokenizer::HuggingFaceTokenizer;
    use crate::serving::model::pipeline_tokenizer;

    fn expected_segments(case: &Value) -> Vec<Segment<'static>> {
        case["expected"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|segment| match segment.get("vision") {
                Some(pad) => Segment::Vision {
                    pad: if pad == "image" {
                        VisionPad::Image
                    } else {
                        VisionPad::Video
                    },
                    tokens: segment["tokens"].as_u64().unwrap() as u32,
                },
                None => Segment::Text(Cow::Owned(segment["text"].as_str().unwrap().to_owned())),
            })
            .collect()
    }

    /// The expected token ids and tags of a vectors request, rebuilt from its
    /// segments and the recorded token ids of the checkpoint tokenizer.
    fn expected_tokens(fixture: &Value, case: &Value) -> (Vec<u32>, Vec<u8>) {
        let token = |name: &str| fixture["tokens"][name].as_u64().unwrap() as u32;
        let mut token_ids = Vec::new();
        let mut tags = Vec::new();
        for segment in case["expected"]["segments"].as_array().unwrap() {
            if let Some(pad) = segment.get("vision") {
                let pad = token(&format!("<|{}_pad|>", pad.as_str().unwrap()));
                let count = segment["tokens"].as_u64().unwrap() as usize;
                token_ids.push(token("<|vision_start|>"));
                token_ids.extend(std::iter::repeat_n(pad, count));
                token_ids.push(token("<|vision_end|>"));
                tags.extend(std::iter::repeat_n(VIDEO_TAG, count + 2));
            } else {
                let ids = segment["token_ids"].as_array().unwrap();
                token_ids.extend(ids.iter().map(|id| id.as_u64().unwrap() as u32));
                tags.extend(std::iter::repeat_n(TEXT_TAG, ids.len()));
            }
        }
        (token_ids, tags)
    }

    /// Requests present the reference's labels, timestamps and vision blocks
    /// in the reference's order.
    #[test]
    fn segments_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let plan = Request::parse(case).plan(&vision).unwrap();
            let prompt = case["prompt"].as_str().unwrap();
            assert_eq!(
                segments(&plan, prompt),
                expected_segments(case),
                "{}",
                case["name"]
            );
        }
    }

    /// A deterministic tokenizer: one token per character, plus the vision
    /// markers and placeholders as special tokens.
    pub(in crate::serving::video) fn character_tokenizer() -> HuggingFaceTokenizer {
        let mut vocab: Vocab = (1_u32..=127)
            .map(|id| (char::from_u32(id).unwrap().to_string(), id))
            .collect();
        vocab.insert("<unk>".to_owned(), 0);
        let specials = [
            ("<|vision_start|>", 128),
            ("<|vision_end|>", 129),
            ("<|image_pad|>", 130),
            ("<|video_pad|>", 131),
        ];
        for (token, id) in specials {
            vocab.insert(token.to_owned(), id);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(
            &specials
                .iter()
                .map(|(token, _)| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        HuggingFaceTokenizer::new(&path).unwrap()
    }

    /// Vision blocks are tagged video, markers included; labels and the
    /// prompt are tagged text and tokenized segment by segment.
    #[test]
    fn vision_blocks_carry_the_video_tag() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let tokenizer = character_tokenizer();
        let case = fixture["requests"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == "ref2va_video_audio")
            .unwrap();
        let plan = Request::parse(case).plan(&vision).unwrap();
        let presentation = present(&tokenizer, &plan, "go").unwrap();

        let mut expected_ids = Vec::new();
        let mut expected_tags = Vec::new();
        for segment in segments(&plan, "go") {
            match segment {
                Segment::Text(text) => {
                    expected_ids.extend(text.chars().map(u32::from));
                    expected_tags.extend(std::iter::repeat_n(TEXT_TAG, text.chars().count()));
                }
                Segment::Vision { pad, tokens } => {
                    let pad = if pad == VisionPad::Image { 130 } else { 131 };
                    expected_ids.push(128);
                    expected_ids.extend(std::iter::repeat_n(pad, tokens as usize));
                    expected_ids.push(129);
                    expected_tags.extend(std::iter::repeat_n(VIDEO_TAG, tokens as usize + 2));
                }
            }
        }
        assert_eq!(presentation.token_ids, expected_ids);
        assert_eq!(presentation.tags, expected_tags);
    }

    /// A prompt that is empty or carries vision tokens is rejected.
    #[test]
    fn prompts_must_be_plain_text() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let tokenizer = character_tokenizer();
        let plan = Request::parse(&fixture["requests"][0])
            .plan(&vision)
            .unwrap();
        for prompt in ["", "a <|image_pad|> b", "<|vision_start|>"] {
            let error = present(&tokenizer, &plan, prompt).unwrap_err();
            assert_eq!(error.field(), Some(RequestField::Prompt), "{prompt:?}");
        }
    }

    /// With the checkpoint's own tokenizer, loaded as the server loads it
    /// (`UNISERVE_MINIMAX_H3_MODEL` names the checkpoint root), the token ids
    /// equal the reference's.
    #[tokio::test]
    async fn checkpoint_tokenizer_matches_the_reference() {
        let Ok(root) = std::env::var("UNISERVE_MINIMAX_H3_MODEL") else {
            eprintln!("UNISERVE_MINIMAX_H3_MODEL is not set; skipping the tokenizer parity test");
            return;
        };
        let pipeline = PipelineCheckpoint::resolve(&root, None)
            .await
            .unwrap()
            .unwrap();
        let tokenizer = pipeline_tokenizer(&pipeline).await.unwrap();
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let plan = Request::parse(case).plan(&vision).unwrap();
            let presentation =
                present(&tokenizer, &plan, case["prompt"].as_str().unwrap()).unwrap();
            let (token_ids, tags) = expected_tokens(&fixture, case);
            assert_eq!(presentation.token_ids, token_ids, "{}", case["name"]);
            assert_eq!(presentation.tags, tags, "{}", case["name"]);
        }
    }
}
