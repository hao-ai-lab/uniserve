//! Readout plans: the prompts and canvases that answer one System One request.
//!
//! [`ReadoutEncoder::plan`] groups a request's questions into canvases,
//! renders each group's prompt with the checkpoint chat template, and lays
//! out each canvas's answer slots with the candidate tokens whose
//! log-probabilities answer them. The engine prefills every prompt, denoises
//! every canvas row once over its prompt, and returns the requested
//! log-probabilities, which [`ReadoutPlan::assemble`] turns into answers.
//!
//! # Canvas scaffold
//!
//! Question `i` (1-based within its canvas) occupies the tokens of `"{i}:"`,
//! one `<mask>` slot (two for a two-letter label), and the tokens of `"\n"`.
//! `<turn|>` follows the last question and `<pad>` fills the canvas to its
//! length: the checkpoint's `canvas_length`, or with [`CanvasMode::Compact`]
//! the smallest multiple of 16 holding the scaffold and `<turn|>`.
//!
//! # Layouts
//!
//! [`ReadoutLayout::Joint`] packs complete questions, in request order, into
//! one canvas until the next question's scaffold and `<turn|>` would exceed
//! `canvas_length`; the next canvas numbers its questions from 1 again. Each
//! canvas has its own prompt, which carries the shared preamble, images, and
//! state followed by only that canvas's questions. [`ReadoutLayout::Independent`]
//! gives every question its own prompt and canvas.
//!
//! # Two-letter labels
//!
//! A choice with more than 52 options is read left to right. The first
//! canvas row, where both of its slots are masked, reads the first letter `X`
//! over the `" X"` tokens its labels use. For every such first letter, one
//! more row repeats the first row with only this question's first slot fixed
//! to `" X"`, and reads the second slot over the `" Y"` tokens its labels use
//! after `X`; all other slots of that row stay masked, so each row conditions
//! on exactly one first letter. Option `X Y` then has probability
//! `P(X) * P(Y | X)` from raw full-vocabulary probabilities.

use std::ops::Range;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::profile::assets::ResolvedModelFiles;
use crate::profile::diffusion_gemma::{
    ControlTokens, DiffusionGemmaProfile, ImagePlacement, ImageTokenMismatch, PatchBudget,
};
use crate::profile::tokenizer::{DynTokenizer, TokenizerError};
use crate::serving::chat::template::renderer::hf::MultimodalRenderInfo;
use crate::serving::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatTemplateLoadOptions,
    ChatToolChoice, HfChatRenderer,
};
use crate::serving::systemone::error::{LocItem, SystemOneError, ValidationIssue};
use crate::serving::systemone::prompt::{
    QuestionText, SECOND_LETTERS, prompt_text, shared_lines, uses_letter_pairs, value_text,
};
use crate::serving::systemone::request::{Criteria, SystemOneRequest};
use crate::serving::systemone::vocabulary::Vocabulary;
use crate::serving::text::TextDecodeOptions;

/// Largest request context the endpoint admits, in tokens (the official 64k
/// limit); a smaller `max_model_len` lowers it.
pub const MAX_REQUEST_CONTEXT: u32 = 65_536;
/// Largest state plus longest question, in tokens (the official 32k limit);
/// half a smaller request context lowers it.
pub const MAX_STATE_AND_QUESTION: u32 = 32_768;
/// Compact canvases are sized in multiples of this many tokens.
const COMPACT_CANVAS_MULTIPLE: usize = 16;

/// How a request's questions are divided among prompts and canvases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadoutLayout {
    /// Questions share prompts and canvases up to the canvas capacity.
    #[default]
    Joint,
    /// Every question has its own prompt and canvas, so adding or removing a
    /// question never changes another question's answer.
    Independent,
}

/// How long each readout canvas is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanvasMode {
    /// The checkpoint's full `canvas_length`.
    #[default]
    Full,
    /// The smallest multiple of 16 tokens that holds the scaffold and `<turn|>`.
    Compact,
    /// A fixed multiple of 16, no longer than the checkpoint's canvas.
    Fixed(u32),
}

/// Spellings whose probability contributes to each answer candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateTokens {
    /// Sum every supported single-token spelling of a candidate.
    #[default]
    Variants,
    /// Read only the primary, space-prefixed spelling of each candidate.
    Primary,
}

impl FromStr for CandidateTokens {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "variants" => Ok(Self::Variants),
            "primary" => Ok(Self::Primary),
            other => Err(format!(
                "unknown readout candidates `{other}`; expected variants or primary"
            )),
        }
    }
}

impl FromStr for ReadoutLayout {
    type Err = String;

    /// Parses `joint` or `independent`.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "joint" => Ok(Self::Joint),
            "independent" => Ok(Self::Independent),
            other => Err(format!(
                "unknown readout layout `{other}`; expected joint or independent"
            )),
        }
    }
}

impl FromStr for CanvasMode {
    type Err = String;

    /// Parses `full`, `compact`, or a positive token count divisible by 16.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "full" => Ok(Self::Full),
            "compact" => Ok(Self::Compact),
            other => match other.parse::<u32>() {
                Ok(length) if length > 0 && length.is_multiple_of(16) => Ok(Self::Fixed(length)),
                _ => Err(format!(
                    "unknown readout canvas `{other}`; expected full, compact, or a positive multiple of 16"
                )),
            },
        }
    }
}

/// Server-wide readout settings, fixed at startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutOptions {
    /// Division of questions among prompts and canvases.
    pub layout: ReadoutLayout,
    /// Canvas length.
    pub canvas: CanvasMode,
    /// Candidate spellings contributing to the returned probabilities.
    #[serde(default)]
    pub candidates: CandidateTokens,
}

/// Pixel dimensions of one decoded request image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Failure to prepare the readout encoder at startup.
#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    /// The checkpoint chat template cannot be loaded or rendered.
    #[error("readout chat template: {0}")]
    Template(#[from] crate::serving::chat::Error),
    /// The tokenizer cannot encode a readout scaffold string.
    #[error("readout tokenization: {0}")]
    Tokenizer(#[from] TokenizerError),
    /// The tokenizer or template violates the readout token contract.
    #[error("readout token contract: {0}")]
    Contract(String),
}

/// Encodes System One requests into readout plans for one DiffusionGemma
/// checkpoint.
pub struct ReadoutEncoder {
    tokenizer: DynTokenizer,
    // The checkpoint's own template with no server-level template overrides
    // or default kwargs: the readout prompt is fixed by the prompt format.
    renderer: HfChatRenderer,
    vocabulary: Vocabulary,
    tokens: ControlTokens,
    images: PatchBudget,
    canvas_length: usize,
    options: ReadoutOptions,
    // Tokens of "{i}:" at index i - 1, for every question number a canvas holds.
    numbers: Vec<Vec<u32>>,
    max_model_len: usize,
    context_limit: usize,
    question_limit: usize,
}

impl ReadoutEncoder {
    /// Prepares the encoder for a checkpoint.
    ///
    /// Loads the checkpoint chat template from `files`, verifies the answer
    /// vocabulary (see `Vocabulary::resolve`), and verifies that the template's
    /// image placeholder tokenizes to the checkpoint's image token.
    /// `max_model_len` is the served context length; the request context is
    /// the smaller of it and [`MAX_REQUEST_CONTEXT`].
    ///
    /// # Errors
    ///
    /// Returns [`StartupError`] when the template cannot be loaded or the
    /// tokenizer violates the readout token contract.
    pub fn load(
        files: &ResolvedModelFiles,
        tokenizer: DynTokenizer,
        profile: &DiffusionGemmaProfile,
        options: ReadoutOptions,
        max_model_len: u32,
    ) -> Result<Self, StartupError> {
        let placeholder = tokenizer.id_to_token(profile.tokens.image).ok_or_else(|| {
            StartupError::Contract(format!(
                "image token {} is not in the vocabulary",
                profile.tokens.image
            ))
        })?;
        let renderer = HfChatRenderer::load(
            files,
            ChatTemplateLoadOptions::default(),
            Some(MultimodalRenderInfo {
                placeholder_token: placeholder,
            }),
        )?;
        let mut vocabulary = Vocabulary::resolve(&tokenizer).map_err(StartupError::Contract)?;
        if options.candidates == CandidateTokens::Primary {
            // The first entry is the verified space-prefixed spelling, so
            // planning and probability assembly share exactly the same set.
            for variants in &mut vocabulary.label_variants {
                variants.truncate(1);
            }
            vocabulary.yes_variants.truncate(1);
            vocabulary.no_variants.truncate(1);
        }
        let canvas_length = match options.canvas {
            CanvasMode::Fixed(length) => {
                if length == 0 || !length.is_multiple_of(16) || length > profile.canvas_length {
                    return Err(StartupError::Contract(format!(
                        "readout canvas {length} must be a positive multiple of 16 no greater than {}",
                        profile.canvas_length
                    )));
                }
                length as usize
            }
            CanvasMode::Full | CanvasMode::Compact => profile.canvas_length as usize,
        };
        let numbers = (1..=canvas_length)
            .map(|number| tokenizer.encode(&format!("{number}:"), false))
            .collect::<Result<Vec<_>, _>>()?;
        let context_limit = MAX_REQUEST_CONTEXT.min(max_model_len) as usize;

        let encoder = Self {
            tokenizer,
            renderer,
            vocabulary,
            tokens: profile.tokens,
            images: profile.images,
            canvas_length,
            options,
            numbers,
            max_model_len: max_model_len as usize,
            context_limit,
            question_limit: (MAX_STATE_AND_QUESTION as usize).min(context_limit / 2),
        };

        // One attached image must render as exactly one image token.
        let probe = encoder.render(
            "probe".to_owned(),
            &["data:image/png;base64,AA==".to_owned()],
        )?;
        let placeholders = probe
            .iter()
            .filter(|token| **token == profile.tokens.image)
            .count();
        if placeholders != 1 {
            return Err(StartupError::Contract(format!(
                "one attached image renders as {placeholders} image tokens"
            )));
        }
        Ok(encoder)
    }

    /// Returns the startup readout settings.
    pub fn options(&self) -> ReadoutOptions {
        self.options
    }

    /// Plans the readout of a validated request.
    ///
    /// `images` gives the decoded dimensions of `request.images`, in order;
    /// each image contributes the Gemma-4 soft tokens of its size, between
    /// `<|image>` and `<image|>`.
    ///
    /// # Errors
    ///
    /// Returns [`SystemOneError::Validation`] when a question cannot fit an
    /// empty canvas, an image has no area, the request text contains image
    /// placeholder tokens, or the encoded request exceeds its limits: the
    /// request context (the smaller of [`MAX_REQUEST_CONTEXT`] and
    /// `max_model_len`), the state plus the longest question (the smaller of
    /// [`MAX_STATE_AND_QUESTION`] and half the request context), or a prompt
    /// and its canvas exceeding `max_model_len`. Returns
    /// [`SystemOneError::Server`] when `images` does not match
    /// `request.images` or rendering fails.
    pub fn plan(
        &self,
        request: &SystemOneRequest,
        images: &[ImageSize],
    ) -> Result<ReadoutPlan, SystemOneError> {
        if images.len() != request.images.len() {
            return Err(SystemOneError::Server(format!(
                "readout planning received {} image sizes for {} images",
                images.len(),
                request.images.len()
            )));
        }
        let soft_tokens = images
            .iter()
            .enumerate()
            .map(|(index, size)| {
                self.images
                    .soft_tokens(size.width, size.height)
                    .map_err(|message| {
                        invalid(
                            vec!["body".into(), "x_images".into(), index.into()],
                            &message,
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let state = value_text(&request.state);
        let shared = shared_lines(&state, images.len());
        let groups = self.group(request)?;
        let mut question_texts: Vec<Vec<QuestionText>> = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .enumerate()
                    .map(|(position, &question)| {
                        QuestionText::new(position + 1, &request.questions[question])
                    })
                    .collect()
            })
            .collect();
        self.check_question_size(request, &state, &soft_tokens, &groups, &question_texts)?;

        let mut readouts: Vec<Option<AnswerReadout>> = vec![None; request.questions.len()];
        let mut candidates = 0;
        let mut prompts = Vec::with_capacity(groups.len());
        for (group, texts) in groups.iter().zip(question_texts.drain(..)) {
            let (token_ids, placements) =
                self.encode_prompt(prompt_text(&shared, &texts), &request.images, &soft_tokens)?;
            let rows = self.canvas_rows(request, group, &mut candidates, &mut readouts);
            prompts.push(ReadoutPrompt {
                token_ids,
                images: placements,
                rows,
            });
        }

        // Every prompt begins with the same preamble, images, and state, and
        // possibly more identical tokens; the request encodes their longest
        // common prefix once and each prompt's remainder separately.
        let shared_prefix = prompts
            .iter()
            .skip(1)
            .fold(prompts[0].token_ids.len(), |shared, prompt| {
                shared.min(common_prefix(&prompts[0].token_ids, &prompt.token_ids))
            });
        let input_tokens = shared_prefix
            + prompts
                .iter()
                .map(|prompt| prompt.token_ids.len() - shared_prefix)
                .sum::<usize>();
        if input_tokens > self.context_limit {
            return Err(invalid(
                vec!["body".into()],
                &format!(
                    "the request encodes {input_tokens} prompt tokens; the limit is {}",
                    self.context_limit
                ),
            ));
        }
        for prompt in &prompts {
            let canvas = prompt.rows[0].token_ids.len();
            if prompt.token_ids.len() + canvas > self.max_model_len {
                return Err(invalid(
                    vec!["body".into()],
                    &format!(
                        "a readout prompt of {} tokens and its {canvas}-token canvas exceed the \
                         {}-token model context",
                        prompt.token_ids.len(),
                        self.max_model_len
                    ),
                ));
            }
        }

        Ok(ReadoutPlan {
            prompts,
            shared_prefix_tokens: to_u32(shared_prefix),
            input_tokens: to_u32(input_tokens),
            readouts: readouts.into_iter().flatten().collect(),
            candidate_count: candidates,
        })
    }

    /// Divides the questions into canvas groups of question indices.
    fn group(&self, request: &SystemOneRequest) -> Result<Vec<Vec<usize>>, SystemOneError> {
        // A question's scaffold tokens at 1-based `number`: the number and
        // colon, its slots, and the line break.
        let cost = |number: usize, slots: usize| {
            self.numbers[number - 1].len() + slots + self.vocabulary.newline.len()
        };
        // Every canvas ends with `<turn|>`.
        let empty = 1;

        let mut groups = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        let mut used = empty;
        for (index, question) in request.questions.iter().enumerate() {
            let slots = slot_count(&question.criteria);
            let joins = match self.options.layout {
                ReadoutLayout::Joint => used + cost(current.len() + 1, slots) <= self.canvas_length,
                ReadoutLayout::Independent => false,
            };
            if !current.is_empty() && !joins {
                groups.push(std::mem::take(&mut current));
                used = empty;
            }
            let needed = cost(current.len() + 1, slots);
            if used + needed > self.canvas_length {
                return Err(invalid(
                    vec![
                        "body".into(),
                        "questions".into(),
                        question.id.as_str().into(),
                    ],
                    &format!(
                        "the question needs {} canvas tokens with the end of turn; a canvas \
                         holds {}",
                        empty + needed,
                        self.canvas_length
                    ),
                ));
            }
            used += needed;
            current.push(index);
        }
        groups.push(current);
        Ok(groups)
    }

    /// Checks the state plus the longest question against its limit.
    ///
    /// The state counts its text tokens and every image's soft tokens and
    /// markers; a question counts the tokens of its block and answer-format
    /// line as rendered in its prompt.
    fn check_question_size(
        &self,
        request: &SystemOneRequest,
        state: &str,
        soft_tokens: &[u32],
        groups: &[Vec<usize>],
        texts: &[Vec<QuestionText>],
    ) -> Result<(), SystemOneError> {
        let state_tokens = self.encode(state)?.len()
            + soft_tokens
                .iter()
                .map(|tokens| *tokens as usize + 2)
                .sum::<usize>();
        let mut longest: Option<(usize, usize)> = None;
        for (group, texts) in groups.iter().zip(texts) {
            for (&question, text) in group.iter().zip(texts) {
                let tokens = self
                    .encode(&format!("{}\n{}", text.block.join("\n"), text.hint))?
                    .len();
                if longest.is_none_or(|(_, most)| tokens > most) {
                    longest = Some((question, tokens));
                }
            }
        }
        let Some((question, question_tokens)) = longest else {
            return Ok(());
        };
        if state_tokens + question_tokens > self.question_limit {
            let id = request.questions[question].id.as_str();
            return Err(invalid(
                vec!["body".into(), "questions".into(), id.into()],
                &format!(
                    "the state and question `{id}` encode to {} tokens; the state plus the \
                     longest question may have {}",
                    state_tokens + question_tokens,
                    self.question_limit
                ),
            ));
        }
        Ok(())
    }

    /// Renders and tokenizes one prompt, expanding each image placeholder into
    /// its soft-token run.
    fn encode_prompt(
        &self,
        text: String,
        sources: &[String],
        soft_tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<ImagePlacement>), SystemOneError> {
        let rendered = self
            .render(text, sources)
            .map_err(|error| SystemOneError::Server(error.to_string()))?;

        // The template writes one image token per image, before the text;
        // each expands into the image's soft-token run.
        let (mut token_ids, placements) = self
            .tokens
            .expand_images(&rendered, soft_tokens)
            .map_err(|mismatch| match mismatch {
                ImageTokenMismatch::Extra(extra) => invalid(
                    vec!["body".into()],
                    &format!(
                        "the request text encodes {extra} image placeholder token(s); images \
                         are attached only through x_images"
                    ),
                ),
                ImageTokenMismatch::Missing { .. } => {
                    SystemOneError::Server(format!("the readout prompt: {mismatch}"))
                }
            })?;
        token_ids.extend(&self.vocabulary.thought_prefix);
        Ok((token_ids, placements))
    }

    /// Renders one user turn with the checkpoint template and an open model
    /// turn, and tokenizes it without adding special tokens.
    fn render(&self, text: String, sources: &[String]) -> Result<Vec<u32>, StartupError> {
        // Images precede the text in the user turn, as the reference places them.
        let content = if sources.is_empty() {
            ChatContent::Text(text)
        } else {
            ChatContent::Parts(
                sources
                    .iter()
                    .map(|source| ChatContentPart::image_url(source.as_str()))
                    .chain([ChatContentPart::text(text)])
                    .collect(),
            )
        };
        let request = ChatRequest {
            messages: vec![ChatMessage::User { content }],
            chat_options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            decode_options: TextDecodeOptions::default(),
        };
        let rendered = self.renderer.render(&request)?;
        Ok(self.tokenizer.encode(&rendered, false)?)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, SystemOneError> {
        self.tokenizer
            .encode(text, false)
            .map_err(|error| SystemOneError::Server(error.to_string()))
    }

    /// Lays out one group's canvas rows and records each question's readout.
    ///
    /// `candidates` counts the candidate log-probabilities planned so far, in
    /// plan order; each question's readout refers to its candidates by that
    /// flat index.
    fn canvas_rows(
        &self,
        request: &SystemOneRequest,
        group: &[usize],
        candidates: &mut usize,
        readouts: &mut [Option<AnswerReadout>],
    ) -> Vec<CanvasRow> {
        let mask = self.tokens.mask;
        let mut token_ids = Vec::with_capacity(self.canvas_length);
        let mut slots = Vec::with_capacity(group.len());
        // Two-letter questions: (question, option names, first-slot position,
        // flat index of the first-letter candidates).
        let mut chained = Vec::new();

        for (position, &question) in group.iter().enumerate() {
            token_ids.extend(&self.numbers[position]);
            let slot = to_u32(token_ids.len());
            let (kind, variants) = match &request.questions[question].criteria {
                Criteria::Noul { .. } => (
                    AnswerKind::Noul,
                    vec![
                        self.vocabulary.no_variants.clone(),
                        self.vocabulary.yes_variants.clone(),
                    ],
                ),
                // A two-letter question reads only its first letter in this
                // row; the rows conditioned on each first letter follow.
                Criteria::Choice(options) if uses_letter_pairs(options.len()) => {
                    let first_letters = options.len().div_ceil(SECOND_LETTERS);
                    token_ids.extend([mask, mask]);
                    let names: Vec<String> =
                        options.iter().map(|option| option.name.clone()).collect();
                    chained.push((question, names, slot, *candidates));
                    slots.push(ReadoutSlot {
                        position: slot,
                        candidates: self.vocabulary.labels[..first_letters].to_vec(),
                    });
                    *candidates += first_letters;
                    token_ids.extend(&self.vocabulary.newline);
                    continue;
                }
                Criteria::Choice(options) => (
                    AnswerKind::Choice(options.iter().map(|option| option.name.clone()).collect()),
                    self.vocabulary.label_variants[..options.len()].to_vec(),
                ),
                Criteria::Score(levels) => (
                    AnswerKind::Score(levels.clone()),
                    self.vocabulary.label_variants[..levels.len()].to_vec(),
                ),
            };
            token_ids.push(mask);
            token_ids.extend(&self.vocabulary.newline);

            // Each key sums the probabilities of its spellings.
            let mut keys = Vec::with_capacity(variants.len());
            for spellings in &variants {
                keys.push(*candidates..*candidates + spellings.len());
                *candidates += spellings.len();
            }
            slots.push(ReadoutSlot {
                position: slot,
                candidates: variants.concat(),
            });
            readouts[question] = Some(AnswerReadout {
                id: request.questions[question].id.clone(),
                kind,
                distribution: Distribution::Summed(keys),
            });
        }
        token_ids.push(self.tokens.turn_end);
        let length = match self.options.canvas {
            CanvasMode::Full | CanvasMode::Fixed(_) => self.canvas_length,
            CanvasMode::Compact => token_ids
                .len()
                .next_multiple_of(COMPACT_CANVAS_MULTIPLE)
                .min(self.canvas_length),
        };
        token_ids.resize(length, self.tokens.pad);

        let mut rows = vec![CanvasRow { token_ids, slots }];
        for (question, names, slot, first_candidates) in chained {
            let options = names.len();
            let first_letters = options.div_ceil(SECOND_LETTERS);
            let mut pairs = Vec::with_capacity(options);
            for first in 0..first_letters {
                let second_letters = (options - first * SECOND_LETTERS).min(SECOND_LETTERS);
                let mut conditioned = rows[0].token_ids.clone();
                conditioned[slot as usize] = self.vocabulary.labels[first];
                rows.push(CanvasRow {
                    token_ids: conditioned,
                    slots: vec![ReadoutSlot {
                        position: slot + 1,
                        candidates: self.vocabulary.labels[..second_letters].to_vec(),
                    }],
                });
                pairs.extend(
                    (0..second_letters)
                        .map(|second| (first_candidates + first, *candidates + second)),
                );
                *candidates += second_letters;
            }
            readouts[question] = Some(AnswerReadout {
                id: request.questions[question].id.clone(),
                kind: AnswerKind::Choice(names),
                distribution: Distribution::Chained(pairs),
            });
        }
        rows
    }
}

/// Answer slots per question: two for a two-letter label, otherwise one.
fn slot_count(criteria: &Criteria) -> usize {
    match criteria {
        Criteria::Choice(options) if uses_letter_pairs(options.len()) => 2,
        _ => 1,
    }
}

fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// A planning refusal. Plan-time limits apply to the encoded request rather
/// than to one input value, so the issue carries no `input`.
fn invalid(loc: Vec<LocItem>, message: &str) -> SystemOneError {
    SystemOneError::Validation(vec![ValidationIssue::value_error(loc, message, None)])
}

/// Converts a token count or position bounded by the request context.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The prompts and canvases that answer one request.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutPlan {
    /// Prompts in plan order. Every prompt begins with the same
    /// `shared_prefix_tokens` tokens.
    pub prompts: Vec<ReadoutPrompt>,
    /// Leading tokens identical in every prompt, which the engine can prefill
    /// once and share.
    pub shared_prefix_tokens: u32,
    /// Unique prompt tokens: the shared prefix once plus every prompt's
    /// remainder, image soft tokens included.
    pub input_tokens: u32,
    pub(super) readouts: Vec<AnswerReadout>,
    pub(super) candidate_count: usize,
}

impl ReadoutPlan {
    /// Returns the number of candidate log-probabilities
    /// [`ReadoutPlan::assemble`] takes: one per candidate of every slot, in
    /// plan order (prompt, then row, then slot, then candidate).
    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

/// One prompt and the canvases denoised over it.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutPrompt {
    /// Prompt tokens, including image soft-token placeholders and the closing
    /// empty thinking channel.
    pub token_ids: Vec<u32>,
    /// Images in `x_images` order with their placeholder runs.
    pub images: Vec<ImagePlacement>,
    /// Canvas rows, each denoised once over the prompt. The first row has
    /// every answer slot masked; any further rows condition a two-letter
    /// label's second slot on its first letter.
    pub rows: Vec<CanvasRow>,
}

/// One canvas denoised in a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanvasRow {
    /// Canvas tokens; decoder positions continue after the prompt.
    pub token_ids: Vec<u32>,
    /// Slots whose candidate log-probabilities the row reads.
    pub slots: Vec<ReadoutSlot>,
}

/// One canvas position read over candidate tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutSlot {
    /// Position in the canvas.
    pub position: u32,
    /// Tokens whose log-probabilities, under the log-softmax over the full
    /// vocabulary of the softcapped logits at `position`, the readout needs.
    pub candidates: Vec<u32>,
}

/// How a question's answer is read from the flat candidate log-probabilities.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AnswerReadout {
    pub(super) id: String,
    pub(super) kind: AnswerKind,
    pub(super) distribution: Distribution,
}

/// The answer type and its keys, in option or level order.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum AnswerKind {
    /// Keys `false` then `true`.
    Noul,
    /// Keys are the option names.
    Choice(Vec<String>),
    /// Keys are the level indices; each level is echoed in the legend.
    Score(Vec<Value>),
}

/// Unnormalized per-key probability of a question's answer.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Distribution {
    /// Per key, the candidates whose probabilities sum into it.
    Summed(Vec<Range<usize>>),
    /// Per option, the candidates of its first letter and of its second letter
    /// given the first; the key's probability is their product.
    Chained(Vec<(usize, usize)>),
}
