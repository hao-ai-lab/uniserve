//! TypeSafe System One decision readout (`POST /v1/systemone`) for DiffusionGemma.
//!
//! A request carries one state and named questions; the model answers each
//! question from the probabilities it assigns to answer tokens at masked
//! canvas slots, after one denoising pass over a prompt that states the
//! questions. Nothing is generated.
//!
//! The request path is:
//!
//! 1. [`SystemOneRequest::from_json`] validates the official System One
//!    0.2.0 request plus the `x_images` extension, failing with FastAPI-style
//!    `422` bodies ([`SystemOneError`]); [`SystemOneRequest::check_model`]
//!    refuses another model name with `404`.
//! 2. [`ReadoutEncoder::plan`] turns the request and its decoded image sizes
//!    into a [`ReadoutPlan`]: prompts with image placements, canvas rows, and
//!    per-slot candidate tokens, in prompt format [`PROMPT_FORMAT`].
//! 3. The engine prefills each prompt, denoises each canvas row once, and
//!    returns the candidates' full-vocabulary log-probabilities, which
//!    [`ReadoutPlan::assemble`] turns into the official response, with the
//!    `x_candidate_mass` extension on every answer.
//!
//! [`ServingRuntime::systemone`](crate::serving::ServingRuntime::systemone)
//! runs these steps for one request body.
//!
//! The readout matches the reference DJev encoder token for token for every
//! request it accepts; `prompt` documents the text rules for the official
//! inputs it does not.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod answer;
mod error;
mod plan;
mod prompt;
mod python;
mod request;
mod serve;
#[cfg(test)]
mod tests;
mod vocabulary;

pub use answer::{Answer, OrderedMap, SystemOneResponse, Usage};
pub use error::{LocItem, SystemOneError, ValidationIssue};
pub use plan::{
    CanvasMode, CanvasRow, ImageSize, MAX_REQUEST_CONTEXT, MAX_STATE_AND_QUESTION, ReadoutEncoder,
    ReadoutLayout, ReadoutOptions, ReadoutPlan, ReadoutPrompt, ReadoutSlot, StartupError,
};
pub use prompt::PROMPT_FORMAT;
pub use request::{
    ChoiceOption, Criteria, MAX_CHOICE_OPTIONS, MAX_IMAGES, MAX_SCORE_LEVELS, Question,
    SystemOneRequest,
};
