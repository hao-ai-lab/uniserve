//! OpenAI-compatible HTTP route handlers.

pub(crate) mod chat_completions;
mod images;
mod models;
pub(crate) mod utils;
mod videos;

pub(crate) use chat_completions::chat_completions;
pub(crate) use images::images_generations;
pub(crate) use models::list_models;
pub(crate) use videos::videos_sync;
