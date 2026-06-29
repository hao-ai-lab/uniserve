//! Validated JSON extractor for automatic request validation.
//! Variation of https://github.com/lightseekorg/smg/blob/main/crates/protocols/src/validated.rs

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;
use uniserve_openai_types::Normalizable;
use validator::Validate;

use crate::error::{ApiError, invalid_request};

/// A JSON extractor that automatically normalizes and validates the request
/// body.

/// After deserialization the extractor first calls
/// [`Normalizable::normalize`] and then `Validate::validate`. The amount of
/// checking performed by the latter is entirely determined by `T`'s `Validate`
/// impl: request types such as `ChatCompletionRequest`/`CompletionRequest`
/// declare real `#[validate(...)]` constraints (ranges, custom and
/// cross-parameter checks), whereas types that derive `Validate` without any
/// field constraints (e.g. the LoRA admin and token-in/token-out generate
/// requests) get a derived no-op `validate` and rely solely on
/// deserialization and downstream lowering for their invariants. When
/// `validate` does report errors, this returns [`ApiError::InvalidRequest`]
/// with the validation details.
pub(crate) struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(mut data) = Json::<T>::from_request(req, state)
            .await
            .map_err(|err: JsonRejection| ApiError::json_parse_error(err.body_text()))?;

        data.normalize();

        data.validate()
            .map_err(|validation_errors| invalid_request!("{}", validation_errors))?;

        Ok(ValidatedJson(data))
    }
}

impl<T> std::ops::Deref for ValidatedJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ValidatedJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
