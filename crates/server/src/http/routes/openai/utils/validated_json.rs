//! Validated JSON extractor for automatic request validation.

use crate::openai::Normalizable;
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;
use validator::Validate;

use crate::openai::ApiError;

/// A JSON extractor that automatically normalizes and validates the request
/// body.
///
/// After deserialization the extractor first calls
/// [`Normalizable::normalize`] and then `Validate::validate`. The amount of
/// checking performed by the latter is entirely determined by `T`'s `Validate`
/// impl. Request types declare `#[validate(...)]` constraints for ranges and
/// cross-parameter checks, while fallible request lowering enforces the
/// configured serving features. When validation reports errors,
/// this returns [`ApiError::InvalidRequest`] with the details.
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

        data.validate().map_err(|validation_errors| {
            ApiError::invalid_request(validation_errors.to_string(), None)
        })?;

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
