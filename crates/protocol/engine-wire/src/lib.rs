//! Canonical frontend-to-engine transport for generation submission, exact-prefix control, typed generation events, startup capabilities, and scheduler metrics.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::any::type_name;
use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport as _;
use uniserve_core::GenerationRequest;
pub use uniserve_engine_api::GenEvent;

use crate::stats::SchedulerStats;

pub mod error;
pub mod generation;
pub mod handshake;
pub mod stats;

pub use error::{Error, Result};
pub use uniserve_core::ModelDtype;

pub type OpaqueValue = rmpv::Value;

/// Dedicated single-frame sentinel emitted when a headless engine can no longer serve requests.
pub const ENGINE_CORE_DEAD_SENTINEL: &[u8] = b"ENGINE_CORE_DEAD";

/// The complete configured frontend-to-engine request set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineRequestKind {
    Submit = 0,
    Abort = 1,
    Cancel = 2,
    CancelAt = 3,
    AcknowledgeAt = 4,
    StopAt = 5,
}

impl EngineRequestKind {
    pub fn as_byte(self) -> u8 {
        self as u8
    }

    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };
        [
            Self::Submit,
            Self::Abort,
            Self::Cancel,
            Self::CancelAt,
            Self::AcknowledgeAt,
            Self::StopAt,
        ]
        .into_iter()
        .find(|kind| kind.as_byte() == *value)
    }

    pub fn to_frame(self) -> Bytes {
        Bytes::copy_from_slice(&[self.as_byte()])
    }
}

/// Physical transport metadata around one canonical generation request.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct GenerationRequestEnvelope {
    pub external_request_id: String,
    pub arrival_time: f64,
    pub client_index: u32,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
    pub request: GenerationRequest,
}

impl GenerationRequestEnvelope {
    pub fn validate(&self) -> Result<()> {
        if self.external_request_id.is_empty() {
            return Err(Error::InvalidGenerationRequest {
                message: "external request id must not be empty".to_string(),
            });
        }
        if !self.arrival_time.is_finite() || self.arrival_time < 0.0 {
            return Err(Error::InvalidGenerationRequest {
                message: "arrival time must be finite and nonnegative".to_string(),
            });
        }
        self.request
            .validate()
            .map_err(|error| Error::InvalidGenerationRequest {
                message: error.to_string(),
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct CancelAt {
    pub external_request_id: String,
    pub output_token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct AcknowledgeAt {
    pub external_request_id: String,
    pub output_token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct StopAt {
    pub external_request_id: String,
    pub output_token_count: u64,
}

/// Exhaustive request/control message sent to one headless engine.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineRequest {
    Submit(Box<GenerationRequestEnvelope>),
    Abort(Vec<String>),
    Cancel(Vec<String>),
    CancelAt(Vec<CancelAt>),
    AcknowledgeAt(Vec<AcknowledgeAt>),
    StopAt(Vec<StopAt>),
}

impl EngineRequest {
    pub fn kind(&self) -> EngineRequestKind {
        match self {
            Self::Submit(_) => EngineRequestKind::Submit,
            Self::Abort(_) => EngineRequestKind::Abort,
            Self::Cancel(_) => EngineRequestKind::Cancel,
            Self::CancelAt(_) => EngineRequestKind::CancelAt,
            Self::AcknowledgeAt(_) => EngineRequestKind::AcknowledgeAt,
            Self::StopAt(_) => EngineRequestKind::StopAt,
        }
    }

    pub fn encode_frames(&self) -> Result<(Bytes, Vec<u8>)> {
        let payload = match self {
            Self::Submit(request) => encode_msgpack(request.as_ref())?,
            Self::Abort(request_ids) | Self::Cancel(request_ids) => encode_msgpack(request_ids)?,
            Self::CancelAt(requests) => encode_msgpack(requests)?,
            Self::AcknowledgeAt(requests) => encode_msgpack(requests)?,
            Self::StopAt(requests) => encode_msgpack(requests)?,
        };
        Ok((self.kind().to_frame(), payload))
    }

    pub fn decode_frames(kind_frame: &[u8], payload: &[u8]) -> Option<Result<Self>> {
        let kind = EngineRequestKind::from_frame(kind_frame)?;
        Some(match kind {
            EngineRequestKind::Submit => decode_msgpack::<GenerationRequestEnvelope>(payload)
                .map(|request| Self::Submit(Box::new(request))),
            EngineRequestKind::Abort => decode_msgpack::<Vec<String>>(payload).map(Self::Abort),
            EngineRequestKind::Cancel => decode_msgpack::<Vec<String>>(payload).map(Self::Cancel),
            EngineRequestKind::CancelAt => {
                decode_msgpack::<Vec<CancelAt>>(payload).map(Self::CancelAt)
            }
            EngineRequestKind::AcknowledgeAt => {
                decode_msgpack::<Vec<AcknowledgeAt>>(payload).map(Self::AcknowledgeAt)
            }
            EngineRequestKind::StopAt => decode_msgpack::<Vec<StopAt>>(payload).map(Self::StopAt),
        })
    }
}

/// One canonical runtime event correlated to its public request identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedGenerationEvent {
    pub external_request_id: String,
    pub event: GenEvent,
}

impl RoutedGenerationEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.event,
            GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
        )
    }
}

/// Bounded batch of canonical events plus routing and load metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationEventBatch {
    pub engine_index: u32,
    pub events: Vec<RoutedGenerationEvent>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub emitted_at: f64,
}

pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + std::fmt::Debug,
{
    messagepack_serde::to_vec_named(value).map_err(|error| Error::Encode {
        target_type: type_name::<T>(),
        message: format!(
            "failed to encode value `{:?}`: {}",
            value,
            error.to_report_string()
        ),
    })
}

pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    messagepack_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{
        ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationResourceBounds, ImageParams, RequestId,
        SamplingParams, UndVisibility,
    };
    use uniserve_engine_api::FinishReason;

    fn generation_request() -> GenerationRequest {
        let constraint = GenerationConstraint::UndOnly;
        let policy = GenerationPolicyDescriptor::default();
        GenerationRequest {
            request_id: RequestId(1),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![1, 2, 3],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 8,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 3,
                max_kv_tokens: 11,
                ..GenerationResourceBounds::default()
            },
        }
    }

    #[test]
    fn canonical_submission_and_events_round_trip() {
        let request = GenerationRequestEnvelope {
            external_request_id: "request-1".to_string(),
            arrival_time: 12.5,
            client_index: 2,
            data_parallel_rank: None,
            trace_headers: None,
            request: generation_request(),
        };
        request.validate().unwrap();
        let encoded = EngineRequest::Submit(Box::new(request.clone()))
            .encode_frames()
            .unwrap();
        assert_eq!(
            EngineRequest::decode_frames(&encoded.0, &encoded.1)
                .unwrap()
                .unwrap(),
            EngineRequest::Submit(Box::new(request))
        );

        let batch = GenerationEventBatch {
            engine_index: 0,
            events: vec![RoutedGenerationEvent {
                external_request_id: "request-1".to_string(),
                event: GenEvent::Finished {
                    reason: FinishReason::MaxTokens,
                    stop_reason: None,
                    prompt_tokens: 3,
                    completion_tokens: 8,
                    images: 0,
                    kv_transfer_params: None,
                },
            }],
            scheduler_stats: None,
            emitted_at: 13.0,
        };
        let decoded: GenerationEventBatch =
            decode_msgpack(&encode_msgpack(&batch).unwrap()).unwrap();
        assert_eq!(decoded, batch);
        assert!(decoded.events[0].is_terminal());
    }
}
