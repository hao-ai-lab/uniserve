//! Frontend ↔ engine wire protocol.
//!
//! These DTOs are inherited from the vllm-rs frontend (the shared ancestry) and
//! keep its msgpack encodings — tuple-encoded request/output structs, the
//! single-byte request-type frame, the HELLO/INIT/READY handshake messages —
//! so vllm-rs protocol tests port over as conformance tests. The deliberate
//! UniServe fork is the [`native`] module: native generation events
//! and request parameters the upstream protocol cannot express.
//!
//! Both sides of the boundary use this crate: `uniserve-engine-client`
//! serializes it over ZMQ in socket mode, and `uniserve-engine-process` hosts
//! the headless `uniserve engine` process.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::any::type_name;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;

use bytes::Bytes;
use rmpv::Value;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;

use crate::logprobs::MaybeWireLogprobs;
use crate::multimodal::MmFeatures;
use crate::native::{NativeOutputExt, NativeRequestExt};
use crate::stats::{PrefillStats, SchedulerStats};
use crate::utility::UtilityOutput;

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed.
pub type OpaqueValue = Value;

fn default_opaque_value_nil() -> OpaqueValue {
    Value::Nil
}

fn is_false(v: &bool) -> bool {
    !v
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

mod classified_outputs;
pub mod dtype;
pub mod error;
pub mod handshake;
pub mod logprobs;
pub mod lora;
pub mod multimodal;
pub mod native;
pub mod stats;
pub mod tensor;
pub mod translate;
pub mod utility;
pub use classified_outputs::{
    ClassifiedEngineCoreOutputs, DpControlMessage, RequestBatchOutputs, UtilityCallOutput,
};
pub use dtype::ModelDtype;
pub use error::{Error, Result};
pub use logprobs::decode_engine_outputs;

/// Dedicated single-frame sentinel emitted by a headless engine process when the
/// engine dies, so the frontend can fail all in-flight requests fast.
pub const ENGINE_CORE_DEAD_SENTINEL: &[u8] = b"ENGINE_CORE_DEAD";

/// Request types are encoded as single-byte protocol constants so they can be
/// sent over the ZMQ socket without an extra encoding step.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    Add = 0,
    Abort = 1,
    /// Reserved for DP wave coordination (vLLM compatibility). UniServe does
    /// not implement the coordinator until engines must stay in collective
    /// lockstep; the discriminant is reserved so the protocol never reuses it.
    StartDpWave = 2,
    Utility = 3,
}

impl EngineCoreRequestType {
    /// The single byte carried in the request-type frame: the `#[repr(u8)]`
    /// discriminant is the wire encoding, so the variant definitions are the
    /// sole source of truth for these values.
    pub fn as_byte(self) -> u8 {
        self as u8
    }

    /// Decode the single-byte request type frame used on the engine input
    /// socket. Returns `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };

        [Self::Add, Self::Abort, Self::StartDpWave, Self::Utility]
            .into_iter()
            .find(|variant| variant.as_byte() == *value)
    }

    /// Encode the request type as the single-byte frame used on the engine
    /// input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::copy_from_slice(&[self.as_byte()])
    }
}

/// One control-path message exchanged between the frontend and an engine over
/// the engine input socket, with the request kind and its payload fused into a
/// single value so dispatch is exhaustive and the wire tag is never a bare
/// literal.
///
/// On the wire each message is two frames — a single-byte [`EngineCoreRequestType`]
/// tag followed by the kind-specific msgpack payload — and this enum converts to
/// and from exactly those frames via [`encode_frames`](Self::encode_frames) and
/// [`decode_frames`](Self::decode_frames).
#[derive(Debug, Clone, PartialEq)]
pub enum EngineCoreControlRequest {
    /// Add a new generation request; payload is the [`EngineCoreRequest`]
    /// add-request tuple.
    Add(Box<EngineCoreRequest>),
    /// Abort the listed request IDs; payload is a msgpack array of strings.
    Abort(Vec<String>),
    /// Invoke an engine utility method; payload is the
    /// [`EngineCoreUtilityRequest`](crate::utility::EngineCoreUtilityRequest)
    /// tuple.
    Utility(Box<crate::utility::EngineCoreUtilityRequest>),
    /// DP wave-start signal; carries no payload (an empty msgpack frame).
    StartDpWave,
}

impl EngineCoreControlRequest {
    /// The request-type tag for this message's kind.
    pub fn request_type(&self) -> EngineCoreRequestType {
        match self {
            Self::Add(_) => EngineCoreRequestType::Add,
            Self::Abort(_) => EngineCoreRequestType::Abort,
            Self::Utility(_) => EngineCoreRequestType::Utility,
            Self::StartDpWave => EngineCoreRequestType::StartDpWave,
        }
    }

    /// Serialize to the two on-wire frames `(type_frame, payload_frame)`. The
    /// payload frame is the same msgpack the per-kind payload encoded to before
    /// this enum existed; `StartDpWave` carries an empty payload frame.
    pub fn encode_frames(&self) -> Result<(Bytes, Vec<u8>)> {
        let payload = match self {
            Self::Add(request) => encode_msgpack(request.as_ref())?,
            Self::Abort(request_ids) => encode_msgpack(request_ids)?,
            Self::Utility(request) => encode_msgpack(request.as_ref())?,
            Self::StartDpWave => Vec::new(),
        };
        Ok((self.request_type().to_frame(), payload))
    }

    /// Reconstruct a message from its on-wire `(type_frame, payload_frame)`.
    /// Returns `None` when the type frame is not a recognized tag; payload
    /// decode failures surface as `Some(Err(..))`.
    pub fn decode_frames(type_frame: &[u8], payload: &[u8]) -> Option<Result<Self>> {
        let request_type = EngineCoreRequestType::from_frame(type_frame)?;
        Some(match request_type {
            EngineCoreRequestType::Add => {
                decode_msgpack::<EngineCoreRequest>(payload).map(|r| Self::Add(Box::new(r)))
            }
            EngineCoreRequestType::Abort => decode_msgpack::<Vec<String>>(payload).map(Self::Abort),
            EngineCoreRequestType::Utility => {
                decode_msgpack::<crate::utility::EngineCoreUtilityRequest>(payload)
                    .map(|r| Self::Utility(Box::new(r)))
            }
            EngineCoreRequestType::StartDpWave => Ok(Self::StartDpWave),
        })
    }
}

/// Reason a request finished: stop, length, abort, error, or repetition.

/// This mirrors the Python enum and uses integer encoding for compact wire
/// representation.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreFinishReason {
    /// A stop string was emitted.
    Stop = 0,
    /// `max_tokens` or `max_model_len` was reached.
    Length = 1,
    /// The request was aborted by the client.
    Abort = 2,
    /// A retryable request-level internal error occurred.
    Error = 3,
    /// A repetitive token pattern was detected.
    Repetition = 4,
}

/// Event types emitted by the engine for one request.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine event associated with one request.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    pub r#type: EngineCoreEventType,
    pub timestamp: f64,
}

/// Controls how intermediate outputs are returned to the frontend.

/// `Cumulative = 0` is intentionally not supported in Rust frontend.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return only token deltas in each update.
    #[default]
    Delta = 1,
    /// Suppress intermediate updates and return only the final output.
    FinalOnly = 2,
}

/// The stop reason associated with a finished output.

/// Python models this as the union-typed `stop_reason: int | str | None`
/// field on `EngineCoreOutput`; the Rust client narrows it into a tagged enum.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Parameters for configuring structured outputs (guided decoding).

/// Exactly one constraint field (`json`, `regex`, `choice`, `grammar`,
/// `json_object`, or `structural_tag`) should be set. The engine backend
/// selects the appropriate grammar compiler based on which field
/// is present.

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StructuredOutputsParams {
    /// JSON schema (as a dict/object or JSON string) constraining the output.
    pub json: Option<serde_json::Value>,
    /// Regular expression the output must match.
    pub regex: Option<String>,
    /// List of allowed output strings (the model must produce one of these).
    pub choice: Option<Vec<String>>,
    /// Context-free grammar (in EBNF-like notation) the output must conform to.
    pub grammar: Option<String>,
    /// When `true`, output must be valid JSON (free-form, no schema).
    pub json_object: Option<bool>,
    /// Disable any additional whitespace in guided JSON output.
    #[serde(skip_serializing_if = "crate::is_false")]
    pub disable_any_whitespace: bool,
    /// Disable `additionalProperties` in JSON schema output.
    #[serde(skip_serializing_if = "crate::is_false")]
    pub disable_additional_properties: bool,
    /// Custom whitespace pattern for guided JSON output.
    pub whitespace_pattern: Option<String>,
    /// Structural tag configuration (JSON-encoded string).
    pub structural_tag: Option<String>,
}

/// Engine-facing sampling parameters for text generation.

/// This is the normalized southbound subset used by the frontend when it talks
/// to the engine over the wire. User-facing request semantics such as
/// `stop` strings, `n`, and output aggregation mode are intentionally handled
/// by higher layers before values reach this DTO.

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    #[serde(default)]
    pub top_k: u32,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate per output sequence.
    pub max_tokens: u32,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    #[serde(default)]
    pub min_tokens: u32,
    /// Whether model EOS tokens should be ignored by scheduler termination.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Number of log probabilities to return per generated token.

    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.

    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// Minimum probability threshold for token sampling.
    #[serde(default)]
    pub min_p: f32,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f32,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f32,
    /// Repetition penalty applied by the sampler.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Token IDs that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Primary EOS token ID used by the engine's dedicated EOS stop path.

    /// This mirrors Python's internal `_eos_token_id` field and is derived by
    /// the frontend from tokenizer/model metadata rather than supplied directly
    /// by end users.
    #[serde(rename = "_eos_token_id")]
    pub eos_token_id: Option<u32>,
    /// Complete stop-token set used by the engine for `min_tokens` masking.

    /// This mirrors Python's internal `_all_stop_token_ids` field and should
    /// contain explicit `stop_token_ids` plus any frontend-derived EOS token
    /// IDs.
    #[serde(rename = "_all_stop_token_ids")]
    pub all_stop_token_ids: BTreeSet<u32>,
    /// Logit biases to apply during sampling.
    /// Keys are token IDs
    #[serde(default)]
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Tokenized bad words to avoid during generation.
    #[serde(default, rename = "_bad_words_token_ids")]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// UniServe extension (frontend-derived, like `_bad_words_token_ids`):
    /// guided-choice alternatives tokenized by the frontend, since the engine
    /// process has no tokenizer. Compiled engine-side into a token-trie
    /// grammar that masks each step.
    #[serde(default, rename = "_choice_token_ids")]
    pub choice_token_ids: Option<Vec<Vec<u32>>>,
    /// Parameters for configuring structured outputs (guided decoding).
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputsParams>,
    /// Specific token IDs for which log probabilities should be returned at
    /// each position.

    /// When set, the engine returns logprobs for exactly these tokens in
    /// addition to the sampled/scored token. Mutually exclusive with the
    /// `logprobs` count field in practice.
    #[serde(default)]
    pub logprob_token_ids: Option<Vec<u32>>,
    /// If `Some(true)`, the request will not attempt to read from the prefix
    /// cache; newly computed blocks may still populate the cache. `None`
    /// defers to engine defaults.
    #[serde(default)]
    pub skip_reading_prefix_cache: Option<bool>,
    /// Additional request parameters for custom extensions (from `uniserve_xargs`).
    #[serde(default)]
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
}

impl EngineCoreSamplingParams {
    /// Constructs a default sampling params for testing purposes only.
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 65536,
            min_tokens: 0,
            ignore_eos: false,
            logprobs: None,
            prompt_logprobs: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            all_stop_token_ids: BTreeSet::new(),
            logit_bias: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            choice_token_ids: None,
            structured_outputs: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            extra_args: None,
        }
    }
}

/// Engine-core add-request payload sent from frontend to engine.

#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Multimodal features attached to the request.
    pub mm_features: Option<MmFeatures>,
    pub sampling_params: Option<EngineCoreSamplingParams>,
    /// Pooling parameters are preserved in the schema but not yet strongly
    /// typed.
    pub pooling_params: Option<OpaqueValue>,
    pub arrival_time: f64,
    #[serde(default)]
    pub lora_request: Option<lora::LoraRequest>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    /// Unsupported in the Rust client because Python uses a custom tensor/aux-
    /// frame encoding path for this field.
    #[serde(default)]
    pub prompt_embeds: Option<OpaqueValue>,
    /// Per-position mask for mixed-mode inputs (e.g. chat completion with
    /// `prompt_embeds` content parts). `Some(true)` means real token id;
    /// `Some(false)` means the position uses a pre-computed entry from
    /// `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    #[serde(default)]
    pub prompt_is_token_ids: Option<Vec<bool>>,
    /// Index of the client, used to ensure outputs are sent back to the same
    /// client when scaling out the frontend.
    #[serde(default)]
    pub client_index: u32,
    /// In DP mode, indicates which wave this request is expected to belong to.
    #[serde(default)]
    pub current_wave: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub trace_headers: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub resumable: bool,
    /// Original user-provided request ID, used for output reporting and aborts.
    #[serde(default)]
    pub external_req_id: Option<String>,
    #[serde(default)]
    pub reasoning_ended: Option<bool>,
    /// Opaque reasoning-parser kwargs forwarded from the frontend to the
    /// structured-output backend.
    #[serde(default)]
    pub reasoning_parser_kwargs: Option<OpaqueValue>,
    /// If `true`, the request should be added to the scheduler's waiting queue
    /// and immediately aborted, so connector-side cleanup runs via the
    /// standard `request_finished` hook.
    #[serde(default)]
    pub abort_immediately: bool,
    /// UniServe protocol extension (appended; absent on the upstream wire):
    /// native generation parameters. `None` for plain text
    /// requests — the request then matches the upstream encoding except for the
    /// extra trailing nil.
    #[serde(default)]
    pub native: Option<NativeRequestExt>,
}

impl EngineCoreRequest {
    /// Validate fields intentionally not supported in the Rust client.
    pub fn validate(&self) -> Result<()> {
        if self.prompt_embeds.is_some() {
            return Err(Error::UnsupportedField {
                context: "EngineCoreRequest",
                field: "prompt_embeds",
            });
        }
        Ok(())
    }
}

/// Engine-core output for a single request.

#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions in this
    /// output.
    #[serde(default)]
    pub new_logprobs: Option<MaybeWireLogprobs>,
    /// Decoded prompt logprobs for the scored prompt positions emitted in this
    /// output.
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<MaybeWireLogprobs>,
    #[serde(default)]
    pub pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    pub finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    pub kv_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub trace_headers: Option<OpaqueValue>,
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    #[serde(default)]
    pub prefill_stats: Option<PrefillStats>,
    #[serde(default)]
    pub routed_experts: Option<OpaqueValue>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    #[serde(default)]
    pub num_nans_in_logits: u32,
    /// UniServe protocol extension (appended; absent on the upstream wire):
    /// typed image events and native finish statistics for native generation
    /// requests. `None` on the plain text path.
    #[serde(default)]
    pub native: Option<NativeOutputExt>,
}

impl EngineCoreOutput {
    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Batch of engine outputs returned to a frontend client.

#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutputs {
    #[serde(default)]
    pub engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default)]
    pub outputs: Vec<EngineCoreOutput>,
    #[serde(default)]
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    #[serde(default)]
    pub timestamp: f64,
    #[serde(default)]
    pub utility_output: Option<UtilityOutput>,
    #[serde(default)]
    pub finished_requests: Option<BTreeSet<String>>,
    /// In DP mode, signals that the current wave finished and engines are
    /// paused.
    #[serde(default)]
    pub wave_complete: Option<u32>,
    /// In DP mode, signals that a request arrived for a prior wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    pub start_wave: Option<u32>,
}

/// Encode a Rust value into MessagePack using the protocol crate's serde model.
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

/// Decode a MessagePack payload into a strongly typed protocol value, with
/// enhanced error reporting.
pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    fn decode_value_preview(bytes: &[u8]) -> String {
        match decode_value(bytes) {
            Ok(value) => format!("{value}"),
            Err(error) => format!("<value decode failed: {error}>"),
        }
    }

    messagepack_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn engine_request_serializes_as_full_array() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            arrival_time: 1234.5,
            client_index: 7,
            ..EngineCoreRequest::default()
        };

        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };

        // 20 upstream positions + the trailing UniServe `native` extension.
        assert_eq!(array.len(), 21);
        assert_eq!(array[0], Value::from("req-1"));
        assert_eq!(array[2], Value::Nil);
        assert_eq!(array[4], Value::Nil);
        assert_eq!(array[10], Value::Nil);
        assert_eq!(array[11], Value::from(7));
        assert_eq!(array[20], Value::Nil);
    }

    /// A 20-element upstream-shaped request (no `native` slot) still decodes:
    /// the UniServe extension defaults to `None`.
    #[test]
    fn upstream_shaped_request_decodes_with_default_native() {
        let request = EngineCoreRequest {
            request_id: "req-up".to_string(),
            ..EngineCoreRequest::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let Value::Array(mut array) = decode_value(&encoded).unwrap() else {
            panic!("expected array");
        };
        array.pop(); // drop the trailing `native` slot -> upstream shape
        assert_eq!(array.len(), 20);
        let mut upstream = Vec::new();
        rmpv::encode::write_value(&mut upstream, &Value::Array(array)).unwrap();

        let decoded: EngineCoreRequest = decode_msgpack(&upstream).unwrap();
        assert_eq!(decoded.request_id, "req-up");
        assert!(decoded.native.is_none());
    }

    #[test]
    fn engine_outputs_roundtrip_finished_fields() {
        let outputs = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![42],
                finish_reason: Some(EngineCoreFinishReason::Length),
                stop_reason: Some(StopReason::Text("stop".to_string())),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        let encoded = encode_msgpack(&outputs).unwrap();
        let decoded: EngineCoreOutputs = decode_msgpack(&encoded).unwrap();

        assert_eq!(decoded.outputs.len(), 1);
        assert_eq!(
            decoded.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }

    /// The request-type byte derives from the `#[repr(u8)]` discriminant, so
    /// the type frame matches the pre-enum constants exactly.
    #[test]
    fn request_type_frame_bytes_match_protocol_constants() {
        assert_eq!(EngineCoreRequestType::Add.to_frame().as_ref(), b"\x00");
        assert_eq!(EngineCoreRequestType::Abort.to_frame().as_ref(), b"\x01");
        assert_eq!(
            EngineCoreRequestType::StartDpWave.to_frame().as_ref(),
            b"\x02"
        );
        assert_eq!(EngineCoreRequestType::Utility.to_frame().as_ref(), b"\x03");

        assert_eq!(
            EngineCoreRequestType::from_frame(b"\x00"),
            Some(EngineCoreRequestType::Add)
        );
        assert_eq!(
            EngineCoreRequestType::from_frame(b"\x01"),
            Some(EngineCoreRequestType::Abort)
        );
        assert_eq!(
            EngineCoreRequestType::from_frame(b"\x02"),
            Some(EngineCoreRequestType::StartDpWave)
        );
        assert_eq!(
            EngineCoreRequestType::from_frame(b"\x03"),
            Some(EngineCoreRequestType::Utility)
        );
        assert_eq!(EngineCoreRequestType::from_frame(b"\x04"), None);
        assert_eq!(EngineCoreRequestType::from_frame(b"\x00\x00"), None);
    }

    /// `Add` frames are byte-identical to the prior `(type_frame, msgpack(EngineCoreRequest))`
    /// pair, and round-trip back to the same value.
    #[test]
    fn control_request_add_frames_match_reference_bytes() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            arrival_time: 1234.5,
            client_index: 7,
            ..EngineCoreRequest::default()
        };

        let reference_type = EngineCoreRequestType::Add.to_frame();
        let reference_payload = encode_msgpack(&request).unwrap();

        let control = EngineCoreControlRequest::Add(Box::new(request.clone()));
        let (type_frame, payload) = control.encode_frames().unwrap();
        assert_eq!(type_frame, reference_type);
        assert_eq!(payload, reference_payload);

        let decoded = EngineCoreControlRequest::decode_frames(&type_frame, &payload)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, EngineCoreControlRequest::Add(Box::new(request)));
    }

    /// `Abort` frames are byte-identical to `(type_frame, msgpack(Vec<String>))`.
    #[test]
    fn control_request_abort_frames_match_reference_bytes() {
        let request_ids = vec!["a".to_string(), "b".to_string()];

        let reference_type = EngineCoreRequestType::Abort.to_frame();
        let reference_payload = encode_msgpack(&request_ids).unwrap();

        let control = EngineCoreControlRequest::Abort(request_ids.clone());
        let (type_frame, payload) = control.encode_frames().unwrap();
        assert_eq!(type_frame, reference_type);
        assert_eq!(payload, reference_payload);

        let decoded = EngineCoreControlRequest::decode_frames(&type_frame, &payload)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, EngineCoreControlRequest::Abort(request_ids));
    }

    /// `Utility` frames are byte-identical to `(type_frame, msgpack(EngineCoreUtilityRequest))`.
    #[test]
    fn control_request_utility_frames_match_reference_bytes() {
        let request =
            crate::utility::EngineCoreUtilityRequest::new(7, 42, "is_sleeping", ()).unwrap();

        let reference_type = EngineCoreRequestType::Utility.to_frame();
        let reference_payload = encode_msgpack(&request).unwrap();

        let control = EngineCoreControlRequest::Utility(Box::new(request.clone()));
        let (type_frame, payload) = control.encode_frames().unwrap();
        assert_eq!(type_frame, reference_type);
        assert_eq!(payload, reference_payload);

        let decoded = EngineCoreControlRequest::decode_frames(&type_frame, &payload)
            .unwrap()
            .unwrap();
        assert_eq!(
            decoded,
            EngineCoreControlRequest::Utility(Box::new(request))
        );
    }

    /// `StartDpWave` carries the `\x02` tag and an empty payload frame, and the
    /// payload is ignored on decode (matching the pre-enum dispatch).
    #[test]
    fn control_request_start_dp_wave_frames_have_empty_payload() {
        let (type_frame, payload) = EngineCoreControlRequest::StartDpWave
            .encode_frames()
            .unwrap();
        assert_eq!(type_frame, EngineCoreRequestType::StartDpWave.to_frame());
        assert!(payload.is_empty());

        let decoded = EngineCoreControlRequest::decode_frames(&type_frame, &payload)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, EngineCoreControlRequest::StartDpWave);
        // A non-empty payload frame is still accepted and ignored.
        let decoded = EngineCoreControlRequest::decode_frames(&type_frame, b"\xc0")
            .unwrap()
            .unwrap();
        assert_eq!(decoded, EngineCoreControlRequest::StartDpWave);
    }

    /// An unrecognized type frame yields `None`, matching `from_frame`.
    #[test]
    fn control_request_unknown_type_frame_is_none() {
        assert!(EngineCoreControlRequest::decode_frames(b"\x09", b"").is_none());
    }

    /// Decoding a payload whose msgpack marker does not match the target type
    /// produces a structured `Error::Decode` naming the target type, with the
    /// original value preserved as a fallback in the message.
    #[test]
    fn decode_msgpack_wrong_marker_reports_target_type_and_value_fallback() {
        // A bare msgpack string where an `EngineCoreOutputs` array/struct is
        // expected: the marker is wrong for the target type.
        let payload = encode_msgpack(&"not an outputs struct".to_string()).unwrap();

        let error = decode_msgpack::<EngineCoreOutputs>(&payload).unwrap_err();

        match error {
            Error::Decode {
                target_type,
                message,
            } => {
                // The target type is named (the &'static type_name path), not a
                // generic "decode failed".
                assert!(
                    target_type.contains("EngineCoreOutputs"),
                    "target_type should name EngineCoreOutputs, got {target_type:?}"
                );
                // The original value is preserved as a decodable fallback rather
                // than discarded.
                assert!(
                    message.contains("value fallback"),
                    "message should carry the value fallback, got {message:?}"
                );
                assert!(
                    message.contains("not an outputs struct"),
                    "message should preserve the original value, got {message:?}"
                );
            }
            other => panic!("expected Error::Decode, got {other:?}"),
        }
    }
}
