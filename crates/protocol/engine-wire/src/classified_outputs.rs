use std::collections::BTreeSet;

use enum_as_inner::EnumAsInner;

use super::utility::UtilityOutput;
use super::{EngineCoreOutput, EngineCoreOutputs};
use crate::stats::SchedulerStats;

/// Data-parallel control notifications multiplexed through `EngineCoreOutputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpControlMessage {
    WaveComplete(u32),
    StartWave(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestBatchOutputs {
    pub engine_index: u32,
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: UtilityOutput,
}

/// Semantic classification of a raw `EngineCoreOutputs` message.
///
/// Python currently uses one product-shaped wire struct for several distinct
/// output families. This enum exposes those families more explicitly without
/// changing the wire format.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum ClassifiedEngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Utility(UtilityCallOutput),
    DpControl {
        engine_index: u32,
        timestamp: f64,
        control: DpControlMessage,
    },
    /// Fallback for wire-shape combinations that do not map cleanly onto the
    /// current semantic families.
    Other(EngineCoreOutputs),
}

impl EngineCoreOutputs {
    /// Classify the raw wire message into a more semantic Rust enum.
    pub fn classify(self) -> ClassifiedEngineCoreOutputs {
        let mut output = self;
        let has_request_payload = !output.outputs.is_empty()
            || output.scheduler_stats.is_some()
            || output.finished_requests.is_some();

        match (
            has_request_payload,
            output.utility_output.take(),
            output.wave_complete.take(),
            output.start_wave.take(),
        ) {
            (true, None, None, None) => {
                ClassifiedEngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                    engine_index: output.engine_index,
                    outputs: output.outputs,
                    scheduler_stats: output.scheduler_stats,
                    timestamp: output.timestamp,
                    finished_requests: output.finished_requests,
                })
            }
            (false, Some(utility_output), None, None) => {
                ClassifiedEngineCoreOutputs::Utility(UtilityCallOutput {
                    engine_index: output.engine_index,
                    timestamp: output.timestamp,
                    output: utility_output,
                })
            }
            (false, None, Some(wave_complete), None) => ClassifiedEngineCoreOutputs::DpControl {
                engine_index: output.engine_index,
                timestamp: output.timestamp,
                control: DpControlMessage::WaveComplete(wave_complete),
            },
            (false, None, None, Some(start_wave)) => ClassifiedEngineCoreOutputs::DpControl {
                engine_index: output.engine_index,
                timestamp: output.timestamp,
                control: DpControlMessage::StartWave(start_wave),
            },
            (has_request_payload, utility_output, wave_complete, start_wave) => {
                output.utility_output = utility_output;
                output.wave_complete = wave_complete;
                output.start_wave = start_wave;
                let _ = has_request_payload;
                ClassifiedEngineCoreOutputs::Other(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineCoreOutput;
    use crate::utility::{UtilityOutput, UtilityResultEnvelope};

    /// A message carrying per-request outputs (and nothing else) classifies as a
    /// `RequestBatch`, preserving the engine index and the contained outputs.
    #[test]
    fn classifies_pure_request_batch() {
        let outputs = EngineCoreOutputs {
            engine_index: 5,
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            ..Default::default()
        };

        let batch = outputs.classify().into_request_batch().unwrap();
        assert_eq!(batch.engine_index, 5);
        assert_eq!(batch.outputs.len(), 1);
        assert_eq!(batch.outputs[0].request_id, "req-1");
    }

    /// `finished_requests` alone (no token outputs) still counts as a request
    /// payload and classifies as `RequestBatch`.
    #[test]
    fn classifies_finished_requests_only_as_request_batch() {
        let outputs = EngineCoreOutputs {
            engine_index: 1,
            finished_requests: Some(BTreeSet::from(["done".to_string()])),
            ..Default::default()
        };

        let batch = outputs.classify().into_request_batch().unwrap();
        assert_eq!(batch.engine_index, 1);
        assert_eq!(
            batch.finished_requests,
            Some(BTreeSet::from(["done".to_string()]))
        );
    }

    /// A message carrying only a `utility_output` classifies as `Utility` with
    /// the same call id surfaced.
    #[test]
    fn classifies_pure_utility_result() {
        let outputs = EngineCoreOutputs {
            engine_index: 2,
            utility_output: Some(UtilityOutput {
                call_id: 42_u64.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(rmpv::Value::Nil)),
            }),
            ..Default::default()
        };

        let utility = outputs.classify().into_utility().unwrap();
        assert_eq!(utility.engine_index, 2);
        assert_eq!(utility.output.call_id, 42_u64);
    }

    /// A message carrying only `start_wave` classifies as a DP-control
    /// `StartWave` message with the wave number preserved.
    #[test]
    fn classifies_start_wave_as_dp_control() {
        let outputs = EngineCoreOutputs {
            engine_index: 3,
            start_wave: Some(11),
            ..Default::default()
        };

        match outputs.classify() {
            ClassifiedEngineCoreOutputs::DpControl {
                engine_index,
                control,
                ..
            } => {
                assert_eq!(engine_index, 3);
                assert_eq!(control, DpControlMessage::StartWave(11));
            }
            other => panic!("expected DpControl/StartWave, got {other:?}"),
        }
    }

    /// A message carrying only `wave_complete` classifies as a DP-control
    /// `WaveComplete` message.
    #[test]
    fn classifies_wave_complete_as_dp_control() {
        let outputs = EngineCoreOutputs {
            engine_index: 4,
            wave_complete: Some(9),
            ..Default::default()
        };

        match outputs.classify() {
            ClassifiedEngineCoreOutputs::DpControl { control, .. } => {
                assert_eq!(control, DpControlMessage::WaveComplete(9));
            }
            other => panic!("expected DpControl/WaveComplete, got {other:?}"),
        }
    }

    /// A message mixing a request payload with a control signal does not fit any
    /// clean family and falls through to `Other`, preserving the raw message.
    #[test]
    fn classifies_mixed_payload_as_other() {
        let outputs = EngineCoreOutputs {
            engine_index: 6,
            outputs: vec![EngineCoreOutput {
                request_id: "req-mix".to_string(),
                ..Default::default()
            }],
            start_wave: Some(2),
            ..Default::default()
        };

        let raw = outputs.classify().into_other().unwrap();
        assert_eq!(raw.engine_index, 6);
        assert_eq!(raw.outputs.len(), 1);
        assert_eq!(raw.start_wave, Some(2));
    }
}
