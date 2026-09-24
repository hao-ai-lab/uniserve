//! Decoded events, finish reasons, and logprob conversion shared by output layers.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which `serving::assembly` consumes; `logprobs` converts engine token-ID
//! candidates into decoded strings; `finish` carries the terminal reason.
//! [`CollectedTextOutput`] folds a decoded event stream into one final value.

pub use decoded::{DecodedTextEvent, Finished, TextDecodeOptions, decoded_text_event_stream};
pub(crate) use decoded::{matches_stop_string, stop_string_holdback_bytes};
pub use finish::{FinishReason, StopReason};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
pub(crate) use logprobs::{decode_logprobs, decode_prompt_logprobs};

mod decoded;
mod finish;
mod logprobs;

use std::sync::Arc;

use futures::{Stream, StreamExt as _, pin_mut};

use crate::serving::text::{Error, Result};

/// Final decoded text plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedTextOutput {
    /// Complete decoded response text.
    pub text: String,
    /// Prompt token identifiers submitted to the engine.
    pub prompt_token_ids: Arc<[u32]>,
    /// Per-position prompt log probabilities, when requested.
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    /// Per-position generated-token log probabilities, when requested.
    pub logprobs: Option<DecodedLogprobs>,
    /// Generated token identifiers in stream order.
    pub token_ids: Vec<u32>,
    /// Total number of generated tokens.
    pub output_token_count: usize,
    /// Number of generated tokens consumed by internal protocol sections.
    pub internal_token_count: usize,
    /// Terminal condition for the request.
    pub finish_reason: FinishReason,
}

impl CollectedTextOutput {
    /// Collects the stream to completion and returns the final decoded text plus
    /// terminal metadata.
    ///
    /// Text, token IDs, and logprob positions of every `TextDelta` are
    /// concatenated in stream order. Prompt metadata comes from the `Start`
    /// event that precedes the first delta; without one, `prompt_token_ids`
    /// is empty and `prompt_logprobs` is `None`. The function returns at the
    /// first delta that carries `finished` and does not poll the stream
    /// further.
    ///
    /// # Errors
    ///
    /// Returns the first error the stream yields, and
    /// [`Error::StreamClosedBeforeTerminalOutput`] when the stream ends
    /// without a terminal delta. `DecodedTextEvent` carries no request ID, so
    /// errors raised here report `"unknown"`.
    pub async fn collect(
        stream: impl Stream<Item = Result<DecodedTextEvent>> + Send,
    ) -> Result<Self> {
        pin_mut!(stream);
        let mut prompt_logprobs = None;
        let mut prompt_token_ids: Arc<[u32]> = Arc::from([]);
        let mut collected: Option<CollectedTextOutput> = None;

        while let Some(event) = stream.next().await.transpose()? {
            match event {
                DecodedTextEvent::Start {
                    prompt_logprobs: start_prompt_logprobs,
                    prompt_token_ids: start_prompt_token_ids,
                    ..
                } => {
                    prompt_logprobs = start_prompt_logprobs;
                    prompt_token_ids = start_prompt_token_ids;
                }
                DecodedTextEvent::TextDelta {
                    delta,
                    token_ids: delta_token_ids,
                    logprobs: mut delta_logprobs,
                    finished,
                    ..
                } => {
                    if let Some(c) = collected.as_mut() {
                        c.text.push_str(&delta);
                        c.token_ids.extend(delta_token_ids);
                        if let Some(dlp) = delta_logprobs.as_mut() {
                            if let Some(lp) = c.logprobs.as_mut() {
                                lp.positions.extend_from_slice(&dlp.positions);
                            } else {
                                c.logprobs = delta_logprobs;
                            }
                        }
                    } else {
                        // The `Error` finish reason is a placeholder that the
                        // terminal delta overwrites; a stream that closes
                        // before that delta returns an error instead.
                        collected = Some(CollectedTextOutput {
                            text: delta,
                            prompt_token_ids: Arc::clone(&prompt_token_ids),
                            prompt_logprobs: prompt_logprobs.take(),
                            logprobs: delta_logprobs,
                            token_ids: delta_token_ids,
                            output_token_count: 0,
                            internal_token_count: 0,
                            finish_reason: FinishReason::new(uniserve_core::FinishReason::Error),
                        })
                    };

                    if let Some(finished) = finished {
                        let Some(mut collected) = collected else {
                            return Err(Error::MalformedOutput {
                                request_id: "unknown".to_string(),
                                message: "terminal text event arrived before any collected text"
                                    .to_string(),
                            });
                        };
                        collected.finish_reason = finished.finish_reason;
                        collected.output_token_count = finished.output_token_count;
                        collected.internal_token_count = finished.internal_token_count;
                        return Ok(collected);
                    }
                }
            }
        }

        Err(Error::StreamClosedBeforeTerminalOutput {
            request_id: "unknown".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::serving::text::FinishReason;
    use futures::stream;

    use super::*;

    #[tokio::test]
    async fn collect_output_retains_prompt_and_sample_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                queued_at: None,
                scheduled_at: None,
                prompt_token_ids: vec![10, 11].into(),
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 0,
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "bc".to_string(),
                token_ids: vec![1, 2],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "a".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "bc".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            }],
                        },
                    ],
                }),
                finished: Some(Finished {
                    prompt_token_count: 2,
                    output_token_count: 2,
                    internal_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                }),
            }),
        ]);

        let collected = CollectedTextOutput::collect(stream).await.unwrap();

        assert_eq!(collected.text, "bc");
        assert_eq!(
            collected.prompt_logprobs,
            Some(DecodedPromptLogprobs {
                first_token_id: 0,
                first_token: "o".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: 0,
                        token: "p".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    }],
                }],
            })
        );
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "bc".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }

    #[tokio::test]
    async fn collect_output_accumulates_intermediate_deltas() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                queued_at: None,
                scheduled_at: None,
                prompt_token_ids: vec![10, 11].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "he".to_string(),
                token_ids: vec![1, 2],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "h".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "e".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            }],
                        },
                    ],
                }),
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "llo".to_string(),
                token_ids: vec![3, 4, 5],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "l".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "l".to_string(),
                                logprob: -0.4,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "o".to_string(),
                                logprob: -0.5,
                                rank: 1,
                            }],
                        },
                    ],
                }),
                finished: Some(Finished {
                    prompt_token_count: 2,
                    output_token_count: 5,
                    internal_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                }),
            }),
        ]);

        let collected = CollectedTextOutput::collect(stream).await.unwrap();

        assert_eq!(collected.text, "hello");
        assert_eq!(collected.prompt_logprobs, None);
        assert_eq!(collected.token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "h".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "e".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "l".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "l".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "o".to_string(),
                            logprob: -0.5,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }
}
