//! Token decoding for generated and prompt logprobs.
//!
//! Converts the engine's token-ID candidates (`uniserve_core::PositionLogprobs`)
//! into decoded candidate strings. Ranks and log probabilities are passed
//! through unchanged from the engine.

use crate::profile::tokenizer::HuggingFaceTokenizer;
use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use uniserve_core::PositionLogprobs;

use crate::serving::text::error::Error;

/// One decoded token candidate and its logprob metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedTokenLogprob {
    /// Original vocabulary token ID for this candidate.
    pub token_id: u32,
    /// Best-effort decoded token string for this candidate.
    ///
    /// The token is decoded in isolation, so the string can differ from the
    /// token's contribution to the fully decoded text.
    pub token: String,
    /// Log probability of this token candidate.
    pub logprob: f32,
    /// One-based competition rank of this candidate at its position; tied log
    /// probabilities share a rank.
    pub rank: u32,
}

/// One position's decoded token candidates and their logprobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPositionLogprobs {
    /// Candidate tokens for this position, in engine order.
    ///
    /// For a generated position the first entry is the sampled token; the
    /// stream consumers in `serving::text::output` and `serving::assembly`
    /// reject engine output that violates this.
    pub entries: Vec<DecodedTokenLogprob>,
}

/// Decoded sample logprobs for generated token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedLogprobs {
    /// Generated token positions covered by this payload.
    pub positions: Vec<DecodedPositionLogprobs>,
}

/// Decoded prompt logprobs for prompt token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPromptLogprobs {
    /// Original vocabulary token ID for the first prompt token.
    pub first_token_id: u32,
    /// Best-effort decoded string for the first prompt token.
    ///
    /// The first prompt token has no left context to score against, so it is
    /// stored separately instead of appearing in `scored_positions`.
    pub first_token: String,
    /// Scored prompt positions after the first prompt token.
    ///
    /// `scored_positions[i]` corresponds to the prompt token at position `i +
    /// 1`.
    pub scored_positions: Vec<DecodedPositionLogprobs>,
}

/// Decodes generated-token logprobs from the engine's token-ID candidates into
/// the text-layer decoded-token representation.
///
/// Each returned position corresponds to one input position, in order.
pub(crate) fn decode_logprobs(
    tokenizer: &HuggingFaceTokenizer,
    positions: &[PositionLogprobs],
    skip_special_tokens: bool,
) -> Result<DecodedLogprobs, Error> {
    Ok(DecodedLogprobs {
        positions: positions
            .iter()
            .map(|position| decode_position_logprobs(tokenizer, position, skip_special_tokens))
            .try_collect()?,
    })
}

/// Decodes prompt logprobs from the engine's token-ID candidates into the
/// text-layer decoded-token representation.
///
/// The first prompt token is stored separately because scored positions begin
/// with the token predicted after it.
///
/// # Errors
///
/// Returns [`Error::EmptyPromptTokenIds`] for an empty prompt and a tokenizer
/// error when a token cannot be decoded.
pub(crate) fn decode_prompt_logprobs(
    request_id: &str,
    tokenizer: &HuggingFaceTokenizer,
    prompt_token_ids: &[u32],
    positions: &[PositionLogprobs],
    skip_special_tokens: bool,
) -> Result<DecodedPromptLogprobs, Error> {
    let Some(first_token_id) = prompt_token_ids.first().copied() else {
        return Err(Error::EmptyPromptTokenIds {
            request_id: request_id.to_string(),
        });
    };
    let first_token = tokenizer.decode(&[first_token_id], skip_special_tokens)?;
    let scored_positions = positions
        .iter()
        .map(|position| decode_position_logprobs(tokenizer, position, skip_special_tokens))
        .try_collect()?;

    Ok(DecodedPromptLogprobs {
        first_token_id,
        first_token,
        scored_positions,
    })
}

/// Decodes one token position's raw candidate set into decoded token strings
/// plus logprob metadata.
///
/// Every candidate token ID is decoded independently with the tokenizer.
fn decode_position_logprobs(
    tokenizer: &HuggingFaceTokenizer,
    position: &PositionLogprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPositionLogprobs, Error> {
    Ok(DecodedPositionLogprobs {
        entries: position
            .entries
            .iter()
            .map(|entry| {
                tokenizer
                    .decode(&[entry.token_id], skip_special_tokens)
                    .map(|token| DecodedTokenLogprob {
                        token_id: entry.token_id,
                        token,
                        logprob: entry.logprob,
                        rank: entry.rank,
                    })
            })
            .try_collect()?,
    })
}

#[cfg(test)]
mod tests {
    use uniserve_core::{PositionLogprobs, TokenLogprob};

    use super::*;

    // The configured test tokenizer maps each printable ASCII character to the
    // token ID equal to its code point, so `b'a' as u32` decodes to "a".

    #[test]
    fn decode_logprobs_decodes_every_candidate_token() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let logprobs = vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: b'a' as u32,
                    logprob: -0.1,
                    rank: 3,
                },
                TokenLogprob {
                    token_id: b'b' as u32,
                    logprob: -0.2,
                    rank: 1,
                },
            ],
        }];

        assert_eq!(
            decode_logprobs(tokenizer.as_ref(), &logprobs, false).unwrap(),
            DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token_id: b'a' as u32,
                            token: "a".to_string(),
                            logprob: -0.1,
                            rank: 3,
                        },
                        DecodedTokenLogprob {
                            token_id: b'b' as u32,
                            token: "b".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                }],
            }
        );
    }

    #[test]
    fn decode_prompt_logprobs_separates_first_prompt_token() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let logprobs = vec![PositionLogprobs {
            entries: vec![TokenLogprob {
                token_id: b'x' as u32,
                logprob: -0.4,
                rank: 1,
            }],
        }];

        assert_eq!(
            decode_prompt_logprobs(
                "test",
                tokenizer.as_ref(),
                &[b'p' as u32, b'x' as u32],
                &logprobs,
                false,
            )
            .unwrap(),
            DecodedPromptLogprobs {
                first_token_id: b'p' as u32,
                first_token: "p".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: b'x' as u32,
                        token: "x".to_string(),
                        logprob: -0.4,
                        rank: 1,
                    }],
                }],
            }
        );
    }
}
