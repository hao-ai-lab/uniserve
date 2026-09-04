//! Host-side construction of position-dependent token masks.
//!
//! Numeric sampling transforms remain worker parameters. This module lowers
//! minimum-token, bad-word, and allowed-token rules into compact allow and
//! suppress lists for each step.

use uniserve_core::SamplingParams;

type TokenMask = Option<Vec<u32>>;
#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Allow and suppress masks computed for one sampling position.
pub(super) struct ProcessorMasks {
    pub(super) allowed: TokenMask,
    pub(super) suppress: TokenMask,
}

/// Per-request, per-step context a processor inspects.
pub(super) struct ProcCtx<'a> {
    pub(super) n_generated: usize,
    pub(super) eos: &'a [u32],
    pub(super) generated: &'a [u32],
    pub(super) sampling: &'a SamplingParams,
}

/// What a processor contributes to the op's masks.
#[derive(Default)]
struct MaskContribution {
    /// Restrict sampling to these ids (intersected across processors).
    pub allowed: Option<Vec<u32>>,
    /// Forbid sampling these ids (unioned across processors).
    pub suppress: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Host-evaluated logits rule that produces token masks.
pub(super) enum BuiltinLogitsProcessor {
    MinTokens,
    BadWords,
    AllowedTokens,
}

const DEFAULT_PIPELINE: [BuiltinLogitsProcessor; 3] = [
    BuiltinLogitsProcessor::MinTokens,
    BuiltinLogitsProcessor::BadWords,
    BuiltinLogitsProcessor::AllowedTokens,
];

impl BuiltinLogitsProcessor {
    /// Produces the allowlist and suppression masks contributed by one logits rule.
    fn contribute(self, ctx: &ProcCtx<'_>) -> MaskContribution {
        match self {
            Self::MinTokens => {
                let mut contribution = MaskContribution::default();
                if ctx.n_generated < ctx.sampling.min_tokens {
                    contribution.suppress.extend_from_slice(ctx.eos);
                }
                contribution
            }
            Self::BadWords => {
                let mut contribution = MaskContribution::default();
                for bad_word in &ctx.sampling.bad_words_ids {
                    if bad_word.is_empty() {
                        continue;
                    }
                    let prefix_len = bad_word.len() - 1;
                    if ctx.generated.len() >= prefix_len
                        && ctx.generated[ctx.generated.len() - prefix_len..]
                            == bad_word[..prefix_len]
                    {
                        contribution.suppress.push(bad_word[prefix_len]);
                    }
                }
                contribution
            }
            Self::AllowedTokens => MaskContribution {
                allowed: ctx.sampling.allowed_token_ids.clone(),
                suppress: Vec::new(),
            },
        }
    }
}

/// Returns mask processors in their canonical composition order.
pub(super) fn default_pipeline() -> Vec<BuiltinLogitsProcessor> {
    DEFAULT_PIPELINE.to_vec()
}

/// Merges all processors' contributions into a single `(allowed, suppress)` pair.
pub(super) fn run_pipeline(
    pipeline: &[BuiltinLogitsProcessor],
    ctx: &ProcCtx<'_>,
) -> ProcessorMasks {
    let mut allowed: Option<Vec<u32>> = None;
    let mut suppress: Vec<u32> = Vec::new();
    for processor in pipeline {
        let c = processor.contribute(ctx);
        if let Some(a) = c.allowed {
            allowed = Some(match allowed {
                None => a,
                Some(prev) => prev.into_iter().filter(|x| a.contains(x)).collect(),
            });
        }
        suppress.extend(c.suppress);
    }
    suppress.sort_unstable();
    suppress.dedup();
    let suppress = if suppress.is_empty() {
        None
    } else {
        Some(suppress)
    };
    ProcessorMasks { allowed, suppress }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(
        n: usize,
        eos: &'a [u32],
        generated: &'a [u32],
        sp: &'a SamplingParams,
    ) -> ProcCtx<'a> {
        ProcCtx {
            n_generated: n,
            eos,
            generated,
            sampling: sp,
        }
    }

    #[test]
    fn min_tokens_suppresses_eos_until_floor() {
        let pipe = default_pipeline();
        let sp = SamplingParams {
            min_tokens: 5,
            ..Default::default()
        };
        let masks = run_pipeline(&pipe, &ctx(2, &[151645, 151643], &[], &sp));
        assert_eq!(masks.suppress, Some(vec![151643, 151645]));
        // at/after the floor, EOS is allowed again
        let masks = run_pipeline(&pipe, &ctx(5, &[151645, 151643], &[], &sp));
        assert_eq!(masks.suppress, None);
    }

    #[test]
    fn bad_words_suppresses_completion() {
        let pipe = default_pipeline();
        let sp = SamplingParams {
            bad_words_ids: vec![vec![7, 8, 9]],
            ..Default::default()
        };
        // recent ends in [7, 8] -> emitting 9 would complete the bad word.
        let masks = run_pipeline(&pipe, &ctx(2, &[0], &[1, 7, 8], &sp));
        assert_eq!(masks.suppress, Some(vec![9]));
        // recent does not match the prefix -> no suppression.
        let masks = run_pipeline(&pipe, &ctx(2, &[0], &[1, 2, 3], &sp));
        assert_eq!(masks.suppress, None);
    }

    #[test]
    fn allowed_tokens_passes_through() {
        let pipe = default_pipeline();
        let sp = SamplingParams {
            allowed_token_ids: Some(vec![3, 4]),
            ..Default::default()
        };
        let masks = run_pipeline(&pipe, &ctx(0, &[0], &[], &sp));
        assert_eq!(masks.allowed, Some(vec![3, 4]));
    }
}
