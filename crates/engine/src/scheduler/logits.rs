//! Host-side logits-processor pipeline. The math processors (min-p, penalties,
//! logit bias, top-k/p, temperature) are parameters the device sampler consumes;
//! the control-flow processors here compute the minimum-token floor, bad-word,
//! and allowed-token masks and express them to the worker as small id masks
//! (`allowed_tokens`/`suppress_tokens`).
//!
//! The pipeline is the seam: a processor is added by pushing it onto the
//! pipeline (`Scheduler::with_logits_processor`) and the scheduler's per-step
//! masking code is unchanged. The built-in masks are cheap host- or
//! position-derived computations that run inline; an admitted custom processor
//! runs on the shared bounded CPU-continuation future so its arbitrary host code
//! suspends only its own request lineage.

use std::sync::Arc;
use uniserve_core::SamplingParams;

const TOKEN_ID_OUTPUT_BOUND: usize = u32::MAX as usize;
pub type TokenMask = Option<Vec<u32>>;
pub type ProcessorMasks = (TokenMask, TokenMask);

/// Per-request, per-step context a processor inspects.
pub struct ProcCtx<'a> {
    pub n_generated: usize,
    pub eos: &'a [u32],
    pub generated: &'a [u32],
    pub sampling: &'a SamplingParams,
}

/// What a processor contributes to the op's masks.
#[derive(Default)]
pub struct MaskContribution {
    /// Restrict sampling to these ids (intersected across processors).
    pub allowed: Option<Vec<u32>>,
    /// Forbid sampling these ids (unioned across processors).
    pub suppress: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorDeclaration {
    pub snapshotable: bool,
    pub deterministic: bool,
    pub max_output_tokens: usize,
    pub max_outstanding_tasks: usize,
}

pub trait LogitsProcessor: Send + Sync {
    fn name(&self) -> &'static str;
    fn declaration(&self) -> ProcessorDeclaration;
    /// the reference `is_argmax_invariant`: whether the processor can change the argmax.
    fn is_argmax_invariant(&self) -> bool;
    fn contribute(&self, ctx: &ProcCtx) -> MaskContribution;
}

/// Suppress EOS (and other stop ids) until `min_tokens` is reached
/// (`MinTokensLogitsProcessor`, `builtin.py:167`).
pub struct MinTokensProcessor;
impl LogitsProcessor for MinTokensProcessor {
    fn name(&self) -> &'static str {
        "min_tokens"
    }
    fn declaration(&self) -> ProcessorDeclaration {
        ProcessorDeclaration {
            snapshotable: true,
            deterministic: true,
            max_output_tokens: TOKEN_ID_OUTPUT_BOUND,
            max_outstanding_tasks: 1,
        }
    }
    fn is_argmax_invariant(&self) -> bool {
        false
    }
    fn contribute(&self, ctx: &ProcCtx) -> MaskContribution {
        let mut c = MaskContribution::default();
        if ctx.n_generated < ctx.sampling.min_tokens {
            c.suppress.extend_from_slice(ctx.eos);
        }
        c
    }
}

/// Suppress the token that would *complete* any configured bad-word sequence.
pub struct BadWordsProcessor;
impl LogitsProcessor for BadWordsProcessor {
    fn name(&self) -> &'static str {
        "bad_words"
    }
    fn declaration(&self) -> ProcessorDeclaration {
        ProcessorDeclaration {
            snapshotable: true,
            deterministic: true,
            max_output_tokens: TOKEN_ID_OUTPUT_BOUND,
            max_outstanding_tasks: 1,
        }
    }
    fn is_argmax_invariant(&self) -> bool {
        false
    }
    fn contribute(&self, ctx: &ProcCtx) -> MaskContribution {
        let mut c = MaskContribution::default();
        for bw in &ctx.sampling.bad_words_ids {
            if bw.is_empty() {
                continue;
            }
            let k = bw.len() - 1; // prefix length to match
            if ctx.generated.len() >= k && ctx.generated[ctx.generated.len() - k..] == bw[..k] {
                c.suppress.push(bw[k]);
            }
        }
        c
    }
}

/// Restrict sampling to a whitelist (`allowed_token_ids`).
pub struct AllowedTokensProcessor;
impl LogitsProcessor for AllowedTokensProcessor {
    fn name(&self) -> &'static str {
        "allowed_tokens"
    }
    fn declaration(&self) -> ProcessorDeclaration {
        ProcessorDeclaration {
            snapshotable: true,
            deterministic: true,
            max_output_tokens: TOKEN_ID_OUTPUT_BOUND,
            max_outstanding_tasks: 1,
        }
    }
    fn is_argmax_invariant(&self) -> bool {
        false
    }
    fn contribute(&self, ctx: &ProcCtx) -> MaskContribution {
        MaskContribution {
            allowed: ctx.sampling.allowed_token_ids.clone(),
            suppress: Vec::new(),
        }
    }
}

/// The default control-flow pipeline (order matches the reference non-argmax-invariant set).
pub fn default_pipeline() -> Vec<Arc<dyn LogitsProcessor>> {
    vec![
        Arc::new(MinTokensProcessor),
        Arc::new(BadWordsProcessor),
        Arc::new(AllowedTokensProcessor),
    ]
}

/// Merge all processors' contributions into a single `(allowed, suppress)` pair.
pub fn run_pipeline(pipeline: &[Arc<dyn LogitsProcessor>], ctx: &ProcCtx) -> ProcessorMasks {
    let mut allowed: Option<Vec<u32>> = None;
    let mut suppress: Vec<u32> = Vec::new();
    for p in pipeline {
        let c = p.contribute(ctx);
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
    (allowed, suppress)
}

pub(crate) fn run_pipeline_checked(
    pipeline: &[Arc<dyn LogitsProcessor>],
    ctx: &ProcCtx,
) -> Result<ProcessorMasks, String> {
    let mut allowed: Option<Vec<u32>> = None;
    let mut suppress = Vec::new();
    for processor in pipeline {
        let contribution = processor.contribute(ctx);
        let declaration = processor.declaration();
        let output_tokens = contribution
            .allowed
            .as_ref()
            .map_or(0, Vec::len)
            .saturating_add(contribution.suppress.len());
        if output_tokens > declaration.max_output_tokens {
            return Err(format!(
                "logits processor `{}` exceeded its declared output bound",
                processor.name()
            ));
        }
        if let Some(tokens) = contribution.allowed {
            allowed = Some(match allowed {
                None => tokens,
                Some(previous) => previous
                    .into_iter()
                    .filter(|token| tokens.contains(token))
                    .collect(),
            });
        }
        suppress.extend(contribution.suppress);
    }
    suppress.sort_unstable();
    suppress.dedup();
    Ok((
        allowed,
        if suppress.is_empty() {
            None
        } else {
            Some(suppress)
        },
    ))
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
        let (_a, sup) = run_pipeline(&pipe, &ctx(2, &[151645, 151643], &[], &sp));
        assert_eq!(sup, Some(vec![151643, 151645]));
        // at/after the floor, EOS is allowed again
        let (_a, sup) = run_pipeline(&pipe, &ctx(5, &[151645, 151643], &[], &sp));
        assert_eq!(sup, None);
    }

    #[test]
    fn bad_words_suppresses_completion() {
        let pipe = default_pipeline();
        let sp = SamplingParams {
            bad_words_ids: vec![vec![7, 8, 9]],
            ..Default::default()
        };
        // recent ends in [7, 8] -> emitting 9 would complete the bad word.
        let (_a, sup) = run_pipeline(&pipe, &ctx(2, &[0], &[1, 7, 8], &sp));
        assert_eq!(sup, Some(vec![9]));
        // recent does not match the prefix -> no suppression.
        let (_a, sup) = run_pipeline(&pipe, &ctx(2, &[0], &[1, 2, 3], &sp));
        assert_eq!(sup, None);
    }

    #[test]
    fn allowed_tokens_passes_through() {
        let pipe = default_pipeline();
        let sp = SamplingParams {
            allowed_token_ids: Some(vec![3, 4]),
            ..Default::default()
        };
        let (allow, _s) = run_pipeline(&pipe, &ctx(0, &[0], &[], &sp));
        assert_eq!(allow, Some(vec![3, 4]));
    }

    /// Adding a processor needs no scheduler edit — just push onto the pipeline.
    #[test]
    fn custom_processor_composes() {
        struct BanZero;
        impl LogitsProcessor for BanZero {
            fn name(&self) -> &'static str {
                "ban_zero"
            }
            fn declaration(&self) -> ProcessorDeclaration {
                ProcessorDeclaration {
                    snapshotable: true,
                    deterministic: true,
                    max_output_tokens: 1,
                    max_outstanding_tasks: 1,
                }
            }
            fn is_argmax_invariant(&self) -> bool {
                false
            }
            fn contribute(&self, _ctx: &ProcCtx) -> MaskContribution {
                MaskContribution {
                    allowed: None,
                    suppress: vec![0],
                }
            }
        }
        let mut pipe = default_pipeline();
        pipe.push(Arc::new(BanZero));
        let sp = SamplingParams {
            min_tokens: 3,
            ..Default::default()
        };
        let (_a, sup) = run_pipeline(&pipe, &ctx(0, &[42], &[], &sp));
        let sup = sup.unwrap();
        assert!(sup.contains(&0) && sup.contains(&42));
    }
}
