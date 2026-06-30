//! Speculative-decode accounting: the default-off n-gram drafter's budget,
//! per-position acceptance counters, and the per-request draft-eligibility gate.

use std::env;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use uniserve_core::GenMode;

use crate::scheduler::{MAX_SPEC_DECODE_POS_STATS, ReqState, SchedStats};

const SPEC_NGRAM_MAX_TOKENS_ENV: &str = "UNISERVE_SPEC_NGRAM_MAX_TOKENS";
const SPEC_NGRAM_MAX_SUFFIX: usize = 8;

/// Owns the drafter budget and acceptance accounting. Per-request KV/control
/// state stays on the coordinator; this component receives the borrowed
/// [`ReqState`] for the requests it reasons about.
pub(crate) struct SpecDecodeAccounting {
    spec_ngram_max_tokens: usize,
    stats: Arc<SchedStats>,
}

impl SpecDecodeAccounting {
    pub(crate) fn new(stats: Arc<SchedStats>) -> Self {
        Self {
            spec_ngram_max_tokens: spec_ngram_max_tokens_from_env(),
            stats,
        }
    }

    pub(crate) fn max_ngram_tokens(&self) -> usize {
        self.spec_ngram_max_tokens
    }

    pub(crate) fn record_acceptance(&self, draft_tokens: usize, accepted: Option<u32>) {
        let accepted = accepted.unwrap_or(0) as usize;
        if accepted == 0 {
            return;
        }
        let accepted = accepted.min(draft_tokens);
        self.stats
            .spec_decode
            .num_accepted_tokens
            .fetch_add(accepted as u64, Ordering::Relaxed);
        for pos in 0..accepted.min(MAX_SPEC_DECODE_POS_STATS) {
            self.stats.spec_decode.num_accepted_tokens_per_pos[pos].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn draft_tokens(
        &self,
        st: &ReqState,
        current_token: u32,
        allowed: Option<&[u32]>,
        suppress: Option<&[u32]>,
    ) -> Option<Vec<u32>> {
        if self.spec_ngram_max_tokens == 0 {
            return None;
        }
        if !self.ngram_enabled_for(st) {
            return None;
        }
        let remaining_output = st.req.max_tokens.saturating_sub(st.n_generated);
        let max_draft = self
            .spec_ngram_max_tokens
            .min(remaining_output.saturating_sub(1));
        if max_draft == 0 {
            return None;
        }
        let mut seq = Vec::with_capacity(st.req.prompt_ids.len() + st.generated_ids.len() + 1);
        if let Some(recompute) = st.recompute_ids.as_ref() {
            seq.extend_from_slice(recompute);
        } else {
            seq.extend_from_slice(&st.req.prompt_ids);
            seq.extend_from_slice(&st.generated_ids);
        }
        if seq.last().copied() != Some(current_token) {
            seq.push(current_token);
        }
        let mut out = Vec::new();
        for _ in 0..max_draft {
            let Some(token) = ngram_draft_one(&seq, SPEC_NGRAM_MAX_SUFFIX, allowed, suppress)
            else {
                break;
            };
            out.push(token);
            seq.push(token);
        }
        (!out.is_empty()).then_some(out)
    }

    pub(crate) fn ngram_enabled_for(&self, st: &ReqState) -> bool {
        let sp = &st.req.sampling;
        // Worker target verification samples with temperature/top-k/top-p/min-p,
        // penalties, logit bias, and static allowed-token masks. Keep drafts off
        // for controls whose legal set can change inside the drafted prefix or
        // whose per-token response semantics are not yet represented.
        st.req.mode == GenMode::Text
            && st.grammar.is_none()
            && st.n_generated >= sp.min_tokens
            && sp.n_logprobs == 0
            && sp.bad_words_ids.is_empty()
    }
}

pub(crate) fn ngram_draft_one(
    sequence: &[u32],
    max_suffix: usize,
    allowed: Option<&[u32]>,
    suppress: Option<&[u32]>,
) -> Option<u32> {
    if sequence.len() < 2 || max_suffix == 0 {
        return None;
    }
    let max_suffix = max_suffix.min(sequence.len() - 1);
    for suffix_len in (1..=max_suffix).rev() {
        let suffix_start = sequence.len() - suffix_len;
        let suffix = &sequence[suffix_start..];
        for start in (0..suffix_start).rev() {
            let end = start + suffix_len;
            if end >= sequence.len() {
                continue;
            }
            if &sequence[start..end] == suffix {
                let candidate = sequence[end];
                if token_allowed(candidate, allowed, suppress) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn token_allowed(token: u32, allowed: Option<&[u32]>, suppress: Option<&[u32]>) -> bool {
    if let Some(allowed) = allowed
        && !allowed.contains(&token)
    {
        return false;
    }
    if let Some(suppress) = suppress
        && suppress.contains(&token)
    {
        return false;
    }
    true
}

fn spec_ngram_max_tokens_from_env() -> usize {
    env::var(SPEC_NGRAM_MAX_TOKENS_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}
