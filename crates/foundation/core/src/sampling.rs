//! The worker-side sampling math, shared by the GPU-free `SimEngine` and
//! mirrored by the Python worker. It applies the logits transforms in the reference's
//! order:
//!
//! allowed-token mask → suppress (bad-words / min-tokens) → logit bias →
//! penalties (repetition / frequency / presence over the recent window) →
//! temperature → min-p → top-k → top-p → sample → gather logprobs.
//!
//! It operates on a logits slice and the per-request `SamplingParams` plus the
//! small descriptor lists the host computed (recent tokens, allowed/suppress
//! masks). No logits tensor crosses the wire — this runs inside the worker.

use std::collections::BTreeMap;

use crate::SamplingParams;

#[derive(Debug, Clone, PartialEq)]
pub struct SampleOutput {
    pub token: u32,
    pub logprob: f32,
    /// Ranked sampled, top, and explicitly requested vocabulary candidates.
    pub top: Vec<(u32, f32, u32)>,
}

const NEG_INF: f32 = f32::NEG_INFINITY;

/// Apply the full sampling pipeline to `logits` (modified in place) and draw a
/// token. `recent` is the bounded recent-output window for penalties; `allowed`
/// / `suppress` are the host-computed masks.
pub fn apply_sampling(
    logits: &mut [f32],
    p: &SamplingParams,
    recent: &[u32],
    allowed: Option<&[u32]>,
    suppress: Option<&[u32]>,
    n_logprobs: usize,
) -> SampleOutput {
    let mut counts = BTreeMap::<u32, u32>::new();
    for &token in recent {
        let count = counts.entry(token).or_default();
        *count = count.saturating_add(1);
    }
    try_apply_sampling_counts(
        logits,
        p,
        &counts.into_iter().collect::<Vec<_>>(),
        allowed,
        suppress,
        n_logprobs,
        seed_from(p, recent),
    )
    .expect("sampling inputs must leave a valid distribution")
}

/// Apply the full sampling pipeline from canonical branch-local token counts.
///
/// `draw_seed` names one semantic sampling coordinate. `None` represents a
/// deterministic invalid distribution: empty logits, NaNs, infinities, or a
/// transform sequence that masks every vocabulary entry.
pub fn try_apply_sampling_counts(
    logits: &mut [f32],
    p: &SamplingParams,
    recent_counts: &[(u32, u32)],
    allowed: Option<&[u32]>,
    suppress: Option<&[u32]>,
    n_logprobs: usize,
    draw_seed: u64,
) -> Option<SampleOutput> {
    let v = logits.len();
    if !valid_distribution(logits) {
        return None;
    }

    // 1. allowed-token whitelist: mask everything else.
    if let Some(allow) = allowed {
        let mut keep = vec![false; v];
        for &t in allow {
            if (t as usize) < v {
                keep[t as usize] = true;
            }
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *l = NEG_INF;
            }
        }
    }
    // 2. suppress (bad-words completion / min-tokens EOS floor).
    if let Some(sup) = suppress {
        for &t in sup {
            if (t as usize) < v {
                logits[t as usize] = NEG_INF;
            }
        }
    }
    // 3. logit bias.
    for &(t, b) in &p.logit_bias {
        if (t as usize) < v && logits[t as usize] != NEG_INF {
            logits[t as usize] += b;
        }
    }
    // 4. penalties over the recent output window.
    if p.repetition_penalty != 1.0 || p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
        for &(t, count) in recent_counts {
            let i = t as usize;
            if count == 0 || i >= v || logits[i] == NEG_INF {
                continue;
            }
            let c = count as f32;
            // repetition penalty (multiplicative, sign-aware — reference semantics)
            if p.repetition_penalty != 1.0 {
                logits[i] = if logits[i] > 0.0 {
                    logits[i] / p.repetition_penalty
                } else {
                    logits[i] * p.repetition_penalty
                };
            }
            // frequency (scaled by count) + presence (flat, once-appeared)
            logits[i] -= p.frequency_penalty * c;
            logits[i] -= p.presence_penalty;
        }
    }
    if !valid_distribution(logits) {
        return None;
    }
    // 5. temperature (0 == greedy; applied at sample time).
    let greedy = p.temperature <= 0.0;
    if !greedy {
        for l in logits.iter_mut() {
            if *l != NEG_INF {
                *l /= p.temperature;
            }
        }
    }
    // 6. min-p: drop tokens below `min_p * max_prob`.
    if p.min_p > 0.0 {
        let probs = softmax(logits);
        let maxp = probs.iter().cloned().fold(0.0f32, f32::max);
        let thresh = p.min_p * maxp;
        for (i, &pr) in probs.iter().enumerate() {
            if pr < thresh {
                logits[i] = NEG_INF;
            }
        }
    }
    // 7. top-k.
    if p.top_k > 0 && (p.top_k as usize) < v {
        let mut idx: Vec<usize> = (0..v).filter(|&i| logits[i] != NEG_INF).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in idx.iter().skip(p.top_k as usize) {
            logits[i] = NEG_INF;
        }
    }
    // 8. top-p (nucleus).
    if p.top_p < 1.0 && p.top_p > 0.0 {
        let probs = softmax(logits);
        let mut order: Vec<usize> = (0..v).filter(|&i| logits[i] != NEG_INF).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &i) in order.iter().enumerate() {
            cum += probs[i];
            if cum >= p.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        for &i in order.iter().skip(cutoff) {
            logits[i] = NEG_INF;
        }
    }
    if !valid_distribution(logits) {
        return None;
    }

    // 9. sample.
    let token = if greedy {
        argmax(logits)
    } else {
        let probs = softmax(logits);
        sample_categorical(&probs, draw_seed)
    };

    // 10. gather logprobs (softmax of the final, masked logits).
    let logprobs = log_softmax(logits);
    let sampled_lp = logprobs.get(token as usize).copied().unwrap_or(NEG_INF);
    let top = if p.generated_logprobs_requested() || n_logprobs > 0 {
        score_token_logprobs(logits, token, n_logprobs, &p.logprob_token_ids)
    } else {
        Vec::new()
    };
    Some(SampleOutput {
        token,
        logprob: sampled_lp,
        top,
    })
}

/// Score one known token against a vocabulary-logits row and return ranked candidates.
pub fn score_token_logprobs(
    logits: &[f32],
    token: u32,
    n_logprobs: usize,
    requested_token_ids: &[u32],
) -> Vec<(u32, f32, u32)> {
    let logprobs = log_softmax(logits);
    let mut order: Vec<usize> = (0..logits.len())
        .filter(|&index| logits[index] != NEG_INF)
        .collect();
    order.sort_by(|&a, &b| {
        logprobs[b]
            .partial_cmp(&logprobs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let token_index = token as usize;
    if token_index >= logits.len() {
        return Vec::new();
    }
    let sampled_rank = competition_rank(&logprobs, logprobs[token_index]);
    let mut entries = vec![(token, logprobs[token_index], sampled_rank)];
    for &index in order.iter().take(n_logprobs) {
        let token_id = index as u32;
        if !entries.iter().any(|(existing, _, _)| *existing == token_id) {
            entries.push((
                token_id,
                logprobs[index],
                competition_rank(&logprobs, logprobs[index]),
            ));
        }
    }
    for &token_id in requested_token_ids {
        let index = token_id as usize;
        if index < logits.len() && !entries.iter().any(|(existing, _, _)| *existing == token_id) {
            entries.push((
                token_id,
                logprobs[index],
                competition_rank(&logprobs, logprobs[index]),
            ));
        }
    }
    entries
}

fn competition_rank(logprobs: &[f32], value: f32) -> u32 {
    let strictly_greater = logprobs
        .iter()
        .filter(|&&candidate| candidate > value)
        .count();
    u32::try_from(strictly_greater)
        .unwrap_or(u32::MAX)
        .saturating_add(1)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bestv = NEG_INF;
    for (i, &l) in logits.iter().enumerate() {
        if l > bestv {
            bestv = l;
            best = i;
        }
    }
    best as u32
}

fn valid_distribution(logits: &[f32]) -> bool {
    !logits.is_empty()
        && logits
            .iter()
            .all(|value| value.is_finite() || *value == NEG_INF)
        && logits.iter().any(|value| value.is_finite())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(NEG_INF, f32::max);
    if m == NEG_INF {
        return vec![0.0; logits.len()];
    }
    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&l| if l == NEG_INF { 0.0 } else { (l - m).exp() })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(NEG_INF, f32::max);
    if m == NEG_INF {
        return vec![NEG_INF; logits.len()];
    }
    let sum: f32 = logits
        .iter()
        .map(|&l| if l == NEG_INF { 0.0 } else { (l - m).exp() })
        .sum();
    let lse = m + sum.ln();
    logits
        .iter()
        .map(|&l| if l == NEG_INF { NEG_INF } else { l - lse })
        .collect()
}

fn sample_categorical(probs: &[f32], rng: u64) -> u32 {
    // splitmix64 finalizer for a deterministic, dependency-free draw. A single
    // xorshift step leaves nearby seeds (which `seed_from` readily produces —
    // consecutive `recent.len` or adjacent last-token ids) strongly
    // correlated; splitmix64's avalanche mixes those into well-separated draws.
    let r = ((splitmix64(rng) >> 11) as f64 / (1u64 << 53) as f64) as f32;
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

/// splitmix64 mixing function: a strong, dependency-free finalizer that maps a
/// 64-bit seed to a well-distributed 64-bit value, decorrelating seeds that
/// differ only in their low bits.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Derive one random seed from request identity and semantic sampling
/// coordinates. The mapping is independent of batching, launch order, and
/// completion order and mirrors the production worker's coordinate function.
#[allow(clippy::too_many_arguments)]
pub fn semantic_sampling_seed(
    session_seed: u64,
    authority_id: u64,
    session_id: u64,
    epoch: u64,
    semantic_token_index: u64,
    processor_stage: u64,
    draw_index: u64,
) -> u64 {
    [
        authority_id,
        session_id,
        epoch,
        semantic_token_index,
        processor_stage,
        draw_index,
    ]
    .into_iter()
    .fold(session_seed, splitmix_coordinate)
}

fn splitmix_coordinate(seed: u64, coordinate: u64) -> u64 {
    let mut value = seed.wrapping_add(coordinate.wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn seed_from(p: &SamplingParams, recent: &[u32]) -> u64 {
    let mut s = p.seed.unwrap_or(0x9E3779B97F4A7C15);
    s = s.wrapping_add(recent.len() as u64);
    if let Some(&last) = recent.last() {
        s = s.wrapping_mul(0x100000001B3).wrapping_add(last as u64);
    }
    s | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_logits() -> Vec<f32> {
        vec![0.0, 1.0, 2.0, 3.0, 0.5]
    } // argmax = 3

    #[test]
    fn greedy_argmax_default() {
        let mut l = base_logits();
        let out = apply_sampling(&mut l, &SamplingParams::default(), &[], None, None, 0);
        assert_eq!(out.token, 3);
    }

    #[test]
    fn logit_bias_changes_winner() {
        let mut l = base_logits();
        let p = SamplingParams {
            logit_bias: vec![(0, 100.0)],
            ..Default::default()
        };
        let out = apply_sampling(&mut l, &p, &[], None, None, 0);
        assert_eq!(out.token, 0, "huge bias must make token 0 win");
    }

    #[test]
    fn allowed_tokens_restricts() {
        let mut l = base_logits();
        let out = apply_sampling(
            &mut l,
            &SamplingParams::default(),
            &[],
            Some(&[1, 4]),
            None,
            0,
        );
        assert_eq!(out.token, 1, "only 1 and 4 allowed; 1 has higher logit");
    }

    #[test]
    fn suppress_masks_argmax() {
        let mut l = base_logits();
        let out = apply_sampling(&mut l, &SamplingParams::default(), &[], None, Some(&[3]), 0);
        assert_eq!(out.token, 2, "token 3 suppressed; next is 2");
    }

    #[test]
    fn canonical_recent_counts_drive_penalties() {
        let mut logits = base_logits();
        let output = try_apply_sampling_counts(
            &mut logits,
            &SamplingParams {
                frequency_penalty: 2.0,
                ..Default::default()
            },
            &[(3, 2)],
            None,
            None,
            0,
            0,
        )
        .expect("valid distribution");
        assert_ne!(output.token, 3);
        assert_eq!(logits[3], -1.0);
    }

    #[test]
    fn all_masked_and_non_finite_distributions_are_invalid() {
        let params = SamplingParams::default();
        let mut all_masked = base_logits();
        assert!(
            try_apply_sampling_counts(&mut all_masked, &params, &[], Some(&[]), None, 0, 0,)
                .is_none()
        );

        let mut nan = vec![0.0, f32::NAN];
        assert!(try_apply_sampling_counts(&mut nan, &params, &[], None, None, 0, 0).is_none());
    }

    #[test]
    fn semantic_sampling_coordinate_matches_worker_mapping() {
        assert_eq!(
            semantic_sampling_seed(11, 3, 5, 7, 13, 17, 19),
            13_207_301_388_388_605_805
        );
    }

    #[test]
    fn repetition_penalty_demotes_recent() {
        // token 3 is argmax; penalize it heavily via repetition over recent.
        let mut l = base_logits();
        let p = SamplingParams {
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let out = apply_sampling(&mut l, &p, &[3, 3, 3], None, None, 0);
        assert_ne!(out.token, 3, "repeated token 3 should be demoted");
    }

    #[test]
    fn min_p_and_logprobs() {
        let mut l = base_logits();
        let p = SamplingParams {
            n_logprobs: 3,
            ..Default::default()
        };
        let out = apply_sampling(&mut l, &p, &[], None, None, 3);
        assert_eq!(out.token, 3);
        assert_eq!(out.top.len(), 3);
        assert_eq!(out.top[0].0, 3); // highest-logprob token first
        assert!(out.logprob <= 0.0); // a log-probability
    }

    #[test]
    fn logprob_ranks_use_competition_ranking_for_ties_and_masked_requests() {
        let logits = [2.0, 2.0, 1.0, NEG_INF, NEG_INF];
        let scored = score_token_logprobs(&logits, 1, logits.len(), &[4]);

        assert_eq!(
            scored
                .iter()
                .map(|&(token, _, rank)| (token, rank))
                .collect::<Vec<_>>(),
            vec![(1, 1), (0, 1), (2, 3), (4, 4)]
        );
        assert!(scored.iter().all(|&(_, _, rank)| rank >= 1));
    }

    #[test]
    fn defaults_are_noop_argmax() {
        // Every transform unset => plain argmax, deterministic.
        let mut l = base_logits();
        let out = apply_sampling(&mut l, &SamplingParams::default(), &[1, 2], None, None, 0);
        assert_eq!(out.token, 3);
    }

    // ----------------------------------------------------------------------
    // Masking transforms (min-p / top-k / top-p) prune low-probability tokens.
    //
    // With greedy defaults (temperature 0.0 => no temperature scaling, top_p 1.0,
    // top_k 0, min_p 0.0 all no-op) the ONLY transform that touches the in-place
    // `logits` slice is the one we enable, so reading the slice back after the
    // call observes exactly which positions were pruned to NEG_INF. The base
    // logits [3,2,1,0] have softmax probs ~[0.644, 0.237, 0.087, 0.032]
    // (descending), which pins the cutoffs below.
    // ----------------------------------------------------------------------

    fn ramp_logits() -> Vec<f32> {
        vec![3.0, 2.0, 1.0, 0.0]
    }

    #[test]
    fn min_p_prunes_below_relative_threshold() {
        // min_p 0.3 => threshold = 0.3 * maxprob(0.644) = 0.193.
        // token0 (0.644) and token1 (0.237) survive; token2 (0.087) and
        // token3 (0.032) fall below threshold and are masked to NEG_INF.
        let mut l = ramp_logits();
        let p = SamplingParams {
            min_p: 0.3,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[], None, None, 0);
        assert!(l[0].is_finite(), "top token survives min-p");
        assert!(l[1].is_finite(), "second token survives min-p");
        assert_eq!(l[2], NEG_INF, "token below min_p*maxp pruned");
        assert_eq!(l[3], NEG_INF, "token below min_p*maxp pruned");
    }

    #[test]
    fn top_k_keeps_only_k_highest_logits() {
        // top_k = 2 keeps the two highest logits (tokens 0 and 1); the rest
        // are masked to NEG_INF regardless of how close their probs are.
        let mut l = ramp_logits();
        let p = SamplingParams {
            top_k: 2,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[], None, None, 0);
        assert!(l[0].is_finite(), "highest logit kept");
        assert!(l[1].is_finite(), "second-highest logit kept");
        assert_eq!(l[2], NEG_INF, "3rd-ranked logit pruned by top_k=2");
        assert_eq!(l[3], NEG_INF, "4th-ranked logit pruned by top_k=2");
    }

    #[test]
    fn top_p_nucleus_keeps_smallest_set_reaching_mass() {
        // top_p 0.7: descending cumulative prob is 0.644 (rank0), 0.881 (rank1),
        // 0.968 (rank2). The cumulative first reaches >=0.7 at rank1, so the
        // nucleus is {token0, token1}; tokens 2 and 3 are pruned to NEG_INF.
        let mut l = ramp_logits();
        let p = SamplingParams {
            top_p: 0.7,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[], None, None, 0);
        assert!(l[0].is_finite(), "token0 inside nucleus");
        assert!(l[1].is_finite(), "token1 completes nucleus mass >=0.7");
        assert_eq!(l[2], NEG_INF, "token2 outside top-p nucleus pruned");
        assert_eq!(l[3], NEG_INF, "token3 outside top-p nucleus pruned");
    }

    #[test]
    fn masking_excludes_pruned_tokens_from_logprobs() {
        // The returned top-logprobs list only contains surviving (non-NEG_INF)
        // tokens: top_k=2 leaves 2 candidates, so requesting 5 logprobs returns
        // exactly the 2 survivors and never a pruned id.
        let mut l = ramp_logits();
        let p = SamplingParams {
            top_k: 2,
            ..Default::default()
        };
        let out = apply_sampling(&mut l, &p, &[], None, None, 5);
        assert_eq!(out.top.len(), 2, "only the 2 survivors are reportable");
        let reported: Vec<u32> = out.top.iter().map(|&(t, _, _)| t).collect();
        assert!(
            !reported.contains(&2),
            "pruned token 2 excluded from logprobs"
        );
        assert!(
            !reported.contains(&3),
            "pruned token 3 excluded from logprobs"
        );
    }

    // ----------------------------------------------------------------------
    // Penalty semantics: frequency (count-scaled additive), presence (flat
    // additive), repetition (multiplicative, sign-aware) are three distinct
    // transforms. With greedy defaults the penalty is the only transform applied
    // to the in-place logits, so reading the slice back reveals the exact math.
    // ----------------------------------------------------------------------

    #[test]
    fn frequency_penalty_scales_with_recent_count() {
        // Equal base logits; token1 appears twice in `recent`, token2 once.
        // frequency penalty subtracts penalty*count, so the more-repeated token1
        // is demoted strictly more than token2, and a non-recent token is
        // untouched. This count scaling is what distinguishes it from presence.
        let mut l = vec![3.0f32, 3.0, 3.0, 3.0];
        let p = SamplingParams {
            frequency_penalty: 1.0,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[1, 1, 2], None, None, 0);
        assert_eq!(l[0], 3.0, "non-recent token untouched");
        assert_eq!(l[3], 3.0, "non-recent token untouched");
        assert_eq!(l[2], 2.0, "count 1 => -1.0");
        assert_eq!(l[1], 1.0, "count 2 => -2.0 (scaled by count)");
        assert!(l[1] < l[2], "higher count demoted more under frequency");
    }

    #[test]
    fn presence_penalty_is_flat_regardless_of_count() {
        // Same recent window as the frequency test, but presence subtracts a
        // flat penalty once for any appearance: token1 (count 2) and token2
        // (count 1) are demoted by the SAME amount, leaving them tied.
        let mut l = vec![3.0f32, 3.0, 3.0, 3.0];
        let p = SamplingParams {
            presence_penalty: 1.0,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[1, 1, 2], None, None, 0);
        assert_eq!(l[0], 3.0, "non-recent token untouched");
        assert_eq!(l[3], 3.0, "non-recent token untouched");
        assert_eq!(l[1], 2.0, "appeared => flat -1.0");
        assert_eq!(l[2], 2.0, "appeared => flat -1.0");
        assert_eq!(l[1], l[2], "presence ignores count: equal demotion");
    }

    #[test]
    fn repetition_penalty_is_multiplicative_and_sign_aware() {
        // token0 has a positive logit, token1 a negative one, both in `recent`.
        // Repetition penalty divides positive logits and multiplies negative
        // logits by the penalty (sign-aware), unlike the additive penalties
        // which would shift both by the same constant.
        let mut l = vec![4.0f32, -4.0, 0.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[0, 1], None, None, 0);
        assert_eq!(l[0], 2.0, "positive logit DIVIDED by penalty (4/2)");
        assert_eq!(l[1], -8.0, "negative logit MULTIPLIED by penalty (-4*2)");
        assert_eq!(l[2], 0.0, "non-recent token untouched");
        // Distinct from additive: the two recent tokens moved by different
        // deltas (-2 and -4), whereas frequency/presence with count 1 each would
        // shift both by an identical constant.
        let delta0 = 4.0 - l[0];
        let delta1 = -4.0 - l[1];
        assert_ne!(delta0, delta1, "sign-aware deltas differ, not a flat shift");
    }

    #[test]
    fn frequency_and_presence_diverge_on_repeated_token() {
        // Direct contrast: with the SAME recent window, frequency demotes the
        // twice-seen token below the once-seen token, while presence leaves them
        // tied. Asserting both orderings in one place proves the semantics are
        // genuinely different transforms, not aliases.
        let recent = [1u32, 1, 2];

        let mut lf = vec![3.0f32, 3.0, 3.0];
        apply_sampling(
            &mut lf,
            &SamplingParams {
                frequency_penalty: 0.5,
                ..Default::default()
            },
            &recent,
            None,
            None,
            0,
        );
        assert!(
            lf[1] < lf[2],
            "frequency: count-2 token below count-1 token"
        );

        let mut lp = vec![3.0f32, 3.0, 3.0];
        apply_sampling(
            &mut lp,
            &SamplingParams {
                presence_penalty: 0.5,
                ..Default::default()
            },
            &recent,
            None,
            None,
            0,
        );
        assert_eq!(lp[1], lp[2], "presence: count-2 and count-1 tokens tied");
    }

    // ----------------------------------------------------------------------
    // sample_categorical determinism + splitmix64 avalanche, exercised through
    // the public `apply_sampling` path with temperature > 0 so the categorical
    // branch (not argmax) runs. Properties are asserted statistically with a
    // documented tolerance, decoupled from the exact mixing constants.
    // ----------------------------------------------------------------------

    /// Draw a token through the public sampler for a given integer seed over a
    /// flat (uniform-target) logits vector of width `v`.
    fn sample_uniform(seed: u64, v: usize) -> u32 {
        let mut l = vec![0.0f32; v]; // equal logits => uniform target probs
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(seed),
            ..Default::default()
        };
        apply_sampling(&mut l, &p, &[], None, None, 0).token
    }

    #[test]
    fn sample_categorical_is_deterministic_per_seed() {
        // Identical seed + params + recent window => identical token, every time.
        for &seed in &[1u64, 7, 12345, 0xDEAD_BEEF, u64::MAX] {
            let a = sample_uniform(seed, 16);
            let b = sample_uniform(seed, 16);
            assert_eq!(a, b, "seed {seed} must reproduce the same draw");
        }
    }

    #[test]
    fn sample_categorical_distribution_is_roughly_uniform() {
        // Over a seed corpus with a flat target, splitmix64's avalanche should
        // spread draws across all buckets. Tolerance is generous (max relative
        // deviation < 0.30 over N=2000, V=8; observed ~0.11) so the test tracks
        // the *property* of good mixing, not the specific mixing function.
        const N: u64 = 2000;
        const V: usize = 8;
        let mut counts = [0u64; V];
        for seed in 0..N {
            counts[sample_uniform(seed, V) as usize] += 1;
        }
        let expected = N as f64 / V as f64;
        for (bucket, &c) in counts.iter().enumerate() {
            let rel_dev = (c as f64 - expected).abs() / expected;
            assert!(
                rel_dev < 0.30,
                "bucket {bucket} count {c} deviates {rel_dev:.3} (>0.30) from uniform {expected}"
            );
        }
    }

    #[test]
    fn adjacent_low_bit_seeds_decorrelate() {
        // splitmix64 avalanche means seeds differing only in low bits map to
        // well-separated draws. Over consecutive odd seeds (the minimal distinct
        // step, since `seed_from` ORs in the low bit) draws differ far more often
        // than a mixer that preserved low-bit correlation would allow.
        const V: usize = 8;
        const PAIRS: u64 = 2000;
        let mut distinct = 0u64;
        for k in (1..(1 + 2 * PAIRS)).step_by(2) {
            if sample_uniform(k, V) != sample_uniform(k + 2, V) {
                distinct += 1;
            }
        }
        let frac = distinct as f64 / PAIRS as f64;
        // A perfect uniform mapping over V=8 gives ~7/8=0.875 distinct; require
        // well above 0.6 (observed ~0.87). A degenerate low-bit-preserving mixer
        // would collapse neighbors to identical buckets, scoring near 0.
        assert!(
            frac > 0.6,
            "adjacent-seed distinct fraction {frac:.3} too low; low bits not avalanched"
        );
    }
}
