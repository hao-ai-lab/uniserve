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

use crate::SamplingParams;

#[derive(Debug, Clone, PartialEq)]
pub struct SampleOutput {
    pub token: u32,
    pub logprob: f32,
 /// Top `(token_id, logprob)` pairs (length == requested n_logprobs).
    pub top: Vec<(u32, f32)>,
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
    let v = logits.len();

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
 // counts of each recent token
        let mut counts: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for &t in recent {
            *counts.entry(t).or_insert(0.0) += 1.0;
        }
        for (&t, &c) in &counts {
            let i = t as usize;
            if i >= v || logits[i] == NEG_INF {
                continue;
            }
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

 // 9. sample.
    let token = if greedy {
        argmax(logits)
    } else {
        let probs = softmax(logits);
        sample_categorical(&probs, seed_from(p, recent))
    };

 // 10. gather logprobs (softmax of the final, masked logits).
    let logprobs = log_softmax(logits);
    let sampled_lp = logprobs.get(token as usize).copied().unwrap_or(NEG_INF);
    let mut top: Vec<(u32, f32)> = Vec::new();
    if n_logprobs > 0 {
        let mut order: Vec<usize> = (0..v).filter(|&i| logits[i] != NEG_INF).collect();
        order.sort_by(|&a, &b| {
            logprobs[b]
                .partial_cmp(&logprobs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in order.iter().take(n_logprobs) {
            top.push((i as u32, logprobs[i]));
        }
    }
    SampleOutput {
        token,
        logprob: sampled_lp,
        top,
    }
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
    fn frequency_penalty_demotes_recent() {
        let mut l = base_logits();
        let p = SamplingParams {
            frequency_penalty: 2.0,
            ..Default::default()
        };
        let out = apply_sampling(&mut l, &p, &[3, 3, 3], None, None, 0);
        assert_ne!(out.token, 3);
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
    fn defaults_are_noop_argmax() {
 // Every transform unset => plain argmax, deterministic.
        let mut l = base_logits();
        let out = apply_sampling(&mut l, &SamplingParams::default(), &[1, 2], None, None, 0);
        assert_eq!(out.token, 3);
    }

    #[test]
    fn categorical_respects_distribution() {
 // A heavily skewed distribution should overwhelmingly draw the dominant
 // outcome across a range of seeds.
        let probs = [0.9f32, 0.05, 0.05];
        let hits = (0..1000u64)
            .filter(|&s| sample_categorical(&probs, s | 1) == 0)
            .count();
        assert!(hits > 800, "dominant outcome under-sampled: {hits}/1000");
    }

    #[test]
    fn categorical_decorrelates_adjacent_seeds() {
 // Adjacent seeds (exactly what seed_from produces as the recent window
 // grows by one token) must not collapse to the same draw under a
 // uniform distribution — a single xorshift step failed this.
        let probs = [0.2f32; 5];
        let draws: Vec<u32> = (0..200u64).map(|s| sample_categorical(&probs, s)).collect();
        let distinct: std::collections::HashSet<u32> = draws.iter().copied().collect();
        assert_eq!(distinct.len(), 5, "all outcomes should be reachable");
 // No long run of identical consecutive draws (correlation symptom).
        let max_run = draws
            .windows(2)
            .fold((1usize, 1usize), |(cur, best), w| {
                let cur = if w[0] == w[1] { cur + 1 } else { 1 };
                (cur, best.max(cur))
            })
            .1;
        assert!(max_run < 10, "adjacent seeds correlated: run length {max_run}");
    }
}
