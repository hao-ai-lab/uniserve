//! Official System One answers assembled from readout log-probabilities.
//!
//! Each question's candidate probabilities are first combined per answer key
//! and then normalized once over its keys:
//!
//! - a single-slot key sums the probabilities of its spellings (`" A"` and
//!   `"A"`; the seven yes or no spellings);
//! - a two-letter key `X Y` is `P(X) * P(Y | X)`.
//!
//! Probabilities are the raw full-vocabulary ones, never renormalized within
//! a subset before this combination. Their sum over the keys is the
//! candidate mass, reported as `x_candidate_mass`; a mass near zero means the
//! model did not answer within the candidates.
//!
//! A choice or score answer's confidence is `clamp((n·p_max − 1)/(n − 1), 0, 1)`
//! for `n > 1` keys and `1.0` for a single key.

use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::Value;

use crate::serving::systemone::error::SystemOneError;
use crate::serving::systemone::plan::{AnswerKind, Distribution, ReadoutPlan};

/// Smallest candidate mass normalized by, as in the reference readout; a
/// smaller mass yields all-zero probabilities.
const MIN_CANDIDATE_MASS: f64 = 1e-30;

/// The official `POST /v1/systemone` response.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SystemOneResponse {
    /// Served model name.
    pub model: String,
    /// Answers keyed by question id, in request order.
    pub answers: OrderedMap<Answer>,
    /// Token usage.
    pub usage: Usage,
}

/// Token usage of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Usage {
    /// Unique prompt tokens encoded for the request, image soft tokens
    /// included and the shared prefix counted once.
    pub input_tokens: u32,
    /// Always zero: a readout generates no tokens.
    pub output_tokens: u32,
}

/// One answer, tagged by its question type.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    /// Yes/no answer.
    Noul {
        /// Normalized probability of yes.
        noul: f64,
        /// Unnormalized probability of all yes and no spellings.
        x_candidate_mass: f64,
    },
    /// Choice answer.
    Choice {
        /// The most probable option; the first on ties.
        choice: String,
        /// Normalized probability of each option, in option order.
        probabilities: OrderedMap<f64>,
        /// Concentration of the distribution, from 0 to 1.
        confidence: f64,
        /// Unnormalized probability of all options.
        x_candidate_mass: f64,
    },
    /// Score answer.
    Score {
        /// Expected level, `Σ i·p_i`.
        score: f64,
        /// Each level (`"0"` to `"n-1"`) echoed from the request criteria.
        legend: OrderedMap<Value>,
        /// Normalized probability of each level, keyed as the legend.
        probabilities: OrderedMap<f64>,
        /// Concentration of the distribution, from 0 to 1.
        confidence: f64,
        /// Unnormalized probability of all levels.
        x_candidate_mass: f64,
    },
}

/// A JSON object whose keys keep their insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedMap<V>(pub Vec<(String, V)>);

impl<V: Serialize> Serialize for OrderedMap<V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl ReadoutPlan {
    /// Assembles the response from the plan's candidate log-probabilities.
    ///
    /// `logprobs` holds, in plan order (prompt, row, slot, candidate), the
    /// natural-log probability of every slot candidate under the log-softmax
    /// over the full vocabulary of the softcapped logits;
    /// [`ReadoutPlan::candidate_count`] gives its length. `model` is the
    /// served model name.
    ///
    /// # Errors
    ///
    /// Returns [`SystemOneError::Server`] when `logprobs` has the wrong length
    /// or contains NaN.
    pub fn assemble(
        &self,
        model: &str,
        logprobs: &[f32],
    ) -> Result<SystemOneResponse, SystemOneError> {
        if logprobs.len() != self.candidate_count {
            return Err(SystemOneError::Server(format!(
                "readout returned {} candidate log-probabilities for {} candidates",
                logprobs.len(),
                self.candidate_count
            )));
        }
        if logprobs.iter().any(|logprob| logprob.is_nan()) {
            return Err(SystemOneError::Server(
                "readout returned a NaN log-probability".to_owned(),
            ));
        }
        let probability = |index: usize| f64::from(logprobs[index]).exp();

        let answers = self
            .readouts
            .iter()
            .map(|readout| {
                let per_key: Vec<f64> = match &readout.distribution {
                    Distribution::Summed(keys) => keys
                        .iter()
                        .map(|spellings| spellings.clone().map(probability).sum())
                        .collect(),
                    Distribution::Chained(pairs) => pairs
                        .iter()
                        .map(|&(first, second)| probability(first) * probability(second))
                        .collect(),
                };
                let mass: f64 = per_key.iter().sum();
                let probabilities: Vec<f64> = per_key
                    .iter()
                    .map(|key| key / mass.max(MIN_CANDIDATE_MASS))
                    .collect();
                (
                    readout.id.clone(),
                    answer(&readout.kind, probabilities, mass),
                )
            })
            .collect();

        Ok(SystemOneResponse {
            model: model.to_owned(),
            answers: OrderedMap(answers),
            usage: Usage {
                input_tokens: self.input_tokens,
                output_tokens: 0,
            },
        })
    }
}

/// Builds one answer from its normalized key probabilities.
fn answer(kind: &AnswerKind, probabilities: Vec<f64>, mass: f64) -> Answer {
    match kind {
        // Keys are `false` then `true`.
        AnswerKind::Noul => Answer::Noul {
            noul: probabilities[1],
            x_candidate_mass: mass,
        },
        AnswerKind::Choice(names) => Answer::Choice {
            choice: names[argmax(&probabilities)].clone(),
            confidence: confidence(&probabilities),
            probabilities: OrderedMap(names.iter().cloned().zip(probabilities).collect()),
            x_candidate_mass: mass,
        },
        AnswerKind::Score(levels) => Answer::Score {
            score: probabilities
                .iter()
                .enumerate()
                .map(|(level, probability)| level as f64 * probability)
                .sum(),
            legend: OrderedMap(
                levels
                    .iter()
                    .enumerate()
                    .map(|(level, criterion)| (level.to_string(), criterion.clone()))
                    .collect(),
            ),
            confidence: confidence(&probabilities),
            probabilities: OrderedMap(
                probabilities
                    .into_iter()
                    .enumerate()
                    .map(|(level, probability)| (level.to_string(), probability))
                    .collect(),
            ),
            x_candidate_mass: mass,
        },
    }
}

/// Index of the most probable key; the first on ties.
fn argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, probability)| {
            if *probability > probabilities[best] {
                index
            } else {
                best
            }
        })
}

/// `clamp((n·p_max − 1)/(n − 1), 0, 1)`, and `1.0` for a single key.
fn confidence(probabilities: &[f64]) -> f64 {
    let keys = probabilities.len() as f64;
    if probabilities.len() <= 1 {
        return 1.0;
    }
    let most = probabilities.iter().copied().fold(0.0, f64::max);
    ((keys * most - 1.0) / (keys - 1.0)).clamp(0.0, 1.0)
}
