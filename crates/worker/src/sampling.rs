//! Host controls for numerical token sampling and speculative verification.

use std::collections::BTreeSet;

use uniserve_core::SamplingParams;
use uniserve_core::philox::{DRAW_LAYOUT_TARGET, sampling_key, sampling_uniform};
use uniserve_worker_ipc::{Call, DrawLayout};

use crate::{Error, Result};

/// Token relays carry continuation in bit 31 and the token in the low 31 bits.
pub const TOKEN_CONTINUATION_BIT: u32 = 1 << 31;
pub const TOKEN_VALUE_MASK: u32 = TOKEN_CONTINUATION_BIT - 1;

/// Numerical selectors supported by the worker's batched sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplingPath {
    Greedy { suppressed: bool },
    TopK(u32),
    Categorical,
}

/// Per-call constraints aligned with candidate logit rows. Tensor values and
/// penalties stay with the numerical backend; no device value is read here.
#[derive(Clone, Debug, Default)]
pub struct SamplingMetadata {
    pub allowed: Vec<Option<Vec<u32>>>,
    pub suppress: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub transition_token_ids: Vec<u32>,
    pub force_finish: bool,
    pub draft_token_ids: Vec<u32>,
    /// One-based prefix length ending at the first terminal draft token.
    pub terminal_draft_prefix: Option<usize>,
    pub return_transition: bool,
}

impl SamplingMetadata {
    /// Resolve admitted and branch-local controls. Draw coordinates describe
    /// the candidate rows, independent of scheduling order or draft acceptance.
    /// The returned draws are absent only for the unshaped greedy path.
    pub fn for_call(
        call: &Call,
        parameters: &SamplingParams,
        finish_tokens: &[u32],
        positions: &[u64],
        drafts: Vec<u32>,
    ) -> Result<(Self, Option<Vec<f32>>)> {
        let state = call.sampling_state.as_ref();
        let allowed = state
            .and_then(|state| state.allowed_token_ids.as_ref())
            .or(parameters.allowed_token_ids.as_ref());
        let mut result = Self {
            allowed: positions
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    parameters
                        .forced_token_ids
                        .get(index)
                        .map_or_else(|| allowed.cloned(), |token| Some(vec![*token]))
                })
                .collect(),
            suppress: state.map_or_else(Vec::new, |state| state.suppressed_token_ids.clone()),
            finish_token_ids: finish_token_ids(call, finish_tokens),
            transition_token_ids: state
                .map_or_else(Vec::new, |state| state.transition_token_ids.clone()),
            force_finish: state.is_some_and(|state| state.force_finish),
            draft_token_ids: drafts,
            ..Self::default()
        };
        result.terminal_draft_prefix = result
            .draft_token_ids
            .iter()
            .position(|token| result.finish_token_ids.contains(token))
            .map(|index| index + 1);

        let draws = if parameters.temperature > 0.0 {
            let rng = call
                .rng
                .as_ref()
                .filter(|rng| rng.draw_layout == DrawLayout::TargetSampling)
                .ok_or_else(|| {
                    Error::Invalid(
                        "stochastic sampling requires target-sampling RNG coordinates".into(),
                    )
                })?;
            if rng.seed != parameters.seed.unwrap_or(0) {
                return Err(Error::Invalid(
                    "call RNG seed disagrees with admitted sampling".into(),
                ));
            }
            if positions.iter().enumerate().any(|(index, position)| {
                rng.semantic_index_base.checked_add(index as u64) != Some(*position)
            }) {
                return Err(Error::Invalid(
                    "sampling positions disagree with registered semantic RNG coordinates".into(),
                ));
            }

            let key = sampling_key(
                rng.seed,
                call.request_key.engine_id,
                call.request_key.request_id.0,
                call.request_key.request_epoch,
                DRAW_LAYOUT_TARGET,
            );
            Some(
                positions
                    .iter()
                    .map(|position| sampling_uniform(key, *position, 0, 0))
                    .collect(),
            )
        } else if result.device_greedy(parameters) {
            None
        } else {
            Some(vec![0.0; positions.len()])
        };
        Ok((result, draws))
    }

    pub fn device_greedy(&self, parameters: &SamplingParams) -> bool {
        self.draft_token_ids.is_empty()
            && parameters.device_greedy()
            && self.allowed.iter().all(Option::is_none)
    }

    /// Select a numerical implementation without inspecting logits on device.
    pub fn path(&self, parameters: &SamplingParams, vocab: usize, cuda: bool) -> SamplingPath {
        if self.device_greedy(parameters) {
            return SamplingPath::Greedy {
                suppressed: !self.suppress.is_empty(),
            };
        }

        // The compiled top-k kernel accepts at most 128 candidates and no
        // branch masks or penalties. All other controls use the general path.
        let top_k = parameters.top_k;
        if cuda
            && self.draft_token_ids.is_empty()
            && !parameters.uses_penalties()
            && self.allowed.iter().all(Option::is_none)
            && self.suppress.is_empty()
            && parameters.logit_bias.is_empty()
            && parameters.typical_p == 1.0
            && (1..=128).contains(&top_k)
            && (top_k as usize) < vocab
        {
            SamplingPath::TopK(top_k)
        } else {
            SamplingPath::Categorical
        }
    }
}

/// Merge admitted stop tokens with the current branch's terminal tokens.
pub fn finish_token_ids(call: &Call, admitted: &[u32]) -> Vec<u32> {
    let local = call
        .sampling_state
        .as_ref()
        .map_or(&[][..], |state| state.finish_token_ids.as_slice());
    if local.is_empty() {
        admitted.to_vec()
    } else if admitted.is_empty() {
        local.to_vec()
    } else {
        admitted
            .iter()
            .chain(local)
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}
