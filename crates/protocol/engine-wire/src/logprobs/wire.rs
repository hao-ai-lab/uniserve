use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::tensor::WireNdArray;

/// Python wire representation of `LogprobsLists` / `LogprobsTensors` before
/// aux-frame references and raw-view payloads are resolved.

/// This mirrors the tuple shape emitted by the Python engine so serde can
/// first deserialize the raw wire payload before the Rust client converts it
/// into semantic per-position logprobs records.

#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireLogprobs {
 /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprob_token_ids: WireNdArray,
 /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprobs: WireNdArray,
 /// Wire array with shape `[num_positions]`.

 /// Python uses the field name `sampled_token_ranks` for sample logprobs and
 /// `selected_token_ranks` for prompt logprobs. Rust keeps one neutral field
 /// because both payloads share the same wire representation.
    pub token_ranks: WireNdArray,
 /// Preserved only for wire compatibility with batch-level Python tensors.
 /// Scheduler-sliced per-request outputs should emit `None` here, and
 /// the semantic Rust decoder rejects any other value.
    #[serde(default)]
    pub cu_num_generated_tokens: Option<Vec<usize>>,
}
