use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::tensor::WireTensor;

/// Multimodal feature payload accepted from higher-level frontend code.

pub type MmFeatures = Vec<MmFeatureSpec>;

/// Represents a single multimodal input with its processed data and metadata.

/// Used to track multimodal data through processing and caching. A request
/// containing multiple multimodal items will have one `MmFeatureSpec`
/// per item.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MmFeatureSpec {
 /// Represents multimodal data for this feature.

 /// Can be `None` if the item is cached, to skip IPC between API server
 /// and engine core processes.
    pub data: Option<MmKwargsItem>,

 /// The input modality, e.g., `"image"`, `"audio"`, `"video"`.
    pub modality: String,

 /// The hash for caching encoder outputs (with LoRA prefix if applicable).
    pub identifier: String,

 /// The location of the `modality` tokens corresponding to this item
 /// in the prompt, e.g., `PlaceholderRange(offset=2, length=336)`.
    pub mm_position: PlaceholderRange,

 /// The hash for caching processor outputs (without LoRA prefix).

 /// # Contract

 /// On this (chat / `MmFeatureSpec`) ingress path the value is the
 /// **content digest as a hex string** — concretely the Blake3 hex-digest of
 /// the decoded media bytes, produced upstream in the chat multimodal builder
 /// (see `crates/frontend/chat/src/multimodal.rs`) and threaded through from
 /// `llm_multimodal::ImageFrame::hash`.

 /// This is intentionally a *different representation* from the native
 /// (image-generation / `ForwardOp`) ingress path, where the encoder-cache
 /// key is a `u64` (`worker-wire::ForwardOp::mm_hash`, fed from
 /// `engine-wire::WireMmItem::hash`, which `native-api`'s builder computes as
 /// an fnv1a fold of the base64 bytes). The two ingress paths are disjoint —
 /// there is no `MmFeatures` -> native `WireMmItem` conversion — so a request
 /// only ever carries one of the two conventions, never both, and the
 /// representations are not interchangeable. Do **not** assume a value seen
 /// here is comparable to a worker-wire `mm_hash` `u64`.

 /// Unifying the two protocols onto a single width/representation requires a
 /// coordinated change to `worker-wire`, `engine-wire::native`, the
 /// `native-api` hash producer, the scheduler, and the Python worker's
 /// `mm_hash` parsing; it is tracked separately and deliberately not
 /// done piecemeal on only one side.
    #[serde(default)]
    pub mm_hash: Option<String>,
}

/// Placeholder location information for multi-modal data.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaceholderRange {
 /// The start index of the placeholder in the prompt.
    pub offset: usize,

 /// The length of the placeholder.
    pub length: usize,

 /// A boolean mask of shape `(length)` indicating which positions
 /// between `offset` and `offset + length` to assign embeddings to.
 /// `None` means all positions.
    #[serde(default)]
    pub is_embed: Option<WireTensor>,
}

/// A dictionary of processed keyword arguments to pass to the model,
/// corresponding to a single item in `MultiModalDataItems`.

pub type MmKwargsItem = BTreeMap<String, MmFieldElem>;

/// Represents a processed keyword argument to pass to a model for a
/// `MmKwargsItem`.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MmFieldElem {
 /// The processed value of this field in `MmKwargsItem`, i.e. the
 /// keyword argument value to be passed to the model.

 /// It may be set to `None` if it is determined that the item is cached
 /// in `EngineCore`.
    pub data: Option<MmKwargValue>,

 /// Defines how to combine this field's processed values with others in
 /// order to batch multi-modal items together for model inference.
    pub field: MmField,
}

/// Processed multimodal keyword argument value.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MmKwargValue {
    Tensor(WireTensor),
    Int(i64),
    Float(f64),
    List(Vec<MmKwargValue>),
}

/// Defines how to interpret tensor data belonging to a keyword argument for
/// `MultiModalKwargsItems`, and vice versa.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MmFieldWire", into = "MmFieldWire")]
pub enum MmField {
    Batched(MmBatchedField),
    Flat(MmFlatField),
    Shared(MmSharedField),
}

/// Info: `MultiModalFieldConfig.batched`.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmBatchedField {
 /// If `True`, then this field is excluded from being moved to the
 /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Info: `MultiModalFieldConfig.flat` and
/// `MultiModalFieldConfig.flat_from_sizes`.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmFlatField {
 /// For each multi-modal item, a slice (`dim=0`) or a tuple of slices
 /// (`dim>0`) that is used to extract the data corresponding to it.
    pub slices: Vec<MmSlice>,

 /// The dimension to extract data, default to 0.
    pub dim: i32,

 /// If `True`, then this field is excluded from being moved to the
 /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Info: `MultiModalFieldConfig.shared`.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmSharedField {
    pub batch_size: usize,

 /// If `True`, then this field is excluded from being moved to the
 /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Python slice encoded as `(start, stop, step)`.

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: Option<isize>,
}

/// A single slice or a tuple of slices used by `MmFlatField`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MmSlice {
    Slice(SliceSpec),
    Slices(Vec<SliceSpec>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
struct MmFieldWire {
    name: String,
    inner: MmFieldWireInner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum MmFieldWireInner {
    Batched(MmBatchedField),
    Flat(MmFlatField),
    Shared(MmSharedField),
}

impl TryFrom<MmFieldWire> for MmField {
    type Error = String;

    fn try_from(value: MmFieldWire) -> Result<Self, Self::Error> {
        match (value.name.as_str(), value.inner) {
            ("batched", MmFieldWireInner::Batched(kwargs)) => Ok(Self::Batched(kwargs)),
            ("flat", MmFieldWireInner::Flat(kwargs)) => Ok(Self::Flat(kwargs)),
            ("shared", MmFieldWireInner::Shared(kwargs)) => Ok(Self::Shared(kwargs)),
            (name, _) => Err(format!(
                "mismatched or unknown multimodal field factory {name:?}"
            )),
        }
    }
}

impl From<MmField> for MmFieldWire {
    fn from(value: MmField) -> Self {
        match value {
            MmField::Batched(kwargs) => Self {
                name: "batched".to_string(),
                inner: MmFieldWireInner::Batched(kwargs),
            },
            MmField::Flat(kwargs) => Self {
                name: "flat".to_string(),
                inner: MmFieldWireInner::Flat(kwargs),
            },
            MmField::Shared(kwargs) => Self {
                name: "shared".to_string(),
                inner: MmFieldWireInner::Shared(kwargs),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::{decode_msgpack, decode_value, encode_msgpack};

    fn encode_value<T: Serialize + std::fmt::Debug>(value: &T) -> Value {
        let bytes = encode_msgpack(value).expect("encode value");
        decode_value(&bytes).expect("decode value")
    }

    #[test]
    fn multimodal_field_serializes_to_python_factory_tuple() {
        let field = MmField::Flat(MmFlatField {
            slices: vec![MmSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(1200),
                step: None,
            })],
            dim: 0,
            keep_on_cpu: false,
        });

        let value = encode_value(&field);
        let Value::Array(items) = value else {
            panic!("field should encode as tuple array");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_str(), Some("flat"));

        let Value::Map(kwargs) = &items[1] else {
            panic!("field kwargs should encode as map");
        };
        assert!(kwargs.iter().any(|(key, _)| key.as_str() == Some("slices")));
        assert!(kwargs.iter().any(|(key, _)| key.as_str() == Some("dim")));
        assert!(
            kwargs
                .iter()
                .any(|(key, _)| key.as_str() == Some("keep_on_cpu"))
        );
    }

    #[test]
    fn multimodal_field_round_trips_python_factory_tuple() {
        let field = MmField::Batched(MmBatchedField { keep_on_cpu: true });
        let encoded = encode_msgpack(&field).expect("encode field");
        let decoded: MmField = decode_msgpack(&encoded).expect("decode field");
        assert_eq!(decoded, field);
    }
}
