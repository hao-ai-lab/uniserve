use super::*;

// ---------------------------------------------------------------------------
// Product references and bounded shapes
// ---------------------------------------------------------------------------

/// The role a product plays for its consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ProductKind {
    Token = 0,
    Logprob = 1,
    VisionFeature = 2,
    LatentFeature = 3,
    Kv = 4,
    Latent = 5,
    Artifact = 6,
    Completion = 7,
    SamplingState = 8,
    SelectedPoint = 9,
}

/// The worker store family that backs a product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum StorageClass {
    DeviceTensor = 0,
    RequestRelay = 1,
    PagedKv = 2,
    LatentArena = 3,
    HostStaging = 4,
    PinnedOutput = 5,
}

/// Element type of a product's backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DType {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    #[serde(rename = "bf16")]
    BF16 = 6,
    F32 = 7,
}

/// One dimension of a bounded shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DimBound {
    /// A host-static extent.
    Static(u32),
    /// The single device-actual axis, bounded by this fixed maximum.
    Device { max: u32 },
}

/// A shape that is host-static except for at most one device-actual axis, which
/// carries a fixed maximum. A product reference never carries an unbounded shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ShapeBound {
    pub dims: Vec<DimBound>,
}

impl ShapeBound {
    pub fn validate(&self) -> ValidationResult<()> {
        let device_dims = self
            .dims
            .iter()
            .filter(|dim| matches!(dim, DimBound::Device { .. }))
            .count();
        ensure_valid!(
            device_dims <= 1,
            "a shape bound carries more than one device-actual dimension"
        );
        ensure_valid!(
            self.dims.iter().all(|dim| match dim {
                DimBound::Static(value) => *value > 0,
                DimBound::Device { max } => *max > 0,
            }),
            "a shape bound contains a zero extent"
        );
        Ok(())
    }

    pub(crate) fn max_elements(&self) -> u64 {
        self.dims.iter().fold(1_u64, |elements, dim| {
            elements.saturating_mul(u64::from(match dim {
                DimBound::Static(value) => *value,
                DimBound::Device { max } => *max,
            }))
        })
    }
}

/// The state points a product spans, rooted at `base_point`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct PointRange {
    pub base_point: u32,
    pub max_points: u32,
}

/// A generation-tagged reference to a declared device or host product. Physical
/// slots, tensors, and events stay worker-local and never appear here.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductRef {
    pub request_key: RequestKey,
    pub producer_op_id: OpId,
    pub output_index: u16,
    pub generation: u32,
    pub kind: ProductKind,
    pub storage_class: StorageClass,
    pub dtype: DType,
    pub shape_bound: ShapeBound,
    pub point_range: PointRange,
}

impl ProductRef {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.generation > 0,
            "product reference has no logical generation"
        );
        self.shape_bound.validate()
    }

    pub fn max_bytes(&self) -> u64 {
        let element_bytes = match self.dtype {
            DType::U8 => 1,
            DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::I64 => 8,
        };
        self.shape_bound
            .max_elements()
            .saturating_mul(element_bytes)
    }
}

/// Envelope metadata reporting whether an atomic registration became visible. It
/// carries no semantic lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RegistrationAck {
    pub visible: bool,
}

/// A resolved product value carried across the boundary: a host-supplied input
/// value the worker consumes (prompt, forced, or draft token ids; encode image
/// bytes) referenced through `Operation::inputs`, or a worker-produced output
/// value the host consumes (requested logprob blobs, materialized image bytes).
/// `product` says what the value is; `bytes` carries the value.
///
/// The IPC decoder treats other product bytes as opaque. This crate fixes
/// the cross-language layouts for token inputs and branch-local sampling state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductPayload {
    pub product: ProductRef,
    /// Raw value bytes. `serde_bytes` keeps the pyo3 boundary on the
    /// bytes fast path (one buffer copy) instead of a per-element
    /// integer-sequence walk, which costs hundreds of milliseconds for a
    /// multi-megabyte image artifact.
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

impl ProductPayload {
    pub fn validate(&self) -> ValidationResult<()> {
        self.product.validate()
    }

    pub(crate) fn validate_input_value(&self) -> ValidationResult<()> {
        if is_transfer_descriptor(&self.bytes) {
            ensure_valid!(
                self.product.storage_class != StorageClass::HostStaging
                    && self.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                "cross-stage product input has an invalid transfer descriptor frame"
            );
            return Ok(());
        }
        match self.product.kind {
            ProductKind::Token => {
                let tokens = decode_token_product_bytes(&self.bytes)?;
                ensure_valid!(
                    tokens.len() as u64 <= self.product.shape_bound.max_elements(),
                    "token input product exceeds its registered element bound"
                );
            }
            ProductKind::SamplingState => {
                decode_sampling_state_bytes(&self.bytes)?;
                ensure_valid!(
                    self.bytes.len() as u64 <= self.product.max_bytes(),
                    "sampling-state input exceeds its registered byte bound"
                );
            }
            _ => ensure_valid!(
                self.bytes.len() as u64 <= self.product.max_bytes(),
                "input product payload exceeds its registered byte bound"
            ),
        }
        Ok(())
    }

    pub(crate) fn validate_output_value(&self) -> ValidationResult<()> {
        self.validate()?;
        if is_transfer_descriptor(&self.bytes) {
            ensure_valid!(
                self.product.storage_class != StorageClass::HostStaging
                    && self.product.storage_class != StorageClass::PinnedOutput
                    && self.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                "output transfer descriptor has an invalid storage class or byte bound"
            );
            return Ok(());
        }
        ensure_valid!(
            self.bytes.len() as u64 <= self.product.max_bytes(),
            "output product payload exceeds its registered byte bound"
        );
        Ok(())
    }
}

pub const TRANSFER_DESCRIPTOR_PREFIX: &[u8] = b"uniserve-transfer\0";
pub const MAX_TRANSFER_DESCRIPTOR_BYTES: usize = 64 * 1024;

pub fn is_transfer_descriptor(bytes: &[u8]) -> bool {
    bytes.starts_with(TRANSFER_DESCRIPTOR_PREFIX)
}

/// Encode a `ProductKind::Token` product value: a little-endian `u32` count
/// followed by that many little-endian `u32` token ids.
pub fn encode_token_product_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + tokens.len() * 4);
    bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

/// Decode a `ProductKind::Token` product value produced by
/// [`encode_token_product_bytes`].
pub fn decode_token_product_bytes(bytes: &[u8]) -> ValidationResult<Vec<u32>> {
    ensure_valid!(
        bytes.len() >= 4,
        "token product bytes are too short to carry a count"
    );
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected = 4 + count * 4;
    ensure_valid!(
        bytes.len() == expected,
        "token product byte length {} does not match declared count {count}",
        bytes.len()
    );
    Ok(bytes[4..]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

/// Branch-local token processor inputs for one sampling operation.
///
/// Token ids in every field are strictly increasing. `allowed_token_ids`
/// distinguishes no whitelist (`None`) from a present empty whitelist, which
/// deterministically represents an invalid all-masked distribution. Penalty
/// token counts are not carried here: they are a device-resident committed base
/// plus bounded per-operation deltas the worker folds on commit, so no host
/// token history participates in a successor's sampling input.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SamplingState {
    pub allowed_token_ids: Option<Vec<u32>>,
    pub suppressed_token_ids: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub transition_token_ids: Vec<u32>,
    pub force_finish: bool,
}

impl SamplingState {
    pub fn canonicalize(&mut self) {
        if let Some(allowed) = &mut self.allowed_token_ids {
            allowed.sort_unstable();
            allowed.dedup();
        }
        self.suppressed_token_ids.sort_unstable();
        self.suppressed_token_ids.dedup();
        self.finish_token_ids.sort_unstable();
        self.finish_token_ids.dedup();
        self.transition_token_ids.sort_unstable();
        self.transition_token_ids.dedup();
    }
}

/// Encode canonical branch-local sampling state.
///
/// Layout: one allowed-presence byte; an allowed count and ids when present;
/// then a suppressed count and ids; a finish count and ids; a transition count
/// and ids; then one force-finish byte.
pub fn encode_sampling_state_bytes(state: &SamplingState) -> Vec<u8> {
    let mut canonical = state.clone();
    canonical.canonicalize();
    let allowed_len = canonical.allowed_token_ids.as_ref().map_or(0, Vec::len);
    let mut bytes = Vec::with_capacity(
        1 + allowed_len * 4
            + 4
            + canonical.suppressed_token_ids.len() * 4
            + 4
            + canonical.finish_token_ids.len() * 4
            + 4
            + canonical.transition_token_ids.len() * 4
            + 1,
    );
    match canonical.allowed_token_ids {
        Some(allowed) => {
            bytes.push(1);
            bytes.extend_from_slice(&(allowed.len() as u32).to_le_bytes());
            for token in allowed {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(canonical.suppressed_token_ids.len() as u32).to_le_bytes());
    for token in canonical.suppressed_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(canonical.finish_token_ids.len() as u32).to_le_bytes());
    for token in canonical.finish_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(canonical.transition_token_ids.len() as u32).to_le_bytes());
    for token in canonical.transition_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.push(u8::from(canonical.force_finish));
    bytes
}

/// Decode and validate canonical branch-local sampling state.
pub fn decode_sampling_state_bytes(bytes: &[u8]) -> ValidationResult<SamplingState> {
    fn take_u32(bytes: &[u8], offset: &mut usize) -> ValidationResult<u32> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| invalid_message!("sampling-state offset overflow"))?;
        ensure_valid!(end <= bytes.len(), "sampling-state bytes are truncated");
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        Ok(value)
    }
    fn take_ids(bytes: &[u8], offset: &mut usize, count: u32) -> ValidationResult<Vec<u32>> {
        let mut values = Vec::with_capacity(count as usize);
        for _ in 0..count {
            values.push(take_u32(bytes, offset)?);
        }
        ensure_valid!(
            values.windows(2).all(|pair| pair[0] < pair[1]),
            "sampling-state token ids are not canonical"
        );
        Ok(values)
    }

    let mut offset = 0;
    ensure_valid!(
        offset < bytes.len(),
        "sampling-state bytes omit allowed presence"
    );
    let allowed_token_ids = match bytes[offset] {
        0 => {
            offset += 1;
            None
        }
        1 => {
            offset += 1;
            let count = take_u32(bytes, &mut offset)?;
            Some(take_ids(bytes, &mut offset, count)?)
        }
        other => bail_invalid!("sampling-state allowed presence {other} is invalid"),
    };
    let suppressed_len = take_u32(bytes, &mut offset)?;
    let suppressed_token_ids = take_ids(bytes, &mut offset, suppressed_len)?;
    let finish_len = take_u32(bytes, &mut offset)?;
    let finish_token_ids = take_ids(bytes, &mut offset, finish_len)?;
    let transition_len = take_u32(bytes, &mut offset)?;
    let transition_token_ids = take_ids(bytes, &mut offset, transition_len)?;
    ensure_valid!(
        offset < bytes.len(),
        "sampling-state bytes omit force-finish"
    );
    let force_finish = match bytes[offset] {
        0 => false,
        1 => true,
        other => bail_invalid!("sampling-state force-finish {other} is invalid"),
    };
    offset += 1;
    ensure_valid!(
        offset == bytes.len(),
        "sampling-state bytes contain trailing data"
    );
    Ok(SamplingState {
        allowed_token_ids,
        suppressed_token_ids,
        finish_token_ids,
        transition_token_ids,
        force_finish,
    })
}
