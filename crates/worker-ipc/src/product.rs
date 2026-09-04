//! Product identities, storage descriptions, transfers, and inline encodings.

use super::*;

/// The role a product plays for its consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ProductKind {
    /// Generated or input token identifiers.
    Token = 0,
    /// Generated or prompt token log probabilities.
    Logprob = 1,
    /// Vision encoder features.
    VisionFeature = 2,
    /// VAE encoder features.
    LatentFeature = 3,
    /// Paged key/value cache state.
    Kv = 4,
    /// Diffusion trajectory state.
    Latent = 5,
    /// Materialized media or feedback artifact.
    Artifact = 6,
    /// Operation completion marker.
    Completion = 7,
    /// Encoded branch-local sampling controls.
    SamplingState = 8,
    /// Device-selected checkpoint point.
    SelectedPoint = 9,
}

/// The worker store family that backs a product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum StorageClass {
    /// Device tensor with operation-scoped ownership.
    DeviceTensor = 0,
    /// Small value relayed between operations in one request.
    RequestRelay = 1,
    /// Page-addressed key/value cache storage.
    PagedKv = 2,
    /// Page-addressed diffusion latent storage.
    LatentArena = 3,
    /// Host-resident operation input or intermediate output.
    HostStaging = 4,
    /// Host-pinned caller-visible output.
    PinnedOutput = 5,
}

/// Element type of a product's backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8 = 0,
    /// Unsigned 16-bit integer.
    U16 = 1,
    /// Unsigned 32-bit integer.
    U32 = 2,
    /// Signed 32-bit integer.
    I32 = 3,
    /// Signed 64-bit integer.
    I64 = 4,
    /// IEEE 754 half precision.
    F16 = 5,
    /// Brain floating-point half precision.
    #[serde(rename = "bf16")]
    BF16 = 6,
    /// IEEE 754 single precision.
    F32 = 7,
}

/// One dimension of a bounded shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DimBound {
    /// A host-static extent.
    Static(u32),
    /// The single device-actual axis, bounded by this fixed maximum.
    Device {
        /// Maximum device-selected extent.
        max: u32,
    },
}

/// A shape that is host-static except for at most one device-actual axis, which
/// carries a fixed maximum. A product reference never carries an unbounded shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ShapeBound {
    /// Ordered tensor dimension bounds.
    pub dims: Vec<DimBound>,
}

impl ShapeBound {
    /// Validates rank and positive dimension bounds.
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

    /// Returns the maximum number of elements represented by this shape.
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
    /// First state point represented by the product.
    pub base_point: u32,
    /// Maximum number of consecutive represented points.
    pub max_points: u32,
}

/// A generation-tagged reference to a declared device or host product. Physical
/// slots, tensors, and events stay worker-local and never appear here.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductRef {
    /// Request lineage that owns the value.
    pub request_key: RequestKey,
    /// Operation that declares the value.
    pub producer_op_id: OpId,
    /// Position in the producer's output list.
    pub output_index: u16,
    /// Nonzero allocation generation preventing identity reuse.
    pub generation: u32,
    /// Semantic role of the value.
    pub kind: ProductKind,
    /// Storage family backing the value.
    pub storage_class: StorageClass,
    /// Element data type.
    pub dtype: DType,
    /// Maximum logical tensor shape.
    pub shape_bound: ShapeBound,
    /// State points represented by the value.
    pub point_range: PointRange,
}

/// Stable identity for one cross-operation buffer, independent of its physical representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId {
    /// Request lineage that owns the buffer.
    pub owner: RequestKey,
    /// Operation that first declares the buffer.
    pub producer_op_id: OpId,
    /// Position in the producer's output list.
    pub output_index: u16,
    /// Nonzero allocation generation preventing identity reuse.
    pub generation: u32,
}

impl BufferId {
    /// Validates a nonzero persistent-buffer identity.
    pub fn validate(self) -> ValidationResult<()> {
        ensure_valid!(self.generation > 0, "buffer id has no logical generation");
        Ok(())
    }
}

impl ProductRef {
    /// Returns the persistent-buffer identity for buffer-backed products.
    pub const fn buffer_id(&self) -> BufferId {
        BufferId {
            owner: self.request_key,
            producer_op_id: self.producer_op_id,
            output_index: self.output_index,
            generation: self.generation,
        }
    }

    /// Validates product identity, shape, storage, and point bounds.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.generation > 0,
            "product reference has no logical generation"
        );
        self.shape_bound.validate()
    }

    /// Returns the maximum encoded byte size allowed by the shape and dtype.
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

    /// Returns whether this value needs an address-stable shared allocation.
    ///
    /// KV and diffusion trajectories use dedicated page placements;
    /// request-relay scalars and host results do not consume this pool.
    pub const fn uses_persistent_buffer(&self) -> bool {
        matches!(
            self.kind,
            ProductKind::VisionFeature | ProductKind::LatentFeature
        ) || matches!(
            (self.kind, self.storage_class),
            (
                ProductKind::Artifact,
                StorageClass::DeviceTensor | StorageClass::LatentArena
            )
        )
    }
}

/// Envelope metadata reporting whether an atomic registration became visible. It
/// carries no semantic lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RegistrationAck {
    /// Whether the submitted registration is visible to subsequent operations.
    pub visible: bool,
}

/// Transport used to publish a product between worker pools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "transport", content = "value")]
pub enum TransferTransport {
    /// Process-local publication resolved through a worker endpoint.
    Local {
        /// Publishing endpoint identity.
        endpoint: String,
        /// Endpoint-local publication key.
        key: u64,
    },
    /// POSIX shared-memory publication with optional asynchronous readiness.
    PosixShm {
        /// Shared-memory object name.
        name: String,
        /// Bytes reserved for the readiness header.
        ready_header_bytes: u32,
        /// Semaphore name used when readiness is asynchronous.
        ready_semaphore: Option<String>,
    },
    /// CUDA IPC publication with shared lifetime and readiness handles.
    CudaIpc {
        /// Publishing worker endpoint.
        endpoint: String,
        /// Stable publication identity.
        publication_id: String,
        /// Opaque CUDA allocation handle.
        #[serde(with = "serde_bytes")]
        storage_handle: Vec<u8>,
        /// Exported allocation size in bytes.
        storage_size_bytes: u64,
        /// Byte offset of the exported storage region.
        storage_offset_bytes: u64,
        /// Element offset of the tensor view.
        tensor_offset: u64,
        /// Tensor stride in elements.
        tensor_stride: Vec<i64>,
        /// Opaque CUDA handle for shared reference-count storage.
        #[serde(with = "serde_bytes")]
        ref_counter_handle: Vec<u8>,
        /// Byte offset of the shared reference counter.
        ref_counter_offset: u64,
        /// Opaque CUDA completion-event handle.
        #[serde(with = "serde_bytes")]
        event_handle: Vec<u8>,
        /// Whether consumers must synchronize on the completion event.
        event_sync_required: bool,
        /// Opaque CUDA event handle signaling publication readiness.
        #[serde(with = "serde_bytes")]
        ready_event_handle: Vec<u8>,
    },
}

/// One transport-native reference with explicit tensor bounds. Transport handles
/// are opaque bytes only where the underlying CUDA API defines an opaque handle;
/// semantic transfer metadata remains typed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferLocator {
    /// Transport-specific publication descriptor.
    pub transport: TransferTransport,
    /// Tensor payload size in bytes.
    pub nbytes: u64,
    /// Stable tensor data-type name.
    pub dtype: String,
    /// Tensor extents in logical order.
    pub shape: Vec<u64>,
    /// Device containing the published tensor.
    pub device: String,
}

/// Kind of product carried by a cross-pool transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferKind {
    /// Encoder feature transfer.
    Encoder,
    /// Device-resident artifact transfer.
    DeviceProduct,
    /// Paged KV state transfer.
    Kv,
    /// Diffusion latent transfer.
    Latent,
}

/// Closed cross-pool transfer algebra.
///
/// Each variant carries the metadata required to install its product family on
/// a destination worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum TransferHandle {
    /// Encoded vision or latent features.
    Encoder {
        /// Producer allocation generation.
        generation: u32,
        /// Source image height in pixels.
        height: u32,
        /// Source image width in pixels.
        width: u32,
        /// Vision or latent feature product kind.
        payload_kind: ProductKind,
        /// Published feature tensor.
        locator: TransferLocator,
    },
    /// Device-resident artifact used by another model stage.
    DeviceProduct {
        /// Producer allocation generation.
        generation: u32,
        /// Artifact height in pixels, or zero for non-image tensors.
        height: u32,
        /// Artifact width in pixels, or zero for non-image tensors.
        width: u32,
        /// Semantic numeric range of the tensor values.
        value_range: String,
        /// Published artifact tensor.
        locator: TransferLocator,
    },
    /// One or more tensors representing paged KV state.
    Kv {
        /// Producer allocation generation.
        generation: u32,
        /// Published KV tensors.
        locators: Vec<TransferLocator>,
        /// Source request checkpoint represented by the publication.
        source: Checkpoint,
        /// Destination worker or pool identity.
        destination: String,
        /// Optional checkpoint already installed at the destination.
        base: Option<Checkpoint>,
        /// KV token extent represented by `base`.
        base_extent: u32,
        /// Total KV token extent represented by this publication.
        published_extent: u32,
        /// KV cache group identity.
        group_id: u32,
        /// Quantization or scale metadata identity.
        scale_identity: String,
    },
    /// Diffusion trajectory tensor.
    Latent {
        /// Producer allocation generation.
        generation: u32,
        /// Output image height in pixels.
        height: u32,
        /// Output image width in pixels.
        width: u32,
        /// Logical latent allocation units.
        latent_units: u32,
        /// Denoising step represented by the tensor.
        step: u32,
        /// Published latent tensor.
        locator: TransferLocator,
    },
}

impl TransferHandle {
    /// Returns the product family carried by this transfer.
    pub const fn kind(&self) -> TransferKind {
        match self {
            Self::Encoder { .. } => TransferKind::Encoder,
            Self::DeviceProduct { .. } => TransferKind::DeviceProduct,
            Self::Kv { .. } => TransferKind::Kv,
            Self::Latent { .. } => TransferKind::Latent,
        }
    }

    /// Returns the producer generation carried by this transfer.
    pub const fn generation(&self) -> u32 {
        match self {
            Self::Encoder { generation, .. }
            | Self::DeviceProduct { generation, .. }
            | Self::Kv { generation, .. }
            | Self::Latent { generation, .. } => *generation,
        }
    }

    /// Returns the ordered physical locators carried by this transfer.
    fn locators(&self) -> &[TransferLocator] {
        match self {
            Self::Encoder { locator, .. }
            | Self::DeviceProduct { locator, .. }
            | Self::Latent { locator, .. } => std::slice::from_ref(locator),
            Self::Kv { locators, .. } => locators,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
/// Materialized product data carried inline or referenced through a transfer handle.
pub enum InlineValue {
    /// Host-resident bytes embedded in the IPC message.
    Bytes(#[serde(with = "serde_bytes")] Vec<u8>),
    /// Cross-stage data identified by a transport-specific handle.
    Transfer(TransferHandle),
}

impl InlineValue {
    /// Returns embedded bytes, or `None` for an external transfer.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Transfer(_) => None,
        }
    }

    /// Returns the external transfer descriptor, if present.
    pub const fn transfer(&self) -> Option<&TransferHandle> {
        match self {
            Self::Bytes(_) => None,
            Self::Transfer(handle) => Some(handle),
        }
    }
}

/// A resolved inline or transfer value matched to one exact product identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductPayload {
    /// Exact product identity and bounds.
    pub product: ProductRef,
    /// Materialized or transport-referenced product value.
    pub value: InlineValue,
}

impl ProductPayload {
    /// Validates product identity and the selected inline value.
    pub fn validate(&self) -> ValidationResult<()> {
        self.product.validate()
    }

    /// Validates this payload as a worker input value.
    pub(crate) fn validate_input_value(&self) -> ValidationResult<()> {
        if let InlineValue::Transfer(handle) = &self.value {
            return validate_transfer_handle(&self.product, handle, false);
        }
        let bytes = self.value.bytes().expect("inline bytes");
        match self.product.kind {
            ProductKind::Token => {
                let tokens = decode_token_product_bytes(bytes)?;
                ensure_valid!(
                    tokens.len() as u64 <= self.product.shape_bound.max_elements(),
                    "token input product exceeds its registered element bound"
                );
            }
            ProductKind::SamplingState => {
                decode_sampling_state_bytes(bytes)?;
                ensure_valid!(
                    bytes.len() as u64 <= self.product.max_bytes(),
                    "sampling-state input exceeds its registered byte bound"
                );
            }
            _ => ensure_valid!(
                bytes.len() as u64 <= self.product.max_bytes(),
                "input product payload exceeds its registered byte bound"
            ),
        }
        Ok(())
    }

    /// Validates this payload as a worker output value.
    pub(crate) fn validate_output_value(&self) -> ValidationResult<()> {
        self.validate()?;
        match &self.value {
            InlineValue::Transfer(handle) => validate_transfer_handle(&self.product, handle, true)?,
            InlineValue::Bytes(bytes) => ensure_valid!(
                bytes.len() as u64 <= self.product.max_bytes(),
                "output product payload exceeds its registered byte bound"
            ),
        }
        Ok(())
    }
}

/// Maximum serialized size accepted for an external transfer handle.
pub const MAX_TRANSFER_HANDLE_BYTES: usize = 64 * 1024;

impl TransferLocator {
    /// Validates common tensor bounds and transport-specific opening metadata.
    fn validate(&self) -> ValidationResult<()> {
        // Tensor metadata is transport-independent and establishes the minimum
        // shape needed to validate every publication mechanism.
        ensure_valid!(
            self.nbytes > 0
                && !self.dtype.is_empty()
                && !self.shape.is_empty()
                && self.shape.iter().all(|extent| *extent > 0)
                && !self.device.is_empty(),
            "transfer locator has invalid tensor bounds"
        );

        // Each transport validates only the handles and readiness metadata its
        // consumer must use to open the publication.
        match &self.transport {
            TransferTransport::Local { endpoint, .. } => {
                ensure_valid!(!endpoint.is_empty(), "local transfer endpoint is empty");
            }
            TransferTransport::PosixShm {
                name,
                ready_header_bytes,
                ready_semaphore,
            } => {
                ensure_valid!(!name.is_empty(), "shared-memory transfer name is empty");
                ensure_valid!(
                    *ready_header_bytes == 0
                        || ready_semaphore
                            .as_ref()
                            .is_some_and(|name| !name.is_empty()),
                    "asynchronous shared-memory transfer has no readiness semaphore"
                );
            }
            TransferTransport::CudaIpc {
                endpoint,
                publication_id,
                storage_handle,
                storage_size_bytes,
                tensor_stride,
                ref_counter_handle,
                event_handle,
                ready_event_handle,
                ..
            } => {
                ensure_valid!(
                    !endpoint.is_empty()
                        && !publication_id.is_empty()
                        && !storage_handle.is_empty()
                        && *storage_size_bytes > 0
                        && tensor_stride.len() == self.shape.len()
                        && !ref_counter_handle.is_empty()
                        && !event_handle.is_empty()
                        && !ready_event_handle.is_empty(),
                    "CUDA IPC transfer handle is incomplete"
                );
                let opaque_bytes = storage_handle
                    .len()
                    .saturating_add(ref_counter_handle.len())
                    .saturating_add(event_handle.len())
                    .saturating_add(ready_event_handle.len());

                ensure_valid!(
                    opaque_bytes <= MAX_TRANSFER_HANDLE_BYTES,
                    "CUDA IPC transfer handles exceed their byte bound"
                );
            }
        }
        Ok(())
    }
}

/// Validates a transfer handle against its declared product and direction.
fn validate_transfer_handle(
    product: &ProductRef,
    handle: &TransferHandle,
    output: bool,
) -> ValidationResult<()> {
    // The product declaration controls which storage classes may cross a pool
    // boundary and binds the handle to one allocation generation.
    ensure_valid!(
        product.storage_class != StorageClass::HostStaging
            && (!output || product.storage_class != StorageClass::PinnedOutput),
        "product transfer handle has an invalid storage class"
    );
    ensure_valid!(
        handle.generation() == product.generation,
        "transfer generation disagrees with its product"
    );

    // Product-family metadata must agree before inspecting physical locators.
    match handle {
        TransferHandle::Encoder {
            payload_kind,
            height,
            width,
            ..
        } => ensure_valid!(
            matches!(
                payload_kind,
                ProductKind::VisionFeature | ProductKind::LatentFeature
            ) && *payload_kind == product.kind
                && *height > 0
                && *width > 0,
            "encoder transfer disagrees with its product"
        ),
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            ..
        } => ensure_valid!(
            (*height == 0) == (*width == 0) && (*height > 0 || value_range.is_empty()),
            "device-product transfer geometry is incomplete"
        ),
        TransferHandle::Kv {
            locators,
            source,
            destination,
            base,
            base_extent,
            published_extent,
            scale_identity,
            ..
        } => {
            ensure_valid!(
                product.kind == ProductKind::Kv,
                "KV transfer names a non-KV product"
            );
            ensure_valid!(!locators.is_empty(), "KV transfer has no physical values");
            source.validate()?;
            if let Some(base) = base {
                base.validate()?;
            }
            ensure_valid!(
                !destination.is_empty()
                    && !scale_identity.is_empty()
                    && *base_extent <= *published_extent
                    && (base.is_some() || *base_extent == 0),
                "KV transfer publication metadata is invalid"
            );
        }
        TransferHandle::Latent {
            height,
            width,
            latent_units,
            ..
        } => ensure_valid!(
            product.kind == ProductKind::Latent && *height > 0 && *width > 0 && *latent_units > 0,
            "latent transfer disagrees with its product"
        ),
    }

    // Validate each locator and bound their combined payload by the declared
    // maximum product shape.
    let mut total_bytes = 0_u64;
    for locator in handle.locators() {
        locator.validate()?;
        total_bytes = total_bytes.saturating_add(locator.nbytes);
    }
    ensure_valid!(
        total_bytes <= product.max_bytes(),
        "transfer values exceed their product byte bound"
    );
    Ok(())
}

/// Encodes a `ProductKind::Token` product value: a little-endian `u32` count
/// followed by that many little-endian `u32` token ids.
pub fn encode_token_product_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + tokens.len() * 4);
    bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

/// Decodes a `ProductKind::Token` product value produced by
/// [`encode_token_product_bytes`].
pub fn decode_token_product_bytes(bytes: &[u8]) -> ValidationResult<Vec<u32>> {
    // Validate the count prefix before interpreting the remaining bytes.
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
    // Exact length makes every remaining chunk a complete token identifier.
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
    /// Optional whitelist of token identifiers eligible for sampling.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Token identifiers excluded from sampling.
    pub suppressed_token_ids: Vec<u32>,
    /// Token identifiers that terminate generation when selected.
    pub finish_token_ids: Vec<u32>,
    /// Token identifiers that advance a structured-generation transition.
    pub transition_token_ids: Vec<u32>,
    /// Whether the current grammar state requires immediate termination.
    pub force_finish: bool,
}

impl SamplingState {
    /// Sorts and deduplicates token identifier sets for deterministic encoding.
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

/// Encodes canonical branch-local sampling state.
///
/// Layout: one allowed-presence byte; an allowed count and ids when present;
/// then a suppressed count and ids; a finish count and ids; a transition count
/// and ids; then one force-finish byte.
pub fn encode_sampling_state_bytes(state: &SamplingState) -> Vec<u8> {
    // Canonical ordering makes the byte representation deterministic.
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
    // Encode the optional whitelist before the required token-id sets.
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
    // A terminal byte keeps the boolean outside all length-prefixed lists.
    bytes.push(u8::from(canonical.force_finish));
    bytes
}

/// Decodes and validates canonical branch-local sampling state.
pub fn decode_sampling_state_bytes(bytes: &[u8]) -> ValidationResult<SamplingState> {
    /// Takes one little-endian word while advancing a checked cursor.
    fn take_u32(bytes: &[u8], offset: &mut usize) -> ValidationResult<u32> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| invalid_message!("sampling-state offset overflow"))?;
        ensure_valid!(end <= bytes.len(), "sampling-state bytes are truncated");
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        Ok(value)
    }

    /// Takes a canonical strictly increasing token-id sequence.
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

    // Decode the optional whitelist using its explicit presence byte.
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
    // Decode each required canonical set in protocol order.
    let suppressed_len = take_u32(bytes, &mut offset)?;
    let suppressed_token_ids = take_ids(bytes, &mut offset, suppressed_len)?;
    let finish_len = take_u32(bytes, &mut offset)?;
    let finish_token_ids = take_ids(bytes, &mut offset, finish_len)?;
    let transition_len = take_u32(bytes, &mut offset)?;
    let transition_token_ids = take_ids(bytes, &mut offset, transition_len)?;
    // Require a strict boolean terminator and reject any trailing bytes.
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
