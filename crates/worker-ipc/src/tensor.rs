//! Tensor identities, bounded representations, and physical transfers.

use super::*;

/// Element type of a tensor's backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8 = 0,
    /// Signed 32-bit integer.
    I32 = 1,
    /// Signed 64-bit integer.
    I64 = 2,
    /// IEEE 754 half precision.
    F16 = 3,
    /// Brain floating-point half precision.
    #[serde(rename = "bf16")]
    BF16 = 4,
    /// IEEE 754 single precision.
    F32 = 5,
    /// Signed 16-bit integer.
    I16 = 6,
}

impl DType {
    /// Physical tensor storage width, including signed 16-bit PCM values.
    pub fn element_bytes(self) -> u64 {
        tensor_dtype(self).1
    }
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
/// carries a fixed maximum. A single device-actual axis without static axes
/// denotes flat capacity; its actual tensor rank is published with the transfer.
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
    pub fn max_elements(&self) -> u64 {
        self.dims.iter().fold(1_u64, |elements, dim| {
            elements.saturating_mul(u64::from(match dim {
                DimBound::Static(value) => *value,
                DimBound::Device { max } => *max,
            }))
        })
    }
}

/// Identity, element type, and bounded capacity of a tensor that survives its
/// producing computation. Actual geometry and locations are published separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorRef {
    /// Request that owns the value.
    pub request_key: RequestKey,
    /// Call that declares the value.
    pub producer_call_id: CallId,
    /// Position in the producer's output list.
    pub output_index: u16,
    /// Nonzero allocation generation preventing identity reuse.
    pub generation: u32,
    /// Element data type.
    pub dtype: DType,
    /// Maximum logical tensor shape.
    pub shape_bound: ShapeBound,
}

/// Stable identity for one cross-call buffer, independent of its physical representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId {
    /// Request that owns the buffer.
    pub owner: RequestKey,
    /// Call that first declares the buffer.
    pub producer_call_id: CallId,
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

impl TensorRef {
    /// Returns the exact storage identity used for routing and retirement.
    pub const fn buffer_id(&self) -> BufferId {
        BufferId {
            owner: self.request_key,
            producer_call_id: self.producer_call_id,
            output_index: self.output_index,
            generation: self.generation,
        }
    }

    /// Validates allocation identity and tensor capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.generation > 0,
            "product reference has no logical generation"
        );
        self.shape_bound.validate()
    }

    /// Returns the maximum encoded byte size allowed by the shape and dtype.
    pub fn max_bytes(&self) -> u64 {
        self.shape_bound
            .max_elements()
            .saturating_mul(self.dtype.element_bytes())
    }
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
    /// POSIX shared-memory publication with endpoint-driven readiness and ownership.
    PosixShm {
        /// Publishing address-space incarnation and reader-lease endpoint.
        endpoint: String,
        /// Shared-memory object name.
        name: String,
    },
    /// CUDA publication whose granted reader receives a physical-allocation descriptor.
    CudaVmm {
        /// Publishing worker endpoint.
        endpoint: String,
        /// Stable publication identity.
        publication_id: String,
        /// Exported allocation size in bytes.
        storage_size_bytes: u64,
        /// Byte offsets of ordered first-axis spans within one allocation.
        storage_offsets_bytes: Vec<u64>,
        /// First-axis lengths of consecutive runs of equally sized physical spans.
        span_lengths: Vec<u64>,
        /// Number of spans in each length run; trailing geometry and strides are shared.
        span_counts: Vec<u32>,
        /// Tensor stride in elements.
        tensor_stride: Vec<i64>,
        /// Opaque CUDA event handle signaling publication readiness.
        #[serde(with = "serde_bytes")]
        ready_event_handle: Vec<u8>,
        /// The producing rank's shareable allocation handle, of the type its
        /// device was probed for. A fabric handle is importable from another
        /// host, so it travels here rather than through a descriptor grant
        /// that only reaches the producer's own host.
        #[serde(with = "serde_bytes")]
        allocation_handle: Vec<u8>,
        /// Byte offset of this publication's acknowledgment header inside the
        /// exported allocation. A consumer writes its own slot's word there
        /// once its reads retire, which is how a product retires across hosts.
        /// Negative when the publication carries no header.
        acknowledgment_offset: i64,
    },
    /// A host product carried on the rank channel's data path.
    ///
    /// Shared memory names a segment in one host's namespace, so a product
    /// whose consumer is on another host travels as bytes: in the producing
    /// rank's result, into the head's custody, and out in the consuming rank's
    /// batch. The head releases its copy when the buffer is freed.
    Channel {
        /// Publishing address-space incarnation, for diagnostics and identity.
        endpoint: String,
        /// The product's bytes in physical tensor order.
        #[serde(with = "serde_bytes")]
        payload: Vec<u8>,
    },
}

/// One transport-native reference with explicit tensor bounds. Transport handles
/// are opaque bytes only where the underlying CUDA API defines an opaque handle;
/// semantic transfer metadata remains typed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Locator {
    /// Loaded producer rank that owns this publication.
    pub source: WorkerEndpoint,
    /// Transport-specific publication descriptor.
    pub transport: TransferTransport,
    /// Tensor payload size in bytes.
    pub nbytes: u64,
    /// Stable tensor data-type name.
    pub dtype: String,
    /// Tensor extents in logical order.
    pub shape: Vec<u64>,
    /// Logical element offset of this shard within its tensor.
    pub offset: Vec<u64>,
    /// Device containing the published tensor.
    pub device: String,
}

/// Actual logical tensor shape and immutable shard or replica locations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorTransfer {
    pub shape: Vec<u64>,
    pub locations: Vec<Locator>,
}

impl TensorTransfer {
    /// Returns whether the available shard/replica boxes cover the complete logical tensor.
    /// Partial rank reports are valid descriptors, so completeness is a separate question.
    pub fn has_complete_coverage(&self) -> bool {
        if self.validate().is_err() {
            return false;
        }
        let mut uncovered = vec![(vec![0; self.shape.len()], self.shape.clone())];
        for location in &self.locations {
            let mut remaining = Vec::new();
            for (start, end) in uncovered {
                let lower = start
                    .iter()
                    .zip(&location.offset)
                    .map(|(a, b)| (*a).max(*b))
                    .collect::<Vec<_>>();
                let upper = end
                    .iter()
                    .zip(&location.offset)
                    .zip(&location.shape)
                    .map(|((end, offset), extent)| (*end).min(offset.saturating_add(*extent)))
                    .collect::<Vec<_>>();
                if lower.iter().zip(&upper).any(|(a, b)| a >= b) {
                    remaining.push((start, end));
                    continue;
                }
                let (mut middle_start, mut middle_end) = (start, end);
                for axis in 0..self.shape.len() {
                    if middle_start[axis] < lower[axis] {
                        let mut slab_end = middle_end.clone();
                        slab_end[axis] = lower[axis];
                        remaining.push((middle_start.clone(), slab_end));
                        middle_start[axis] = lower[axis];
                    }
                    if upper[axis] < middle_end[axis] {
                        let mut slab_start = middle_start.clone();
                        slab_start[axis] = upper[axis];
                        remaining.push((slab_start, middle_end.clone()));
                        middle_end[axis] = upper[axis];
                    }
                }
            }
            if remaining.is_empty() {
                return true;
            }
            uncovered = remaining;
        }
        false
    }

    /// Validates physical coverage and returns the logical, replica-independent byte size.
    pub fn validate(&self) -> ValidationResult<u64> {
        ensure_valid!(
            !self.shape.is_empty() && self.shape.iter().all(|&n| n > 0),
            "tensor transfer has no geometry"
        );
        let first = self
            .locations
            .first()
            .ok_or_else(|| invalid_message!("tensor transfer has no locations"))?;
        first.validate()?;
        let elements = tensor_elements(&first.shape)?;
        ensure_valid!(
            first.nbytes >= elements && first.nbytes % elements == 0,
            "tensor transfer has an invalid element size"
        );
        let element_bytes = first.nbytes / elements;
        for location in &self.locations {
            location.validate()?;
            ensure_valid!(
                location.shape.len() == self.shape.len()
                    && location
                        .offset
                        .iter()
                        .zip(&location.shape)
                        .zip(&self.shape)
                        .all(|((&offset, &extent), &bound)| offset
                            .checked_add(extent)
                            .is_some_and(|end| end <= bound))
                    && location.dtype == first.dtype
                    && tensor_elements(&location.shape)?.checked_mul(element_bytes)
                        == Some(location.nbytes),
                "tensor location disagrees with its logical representation"
            );
        }
        tensor_elements(&self.shape)?
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid_message!("logical tensor byte size overflows"))
    }
}

fn tensor_elements(shape: &[u64]) -> ValidationResult<u64> {
    shape.iter().try_fold(1u64, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| invalid_message!("tensor element count overflows"))
    })
}

/// A published KV extent and the physical tensors needed to install its suffix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTransfer {
    /// Published KV tensors.
    pub tensors: Vec<TensorTransfer>,
    /// Published buffer identity represented by the publication.
    pub source: BufferId,
    /// Destination worker or pool identity.
    pub destination: String,
    /// Optional published buffer already installed at the destination.
    pub base: Option<BufferId>,
    /// KV token extent represented by `base`.
    pub base_extent: u32,
    /// Total KV token extent represented by this publication.
    pub published_extent: u32,
    /// KV cache group identity.
    pub group_id: u32,
    /// Compute precision used when reading quantized source pages.
    pub compute_dtype: String,
    /// Tokens per source page, including the boundary scale interpretation.
    pub page_size: u32,
}

impl KvTransfer {
    /// Validate source/base identities and raw K/V/scale geometry, including rank shards.
    pub fn validate(&self) -> ValidationResult<()> {
        let Self {
            tensors,
            source,
            destination,
            base,
            base_extent,
            published_extent,
            compute_dtype,
            page_size,
            ..
        } = self;
        ensure_valid!(
            tensors.is_empty() == (published_extent == base_extent),
            "KV tensor presence disagrees with its incremental extent"
        );
        source.validate()?;
        if let Some(base) = base {
            base.validate()?;
        }
        ensure_valid!(
            !destination.is_empty()
                && matches!(
                    compute_dtype.as_str(),
                    "float16" | "bfloat16" | "float32" | "float64"
                )
                && *page_size > 0
                && *base_extent <= *published_extent
                && (base.is_some() || *base_extent == 0),
            "KV transfer publication metadata is invalid"
        );
        if published_extent > base_extent {
            ensure_valid!(
                matches!(tensors.len(), 2 | 3),
                "KV transfer requires raw keys, values and optional scales"
            );
            let key = &tensors[0];
            let value = &tensors[1];
            let dtype = key
                .locations
                .first()
                .map(|location| location.dtype.as_str());
            ensure_valid!(
                key.shape.len() == 4
                    && key.shape[0] == u64::from(published_extent - base_extent)
                    && value.shape == key.shape
                    && value
                        .locations
                        .first()
                        .map(|location| location.dtype.as_str())
                        == dtype
                    && matches!(
                        dtype,
                        Some("float16" | "bfloat16" | "float32" | "float64" | "float8_e4m3fn")
                    ),
                "KV transfer has invalid raw token/layer/head geometry"
            );
            let quantized = dtype == Some("float8_e4m3fn");
            ensure_valid!(
                (tensors.len() == 3) == quantized,
                "KV transfer scale presence disagrees with its storage"
            );
            if quantized {
                let scales = &tensors[2];
                let tokens =
                    u64::from(base_extent % page_size) + u64::from(published_extent - base_extent);
                let pages = tokens.div_ceil(u64::from(*page_size));
                ensure_valid!(
                    scales.shape.len() == 4
                        && scales.shape[..3] == [pages, 2, key.shape[1]]
                        && scales.shape[3] > 0
                        && key.shape[2] % scales.shape[3] == 0
                        && scales
                            .locations
                            .first()
                            .map(|location| location.dtype.as_str())
                            == Some("float32"),
                    "KV transfer scales disagree with its source pages"
                );
            }
        }
        for tensor in tensors {
            tensor.validate()?;
        }
        ensure_valid!(
            self.encoded_size_bound() <= MAX_TRANSFER_HANDLE_BYTES,
            "KV transfer exceeds its descriptor byte bound"
        );
        Ok(())
    }

    /// Conservative wire size, including every shard and replica locator.
    pub fn encoded_size_bound(&self) -> usize {
        transfer_encoded_size(&self.tensors)
            .saturating_add(self.destination.len())
            .saturating_add(self.compute_dtype.len())
    }

    /// Merge reports for one immutable publication without exposing a partial update.
    pub fn merge_locations(&mut self, other: &Self) -> ValidationResult<()> {
        ensure_valid!(
            self.source == other.source
                && self.destination == other.destination
                && self.base == other.base
                && self.base_extent == other.base_extent
                && self.published_extent == other.published_extent
                && self.group_id == other.group_id
                && self.compute_dtype == other.compute_dtype
                && self.page_size == other.page_size,
            "KV locations disagree on publication metadata"
        );
        let mut candidate = self.clone();
        merge_tensor_locations(&mut candidate.tensors, &other.tensors)?;
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

/// Encoding represented by a reusable image-feature publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureKind {
    #[serde(rename = "vision_feature")]
    Vision,
    #[serde(rename = "latent_feature")]
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
        /// Source image height in pixels.
        height: u32,
        /// Source image width in pixels.
        width: u32,
        /// Vision or latent feature product kind.
        payload_kind: FeatureKind,
        /// Published feature tensor.
        tensor: TensorTransfer,
    },
    /// Device-resident artifact used by another model stage.
    DeviceProduct {
        /// Artifact height in pixels, or zero for non-image tensors.
        height: u32,
        /// Artifact width in pixels, or zero for non-image tensors.
        width: u32,
        /// Semantic numeric range of the tensor values.
        value_range: String,
        /// Published artifact tensor.
        tensor: TensorTransfer,
    },
    /// Diffusion trajectory tensor.
    Latent {
        /// Output image height in pixels.
        height: u32,
        /// Output image width in pixels.
        width: u32,
        /// Logical latent allocation units.
        latent_units: u32,
        /// Denoising step represented by the tensor.
        step: u32,
        /// Published latent tensor.
        tensor: TensorTransfer,
    },
}

impl TransferHandle {
    /// Mutable tensor descriptors for endpoint binding and rank-location invalidation.
    pub fn tensors_mut(&mut self) -> &mut [TensorTransfer] {
        match self {
            Self::Encoder { tensor, .. }
            | Self::DeviceProduct { tensor, .. }
            | Self::Latent { tensor, .. } => std::slice::from_mut(tensor),
        }
    }

    /// Returns logical tensors in their product-family order.
    pub fn tensors(&self) -> &[TensorTransfer] {
        match self {
            Self::Encoder { tensor, .. }
            | Self::DeviceProduct { tensor, .. }
            | Self::Latent { tensor, .. } => std::slice::from_ref(tensor),
        }
    }

    /// Conservative size bound for the complete wire descriptor, including replicas.
    pub fn encoded_size_bound(&self) -> usize {
        transfer_encoded_size(self.tensors())
    }

    /// Returns every physical shard and replica carried by this product.
    pub fn locators(&self) -> impl Iterator<Item = &Locator> {
        self.tensors().iter().flat_map(|tensor| &tensor.locations)
    }
}

fn transfer_encoded_size(tensors: &[TensorTransfer]) -> usize {
    let mut size = 512usize;
    for tensor in tensors {
        size = size
            .saturating_add(64)
            .saturating_add(8usize.saturating_mul(tensor.shape.len()));
        for location in &tensor.locations {
            let source = &location.source;
            size = size
                .saturating_add(256)
                .saturating_add(location.dtype.len())
                .saturating_add(location.device.len())
                .saturating_add(16usize.saturating_mul(location.shape.len()))
                .saturating_add(source.worker_id.len())
                .saturating_add(source.node.len())
                .saturating_add(source.address_space.len())
                .saturating_add(source.incarnation.len());
            let native = match &location.transport {
                TransferTransport::Local { endpoint, .. } => endpoint.len().saturating_add(16),
                TransferTransport::PosixShm { endpoint, name } => {
                    endpoint.len().saturating_add(name.len()).saturating_add(16)
                }
                TransferTransport::CudaVmm {
                    endpoint,
                    publication_id,
                    ready_event_handle,
                    allocation_handle,
                    tensor_stride,
                    span_lengths,
                    storage_offsets_bytes,
                    ..
                } => endpoint
                    .len()
                    .saturating_add(publication_id.len())
                    .saturating_add(ready_event_handle.len())
                    .saturating_add(allocation_handle.len())
                    .saturating_add(8usize.saturating_mul(tensor_stride.len()))
                    .saturating_add(8usize.saturating_mul(storage_offsets_bytes.len()))
                    .saturating_add(12usize.saturating_mul(span_lengths.len()))
                    .saturating_add(64),
                // The bytes are the product, not the handle: their budget is
                // the channel's capacity and the rank channel's message caps,
                // so only the locator's framing counts toward the handle bound.
                TransferTransport::Channel { endpoint, .. } => endpoint.len().saturating_add(24),
            };
            size = size.saturating_add(native);
        }
    }
    size
}

fn merge_tensor_locations(
    destination: &mut [TensorTransfer],
    source: &[TensorTransfer],
) -> ValidationResult<()> {
    ensure_valid!(
        destination.len() == source.len(),
        "locations disagree on tensor count"
    );
    for (destination, source) in destination.iter_mut().zip(source) {
        ensure_valid!(
            destination.validate()? == source.validate()?
                && destination.shape == source.shape
                && destination.locations[0].dtype == source.locations[0].dtype,
            "product locations disagree on logical tensor representation"
        );
        for location in &source.locations {
            if !destination.locations.contains(location) {
                destination.locations.push(location.clone());
            }
        }
    }
    Ok(())
}

/// Published tensor storage matched to one exact product identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPublication {
    /// Exact product identity and bounds.
    pub product: TensorRef,
    /// Transport descriptor for the published tensor.
    pub value: TransferHandle,
}

impl TensorPublication {
    /// Adds locations of the same immutable logical value without changing its metadata.
    pub fn merge_locations(&mut self, other: &Self) -> ValidationResult<()> {
        ensure_valid!(
            self.product == other.product,
            "tensor locations belong to different tensor identities"
        );
        let agrees = match (&self.value, &other.value) {
            (
                TransferHandle::Encoder {
                    height,
                    width,
                    payload_kind,
                    ..
                },
                TransferHandle::Encoder {
                    height: other_height,
                    width: other_width,
                    payload_kind: other_payload_kind,
                    ..
                },
            ) => {
                height == other_height && width == other_width && payload_kind == other_payload_kind
            }
            (
                TransferHandle::DeviceProduct {
                    height,
                    width,
                    value_range,
                    ..
                },
                TransferHandle::DeviceProduct {
                    height: other_height,
                    width: other_width,
                    value_range: other_value_range,
                    ..
                },
            ) => height == other_height && width == other_width && value_range == other_value_range,
            (
                TransferHandle::Latent {
                    height,
                    width,
                    latent_units,
                    step,
                    ..
                },
                TransferHandle::Latent {
                    height: other_height,
                    width: other_width,
                    latent_units: other_latent_units,
                    step: other_step,
                    ..
                },
            ) => {
                height == other_height
                    && width == other_width
                    && latent_units == other_latent_units
                    && step == other_step
            }
            _ => false,
        };
        ensure_valid!(agrees, "product locations disagree on semantic metadata");
        let mut candidate = self.clone();
        merge_tensor_locations(candidate.value.tensors_mut(), other.value.tensors())?;
        ensure_valid!(
            candidate.value.encoded_size_bound() <= MAX_TRANSFER_HANDLE_BYTES,
            "merged tensor locations exceed their descriptor byte bound"
        );
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    /// Validates identity, bounds, and published tensor storage.
    pub fn validate(&self) -> ValidationResult<()> {
        self.product.validate()?;
        validate_transfer_handle(&self.product, &self.value)
    }
}

/// Maximum serialized size accepted for an external transfer handle.
pub const MAX_TRANSFER_HANDLE_BYTES: usize = 64 * 1024;

impl Locator {
    /// Validates common tensor bounds and transport-specific opening metadata.
    fn validate(&self) -> ValidationResult<()> {
        self.source.validate()?;
        // Tensor metadata is transport-independent and establishes the minimum
        // shape needed to validate every publication mechanism.
        ensure_valid!(
            self.nbytes > 0
                && !self.dtype.is_empty()
                && !self.shape.is_empty()
                && self.shape.iter().all(|extent| *extent > 0)
                && !self.device.is_empty()
                && self.offset.len() == self.shape.len(),
            "transfer locator has invalid tensor bounds"
        );

        // Each transport validates only the handles and readiness metadata its
        // consumer must use to open the publication.
        match &self.transport {
            TransferTransport::Local { endpoint, .. } => {
                ensure_valid!(!endpoint.is_empty(), "local transfer endpoint is empty");
            }
            TransferTransport::PosixShm { endpoint, name } => {
                ensure_valid!(
                    !endpoint.is_empty(),
                    "shared-memory transfer endpoint is empty"
                );
                ensure_valid!(!name.is_empty(), "shared-memory transfer name is empty");
            }
            TransferTransport::CudaVmm {
                endpoint,
                publication_id,
                storage_size_bytes,
                storage_offsets_bytes,
                span_lengths,
                span_counts,
                tensor_stride,
                ready_event_handle,
                allocation_handle,
                ..
            } => {
                ensure_valid!(
                    !endpoint.is_empty()
                        && publication_id.len() == 32
                        && *storage_size_bytes > 0
                        && !storage_offsets_bytes.is_empty()
                        && span_counts.len() == span_lengths.len()
                        && span_counts.iter().all(|count| *count > 0)
                        && span_counts
                            .iter()
                            .map(|count| u64::from(*count))
                            .sum::<u64>()
                            == storage_offsets_bytes.len() as u64
                        && storage_offsets_bytes
                            .iter()
                            .all(|offset| offset < storage_size_bytes)
                        && span_lengths.iter().all(|length| *length > 0)
                        && span_lengths.iter().zip(span_counts).try_fold(
                            0u64,
                            |sum, (length, count)| {
                                sum.checked_add(length.checked_mul(u64::from(*count))?)
                            }
                        ) == self.shape.first().copied()
                        && tensor_stride.len() == self.shape.len()
                        // A publication read only from this host carries the
                        // event its consumers wait on; one read from another
                        // host carries no fence, because none would reach
                        // there, and its producer drained its stream before
                        // publishing instead.
                        && (ready_event_handle.len() == 64
                            || ready_event_handle.is_empty())
                        && matches!(allocation_handle.len(), 4 | 64)
                        && tensor_stride.iter().all(|stride| *stride >= 0),
                    "CUDA VMM transfer handle is incomplete"
                );
                let opaque_bytes = ready_event_handle.len() + allocation_handle.len();

                ensure_valid!(
                    opaque_bytes <= MAX_TRANSFER_HANDLE_BYTES,
                    "CUDA VMM transfer handles exceed their byte bound"
                );
            }
            TransferTransport::Channel { endpoint, payload } => {
                ensure_valid!(!endpoint.is_empty(), "channel transfer endpoint is empty");
                // The bytes are the product; a locator without them names
                // nothing a consumer could read.
                ensure_valid!(
                    payload.len() as u64 == self.nbytes,
                    "channel transfer payload disagrees with its product"
                );
            }
        }
        Ok(())
    }
}

/// Validates a transfer handle against its declared product.
fn validate_transfer_handle(product: &TensorRef, handle: &TransferHandle) -> ValidationResult<()> {
    // Product-family metadata must agree before inspecting physical locators.
    match handle {
        TransferHandle::Encoder {
            payload_kind,
            height,
            width,
            ..
        } => ensure_valid!(
            matches!(payload_kind, FeatureKind::Vision | FeatureKind::Latent)
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
        TransferHandle::Latent {
            height,
            width,
            latent_units,
            ..
        } => ensure_valid!(
            *height > 0 && *width > 0 && *latent_units > 0,
            "latent transfer disagrees with its product"
        ),
    }

    // Validate each locator and bound their combined payload by the declared
    // maximum product shape.
    {
        let tensor = &handle.tensors()[0];
        let (dtype, element_bytes) = tensor_dtype(product.dtype);
        let nbytes = tensor.validate()?;
        ensure_valid!(
            tensor.locations[0].dtype == dtype
                && tensor_elements(&tensor.shape)?.checked_mul(element_bytes) == Some(nbytes),
            "transfer tensor dtype disagrees with its product"
        );
        let bounds = &product.shape_bound.dims;
        let shape_matches = match bounds.as_slice() {
            [] => tensor_elements(&tensor.shape)? == 1,
            // A single dynamic extent describes flat capacity, including
            // encoder features and images whose actual rank is published later.
            [DimBound::Device { max }] => tensor_elements(&tensor.shape)? <= u64::from(*max),
            _ => {
                tensor.shape.len() == bounds.len()
                    && tensor
                        .shape
                        .iter()
                        .zip(bounds)
                        .all(|(&size, bound)| match bound {
                            DimBound::Static(expected) => size == u64::from(*expected),
                            DimBound::Device { max } => size <= u64::from(*max),
                        })
            }
        };
        ensure_valid!(
            shape_matches,
            "transfer tensor shape disagrees with its product"
        );
    }
    let mut total_bytes = 0_u64;
    for tensor in handle.tensors() {
        total_bytes = total_bytes.saturating_add(tensor.validate()?);
    }
    let byte_bound = product
        .shape_bound
        .max_elements()
        .saturating_mul(tensor_dtype(product.dtype).1);
    ensure_valid!(
        total_bytes <= byte_bound,
        "transfer values exceed their product byte bound"
    );
    ensure_valid!(
        handle.encoded_size_bound() <= MAX_TRANSFER_HANDLE_BYTES,
        "transfer handle exceeds its byte bound"
    );
    Ok(())
}

/// Scalar storage representation used by both allocation and transfer validation.
fn tensor_dtype(dtype: DType) -> (&'static str, u64) {
    match dtype {
        DType::U8 => ("uint8", 1),
        DType::I32 => ("int32", 4),
        DType::I16 => ("int16", 2),
        DType::I64 => ("int64", 8),
        DType::F16 => ("float16", 2),
        DType::BF16 => ("bfloat16", 2),
        DType::F32 => ("float32", 4),
    }
}
