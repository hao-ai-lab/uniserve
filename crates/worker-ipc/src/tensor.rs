//! Tensor identities, bounded representations, and physical transfers.
//!
//! The engine's scheduler declares a cross-call tensor product as a
//! [`TensorRef`]: a request-scoped identity plus a [`DType`] and a
//! [`ShapeBound`] that fix the product's maximum size before its producing call
//! runs. A producing worker reports the product as a [`TensorPublication`],
//! whose [`TransferHandle`] carries the actual logical shape
//! ([`TensorTransfer`]) and one [`Locator`] per shard or replica, each naming
//! the [`TransferTransport`] a consumer opens. A published KV extent travels as
//! a [`KvTransfer`] identified by a [`BufferId`] alone.
//!
//! The validators here are the wire contract: the codec runs them, through
//! `Batch::validate` and `BatchOutput::validate`, whenever it encodes or
//! decodes a batch or a batch result. The Python worker's
//! `uniserve_worker.protocol` package reimplements these checks and the
//! descriptor size estimate bounded by [`MAX_TRANSFER_HANDLE_BYTES`], so the
//! two sides must change together.

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
    /// Storage width of one element in bytes.
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
/// An empty bound (the default) admits exactly one element.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ShapeBound {
    /// Ordered tensor dimension bounds.
    pub dims: Vec<DimBound>,
}

impl ShapeBound {
    /// Rejects more than one device-actual dimension and any zero extent.
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

    /// Returns the maximum number of elements represented by this shape,
    /// saturating at `u64::MAX`; an empty bound yields one.
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
///
/// A [`TensorRef`] projects to one through [`TensorRef::buffer_id`]; a KV
/// publication carries one directly. The engine routes buffers and frees them
/// (`BatchCommand::Free`) by this identity.
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

    /// Returns the maximum storage size in bytes: the bound's maximum element
    /// count times the element width, saturating. `Batch::validate` requires
    /// each buffer output's allocation to be at least this large.
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
    /// POSIX shared-storage publication with endpoint-driven readiness and ownership.
    PosixShm {
        /// Publishing address-space incarnation and reader-lease endpoint.
        endpoint: String,
        /// Shared-storage object name.
        name: String,
    },
    /// CUDA VMM publication of device storage, which a consumer maps by
    /// importing the producer's allocation handle and viewing its spans.
    CudaVmm {
        /// Publishing worker endpoint.
        endpoint: String,
        /// Stable publication identity: exactly 32 bytes long (the producer
        /// uses a hex UUID). It also keys the producer's descriptor grant
        /// when `allocation_handle` is a process descriptor.
        publication_id: String,
        /// Exported allocation size in bytes.
        storage_size_bytes: u64,
        /// Byte offsets of ordered first-axis spans within one allocation, one
        /// per span; each must lie below `storage_size_bytes`.
        storage_offsets_bytes: Vec<u64>,
        /// First-axis lengths of consecutive runs of equally sized physical
        /// spans. Expanded by `span_counts`, they sum to the locator's first
        /// extent.
        span_lengths: Vec<u64>,
        /// Number of spans in each length run; trailing geometry and strides are shared.
        span_counts: Vec<u32>,
        /// Non-negative element strides shared by every span view, one per
        /// locator axis.
        tensor_stride: Vec<i64>,
        /// Opaque 64-byte CUDA IPC event handle signaling publication
        /// readiness, or empty. It is empty when any of the producing rank's
        /// consumers is on another host, where an event handle does not
        /// reach; the producer then synchronizes its stream before publishing.
        #[serde(with = "serde_bytes")]
        ready_event_handle: Vec<u8>,
        /// The producing rank's shareable allocation handle, of the type its
        /// device was probed for: a 64-byte fabric handle or a 4-byte process
        /// descriptor. A fabric handle is importable from another host, so it
        /// travels here rather than through a descriptor grant that only
        /// reaches the producer's own host. A descriptor names an open file
        /// of the producing process, so a consumer receives the usable one
        /// over the producer's grant socket, keyed by `publication_id`.
        #[serde(with = "serde_bytes")]
        allocation_handle: Vec<u8>,
        /// Byte offset of this publication's acknowledgment header inside the
        /// exported allocation. A consumer writes its own slot's word there
        /// once its reads retire, which is how a product retires across hosts.
        /// Negative when the publication carries no header, which is the case
        /// for storage exported where it lies rather than copied into the
        /// producer's device pool.
        acknowledgment_offset: i64,
    },
    /// A host product carried on the rank channel's data path.
    ///
    /// Shared storage names a segment in one host's namespace, so a product
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
    /// Payload size in bytes of this shard or replica.
    pub nbytes: u64,
    /// Stable tensor data-type name: the torch dtype name without its
    /// `torch.` prefix, such as `float32`.
    pub dtype: String,
    /// Extents of this shard or replica in logical axis order.
    pub shape: Vec<u64>,
    /// Per-axis element index at which this shard or replica starts within
    /// the logical tensor; it has one entry per axis of `shape`.
    pub offset: Vec<u64>,
    /// Device containing the published tensor.
    pub device: String,
}

/// Actual logical tensor shape and immutable shard or replica locations.
///
/// Each locator covers the box `[offset, offset + shape)` of the logical
/// tensor. Replicas may overlap, and a descriptor built from some ranks'
/// reports may leave regions uncovered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorTransfer {
    /// Actual logical extents of the whole tensor.
    pub shape: Vec<u64>,
    /// Shards and replicas that hold the tensor's bytes.
    pub locations: Vec<Locator>,
}

impl TensorTransfer {
    /// Returns whether the available shard/replica boxes cover the complete logical tensor.
    /// Partial rank reports are valid descriptors, so completeness is a separate question.
    ///
    /// Returns false for a descriptor that fails [`Self::validate`], including
    /// one without locations. The engine's worker executor calls this after
    /// dropping the locators of lost ranks to decide which buffers are lost.
    pub fn has_complete_coverage(&self) -> bool {
        if self.validate().is_err() {
            return false;
        }

        // Box subtraction: `uncovered` holds disjoint half-open boxes
        // `(start, end)` not yet covered by any locator. Each locator splits
        // every box it intersects into at most two slabs per axis outside the
        // intersection, and the intersection itself is discarded.
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
                // Peel off the part of the box below and above the
                // intersection on each axis in turn; after the last axis the
                // middle box equals the intersection and is dropped.
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

    /// Validates every locator against the logical shape and returns the
    /// logical, replica-independent byte size.
    ///
    /// Each locator's box must lie within `shape`, and all locators must share
    /// one dtype name and one element width. The width is inferred from the
    /// first locator's `nbytes` rather than from its dtype name; callers that
    /// know the expected dtype, such as `validate_transfer_handle`, check it.
    /// Coverage is not required; see [`Self::has_complete_coverage`].
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

/// Checked product of the extents; an empty shape yields one.
fn tensor_elements(shape: &[u64]) -> ValidationResult<u64> {
    shape.iter().try_fold(1u64, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| invalid_message!("tensor element count overflows"))
    })
}

/// One cache group's share of a KV publication.
///
/// The tensors carry the group's tokens `[start, published_extent)` of the
/// enclosing [`KvTransfer`]. A full-attention group starts at the
/// publication's `base_extent`; a sliding-window group starts no earlier than
/// the first token its readers need, so it never carries retired history.
/// With `T = published_extent - start`, the tensors use this layout, whose
/// shape relations [`KvTransfer::validate`] checks:
///
/// - keys and values: `[T, layers, kv heads, head dim]` over the group's
///   layers, one dtype shared by both (`float16`, `bfloat16`, `float32`,
///   `float64`, or `float8_e4m3fn`);
/// - with `float8_e4m3fn` only, a third `float32` scale tensor
///   `[pages, 2, layers, head groups]`, where `pages` counts the source pages
///   the carried tokens touch (the first one holds `start`), the second axis
///   selects K or V, and the kv-head count is a multiple of `head groups`.
///
/// A group that carries no token (`start == published_extent`) has no
/// tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvGroupTransfer {
    /// First absolute token the tensors carry.
    pub start: u32,
    /// Tokens per source page. With quantized storage it also determines how
    /// many scale rows the carried tokens span.
    pub page_tokens: u32,
    /// Published KV tensors: keys, values, and scales when the storage is
    /// quantized, in that order.
    pub tensors: Vec<TensorTransfer>,
}

impl KvGroupTransfer {
    /// Validates the carried interval and the raw K/V/scale geometry.
    fn validate(&self, base_extent: u32, published_extent: u32) -> ValidationResult<()> {
        let Self {
            start,
            page_tokens,
            tensors,
        } = self;
        ensure_valid!(
            *page_tokens > 0 && base_extent <= *start && *start <= published_extent,
            "KV group transfer interval is invalid"
        );
        ensure_valid!(
            tensors.is_empty() == (*start == published_extent),
            "KV group tensor presence disagrees with its carried interval"
        );
        if tensors.is_empty() {
            return Ok(());
        }

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
                && key.shape[0] == u64::from(published_extent - start)
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
            // Scale rows cover whole source pages, starting at the page that
            // holds `start`, so a partially filled boundary page contributes
            // its already installed tokens to the count.
            let scales = &tensors[2];
            let tokens = u64::from(start % page_tokens) + u64::from(published_extent - start);
            let pages = tokens.div_ceil(u64::from(*page_tokens));
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
        for tensor in tensors {
            tensor.validate()?;
        }
        Ok(())
    }
}

/// A published KV extent and the physical tensors needed to install its suffix.
///
/// The publication is incremental: the destination already holds `base`, when
/// set, up to `base_extent` tokens, and each group carries only its tokens
/// from its [`KvGroupTransfer::start`] on. An unchanged extent
/// (`published_extent == base_extent`) carries no groups; otherwise there is
/// one entry per cache group, in table order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTransfer {
    /// Published tensors of every cache group, in group order.
    pub groups: Vec<KvGroupTransfer>,
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
    /// Compute precision used when reading quantized source pages.
    pub compute_dtype: String,
}

impl KvTransfer {
    /// Validate source/base identities and every group's raw K/V/scale
    /// geometry, including rank shards.
    ///
    /// Also bounds the descriptor by [`MAX_TRANSFER_HANDLE_BYTES`] as
    /// estimated by [`KvTransfer::encoded_size_bound`].
    pub fn validate(&self) -> ValidationResult<()> {
        let Self {
            groups,
            source,
            destination,
            base,
            base_extent,
            published_extent,
            compute_dtype,
        } = self;
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
                && *base_extent <= *published_extent
                && (base.is_some() || *base_extent == 0),
            "KV transfer publication metadata is invalid"
        );
        ensure_valid!(
            groups.is_empty() == (published_extent == base_extent),
            "KV group presence disagrees with its incremental extent"
        );
        for group in groups {
            group.validate(*base_extent, *published_extent)?;
        }
        ensure_valid!(
            self.encoded_size_bound() <= MAX_TRANSFER_HANDLE_BYTES,
            "KV transfer exceeds its descriptor byte bound"
        );
        Ok(())
    }

    /// Iterates over every group's published tensors in group order.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorTransfer> {
        self.groups.iter().flat_map(|group| &group.tensors)
    }

    /// Iterates mutably over every group's published tensors in group order,
    /// for rebinding or pruning their locators.
    pub fn tensors_mut(&mut self) -> impl Iterator<Item = &mut TensorTransfer> {
        self.groups.iter_mut().flat_map(|group| &mut group.tensors)
    }

    /// Conservative wire size, including every shard and replica locator.
    /// The Python worker's `KvTransfer.encoded_size_bound` computes a
    /// parallel estimate.
    pub fn encoded_size_bound(&self) -> usize {
        // Each group record adds its interval and page size to the envelope.
        transfer_encoded_size(self.tensors())
            .saturating_add(32usize.saturating_mul(self.groups.len()))
            .saturating_add(self.destination.len())
            .saturating_add(self.compute_dtype.len())
    }

    /// Merge reports for one immutable publication without exposing a partial update.
    ///
    /// The engine merges reports that carry locators for the same `source`,
    /// such as per-rank reports of one call, this way. Every publication field
    /// except the locators must agree, and the merged result must pass
    /// [`KvTransfer::validate`]; on any error `self` is left unchanged.
    pub fn merge_locations(&mut self, other: &Self) -> ValidationResult<()> {
        ensure_valid!(
            self.source == other.source
                && self.destination == other.destination
                && self.base == other.base
                && self.base_extent == other.base_extent
                && self.published_extent == other.published_extent
                && self.compute_dtype == other.compute_dtype
                && self.groups.len() == other.groups.len()
                && self.groups.iter().zip(&other.groups).all(|(left, right)| {
                    left.start == right.start && left.page_tokens == right.page_tokens
                }),
            "KV locations disagree on publication metadata"
        );
        let mut candidate = self.clone();
        for (destination, source) in candidate.groups.iter_mut().zip(&other.groups) {
            merge_tensor_locations(&mut destination.tensors, &source.tensors)?;
        }
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

/// Encoding represented by a reusable image-feature publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureKind {
    /// Vision-encoder features, consumed through a call's `vision_inputs`.
    #[serde(rename = "vision_feature")]
    Vision,
    /// Latent image features, consumed through a call's
    /// `latent_feature_input`.
    #[serde(rename = "latent_feature")]
    Latent,
}

/// Closed cross-pool transfer algebra.
///
/// Each variant carries the metadata required to install its product family on
/// a destination worker. Every variant carries exactly one tensor, which
/// [`TransferHandle::tensors`] exposes as a slice.
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
    ///
    /// The engine uses it to drop locators, such as those of lost ranks or
    /// those a destination's transfer edges cannot carry. Mutations through
    /// this view are not revalidated here.
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

/// Conservative estimate of a descriptor's encoded size, used to enforce
/// [`MAX_TRANSFER_HANDLE_BYTES`].
///
/// The integer constants are per-record overhead allowances rather than exact
/// FlatBuffers sizes; only string and list lengths are measured. The Python
/// worker checks its own estimate, `_tensor_transfers_size`, when it commits a
/// call's outputs; a Python estimate below this one lets the worker emit a
/// descriptor that this side rejects.
fn transfer_encoded_size<'a>(tensors: impl IntoIterator<Item = &'a TensorTransfer>) -> usize {
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

/// Appends `source`'s locators to the matching tensors of `destination`,
/// skipping exact duplicates.
///
/// Tensors pair by position and must agree on logical byte size, shape, and
/// dtype name. On error `destination` may be partially updated, so callers
/// merge into a clone and commit only on success.
fn merge_tensor_locations(
    destination: &mut [TensorTransfer],
    source: &[TensorTransfer],
) -> ValidationResult<()> {
    ensure_valid!(
        destination.len() == source.len(),
        "locations disagree on tensor count"
    );
    for (destination, source) in destination.iter_mut().zip(source) {
        // `validate` rejects an empty locator list, so the first-locator
        // indexing below cannot panic.
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
    ///
    /// The engine merges reports of one product, such as per-rank reports,
    /// this way. The product identities and the variant's semantic metadata
    /// must match, and the merged publication must stay within
    /// [`MAX_TRANSFER_HANDLE_BYTES`] and pass [`TensorPublication::validate`];
    /// on any error `self` is left unchanged.
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

/// Maximum estimated encoded size in bytes of one transfer descriptor, as
/// computed by `TransferHandle::encoded_size_bound` and
/// `KvTransfer::encoded_size_bound`. The Python worker's
/// `uniserve_worker.protocol.transfer` module defines the same value.
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
                    "shared-storage transfer endpoint is empty"
                );
                ensure_valid!(!name.is_empty(), "shared-storage transfer name is empty");
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
                        // A process descriptor is 4 bytes, a fabric handle 64.
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
///
/// Checks the variant's semantic metadata, then the tensor's locators, dtype,
/// and actual shape against `product`, then the logical byte size against the
/// product's byte bound, and finally the descriptor size estimate against
/// [`MAX_TRANSFER_HANDLE_BYTES`].
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
            // Height and width are both positive for an image, or both zero
            // for a non-image tensor, which then carries no value range.
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

    // Every variant carries one tensor. Its locators must be valid, their
    // dtype name and element width must match the product's dtype, and the
    // actual shape must fit the product's shape bound.
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
            // A single device-actual extent describes flat capacity, such as
            // encoder features, latents, and feedback images, which the
            // engine's planner bounds by bytes; the transfer publishes the
            // actual rank, and only its element count is bounded.
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

    // Logical byte sizes count each replicated region once.
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

/// Transport dtype name and element width in bytes of a declared dtype, used
/// by both allocation and transfer validation. The names match the Python
/// worker's `dtype_name`, which strips the `torch.` prefix.
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
