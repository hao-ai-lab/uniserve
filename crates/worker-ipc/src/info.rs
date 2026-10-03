//! Worker identity, supported calls, and startup capacity handshake.
//!
//! A rank answers the engine's `Info` request with a [`WorkerInfo`]. The engine
//! validates it, checks it against the launch configuration, and requires a
//! replacement rank to report an identical description apart from its
//! endpoint. The executor then derives the capacity view the scheduler plans
//! against from the descriptions of all workers (`ExecutorInfo::runtime_info`).

use super::*;

/// Physical KV unit pool exposed by a worker that executes AR work.
///
/// The pool is `num_units` allocation units, unit zero being the padding
/// sentinel. Each group's logical pages draw `units_per_page` units from the
/// one pool, so the scheduler accounts every group's demand in units. Each
/// group also places this rank's share of the model's logical cache: its
/// layers by global id and its KV heads by offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheInfo {
    /// Allocation units in the pool, including the unit-zero sentinel.
    pub num_units: u32,
    /// Physical bytes one unit occupies on this rank: every column plane of
    /// the unit plus its per-unit metadata (initialization flags and, with
    /// FP8, scales).
    pub unit_bytes: u64,
    /// Element data type of KV tensors.
    pub dtype: KvCacheDtype,
    /// Cache groups in table order.
    pub groups: Vec<KvCacheGroup>,
}

impl KvCacheInfo {
    /// Returns the units the scheduler may allocate: all but the sentinel.
    pub fn usable_units(&self) -> u32 {
        self.num_units.saturating_sub(1)
    }

    /// Returns the largest page size of any group, in tokens.
    ///
    /// Every page size divides it (`validate`), so a length aligned to it is
    /// a whole number of pages in every group.
    pub fn max_page_tokens(&self) -> u32 {
        self.groups
            .iter()
            .map(|group| group.page_tokens)
            .max()
            .unwrap_or(0)
    }

    /// Bounds the logical publication of a `tokens`-long visible KV extent,
    /// independently of the producing TP size.
    ///
    /// A publication carries each group's keys and values over the group's
    /// layers (`layer_ids`) and `total_kv_heads`; a sliding-window group
    /// carries at most `window` tokens, the history its readers need. With
    /// FP8 the bound also reserves one FP32 scale for each key and value
    /// head, layer and token, because a partial-page suffix may carry a
    /// complete scale for every head group.
    pub fn publication_bytes(&self, tokens: u32) -> u64 {
        let width = match self.dtype {
            KvCacheDtype::Float16 | KvCacheDtype::BFloat16 => 2,
            KvCacheDtype::Float32 => 4,
            KvCacheDtype::Float8E4m3Fn => 1,
        };

        // FP8 scales are FP32.
        let scale_bytes = if self.dtype == KvCacheDtype::Float8E4m3Fn {
            4
        } else {
            0
        };

        self.groups
            .iter()
            .map(|group| {
                let published = group
                    .kind
                    .window()
                    .map_or(tokens, |window| tokens.min(window));
                let head_bytes = u64::from(group.head_dim) * width + scale_bytes;

                // Key and value tensors over every layer and head of the group.
                (2 * group.layer_ids.len() as u64)
                    .saturating_mul(u64::from(group.total_kv_heads))
                    .saturating_mul(head_bytes)
                    .saturating_mul(u64::from(published))
            })
            .fold(0u64, u64::saturating_add)
    }

    /// Validates the pool size, page shapes and each group's layer and head
    /// placement.
    ///
    /// Every page size must be a power of two that divides the largest one,
    /// so page-aligned prefix lengths agree across groups. A layer belongs to
    /// at most one group. A sliding-window group may not retain sink tokens:
    /// a table keeps one contiguous run of pages.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.num_units > 1 && self.unit_bytes > 0 && !self.groups.is_empty(),
            "worker info declare an incomplete KV unit pool"
        );

        let largest = self.max_page_tokens();
        let mut layers = HashSet::new();
        for group in &self.groups {
            ensure_valid!(
                group.page_tokens.is_power_of_two()
                    && largest.is_multiple_of(group.page_tokens)
                    && group.units_per_page > 0
                    && group.units_per_page < self.num_units,
                "worker KV group page shape is invalid"
            );
            ensure_valid!(
                !group.layer_ids.is_empty()
                    && group.layer_ids.iter().all(|layer| layers.insert(*layer)),
                "worker KV groups omit or repeat a layer"
            );
            ensure_valid!(
                group.num_kv_heads > 0
                    && u64::from(group.kv_head_offset) + u64::from(group.num_kv_heads)
                        <= u64::from(group.total_kv_heads)
                    && group.head_dim > 0,
                "worker KV group head interval exceeds its logical bounds"
            );
            ensure_valid!(
                !matches!(group.kind, KvGroupKind::SlidingWindow { sink, .. } if sink > 0),
                "worker KV group retains sliding-window sink tokens"
            );
        }
        Ok(())
    }

    /// Combines the pools of two KV stages that one request may traverse.
    ///
    /// Stages must agree on dtype and on every group's retention policy and
    /// page shape; their rank-local layer ids and head placement may differ.
    /// The combined pool has the smaller unit count, because the scheduler
    /// assigns one set of unit ids valid in both, the larger unit size, and
    /// per group the union of both stages' layers, which bounds publications.
    pub fn merge(&self, other: &Self) -> ValidationResult<Self> {
        ensure_valid!(
            self.dtype == other.dtype
                && self.groups.len() == other.groups.len()
                && self
                    .groups
                    .iter()
                    .zip(&other.groups)
                    .all(|(left, right)| left.same_page_shape(right)),
            "KV stages expose incompatible cache layouts"
        );

        let groups = self
            .groups
            .iter()
            .zip(&other.groups)
            .map(|(left, right)| {
                let mut layer_ids = left.layer_ids.clone();
                layer_ids.extend(
                    right
                        .layer_ids
                        .iter()
                        .filter(|layer| !left.layer_ids.contains(layer)),
                );
                layer_ids.sort_unstable();
                KvCacheGroup {
                    layer_ids,
                    ..left.clone()
                }
            })
            .collect();
        let merged = Self {
            num_units: self.num_units.min(other.num_units),
            unit_bytes: self.unit_bytes.max(other.unit_bytes),
            dtype: self.dtype,
            groups,
        };
        merged.validate()?;
        Ok(merged)
    }
}

/// One component's finalized configuration shared by all worker descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component name; `WorkerInfo::validate` requires it to be non-empty and
    /// unique within one worker.
    pub name: String,
    #[serde(flatten)]
    pub config: uniserve_core::ComponentConfig,
    /// Named tensor results declared by the loaded computation. Their order
    /// defines product output indices independently of Worker grouping.
    pub outputs: Vec<OutputInfo>,
}

/// A computation's bounded tensor result, before request and storage binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputInfo {
    /// Result name; `OutputInfo::validate` requires it to be non-empty and
    /// `WorkerInfo::validate` unique within its component.
    pub name: String,
    /// Element type of the result tensor.
    pub dtype: DType,
    /// Upper bound on the result's shape, used to size storage before the
    /// actual shape is known.
    pub shape_bound: ShapeBound,
    /// The axes holding a video request's raster, for a result laid out at
    /// the request's canvas. Their extents in `shape_bound` are the largest
    /// height and width the worker admits; a request binds them to its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_axes: Option<RasterAxes>,
}

/// The two axes of a result that hold a video request's raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RasterAxes {
    /// Axis holding the canvas height in pixels.
    pub height: u32,
    /// Axis holding the canvas width in pixels.
    pub width: u32,
}

impl OutputInfo {
    /// Validate the logical representation advertised to allocation and routing.
    ///
    /// Raster axes must be two distinct static axes of the bound.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(!self.name.is_empty(), "tensor result must have a name");
        self.shape_bound.validate()?;
        if let Some(raster) = self.raster_axes {
            let is_static = |axis: u32| {
                matches!(
                    self.shape_bound.dims.get(axis as usize),
                    Some(DimBound::Static(_))
                )
            };
            ensure_valid!(
                raster.height != raster.width
                    && is_static(raster.height)
                    && is_static(raster.width),
                "a tensor result's raster axes must be two distinct static axes"
            );
        }
        Ok(())
    }
}

/// A loaded rank incarnation in an explicitly identified host address space.
/// Backend addresses and storage generations are carried by each publication.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerEndpoint {
    /// Logical instance selected by the engine.
    pub worker_id: String,
    /// Member coordinate within that instance.
    pub rank: u32,
    /// Host identity, independent of process-local device ordinals.
    pub node: String,
    /// Process lifetime identity; multiple Workers may share it.
    pub address_space: String,
    /// Lifetime identity of this loaded Worker rank.
    pub incarnation: String,
}

impl WorkerEndpoint {
    /// Checks identities before accepting startup metadata or a publication.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.worker_id.is_empty()
                && !self.node.is_empty()
                && !self.address_space.is_empty()
                && !self.incarnation.is_empty(),
            "worker endpoint identity is incomplete"
        );
        Ok(())
    }
}

/// Python package of the weightless stub model, the one model that reports
/// no checkpoint identity because it loads no checkpoint.
const STUB_MODEL_PREFIX: &str = "uniserve_models.stub";

/// Whole-tile condition packing of a multi-region video denoiser.
///
/// Each condition occupies whole tiles of `rows` rows: its audio rows fill
/// tiles of their own, then an image's rows fill tiles, while a video's
/// `(latent frames, height, width)` token grid is cut into tiles of `video`
/// tokens along those axes, each tile taking `rows` rows however few tokens
/// it holds. A condition's packed rows are its tile count times `rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionTiles {
    /// Rows of one tile.
    pub rows: u32,
    /// Tokens of one video tile along latent frames, height and width; their
    /// product is `rows`.
    pub video: [u32; 3],
}

impl ConditionTiles {
    /// Packed rows of a condition with `audio_rows` audio rows and either
    /// `image_rows` image rows or a video token grid `(frames, height,
    /// width)`.
    pub fn rows(&self, audio_rows: u32, image_rows: u32, video_grid: Option<[u32; 3]>) -> u64 {
        let tiles = |count: u32, size: u32| u64::from(count.div_ceil(size));
        let visual = match video_grid {
            Some(grid) => grid
                .iter()
                .zip(self.video)
                .map(|(&extent, size)| tiles(extent, size))
                .product(),
            None => tiles(image_rows, self.rows),
        };
        (tiles(audio_rows, self.rows) + visual) * u64::from(self.rows)
    }
}

/// What the video denoiser a deployment places serves.
///
/// The denoiser fixes its schedule, so requests may only restate it; the
/// tasks, canvases, condition packing and sequence capacity bound what
/// admission accepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoDenoiserInfo {
    /// Task names the denoiser serves, in canonical order.
    pub tasks: Vec<String>,
    /// Sigma points of the fixed schedule, the clean endpoint included: one
    /// more than the network evaluations (`WorkerInfo::num_inference_steps`).
    pub schedule_points: u32,
    /// Shift of the video schedule.
    pub video_shift: f64,
    /// Shift of the audio schedule.
    pub audio_shift: f64,
    /// The only canvases the deployment serves: those it prepares. Empty
    /// declares every canvas of the model's canvas rule.
    #[serde(default)]
    pub canvases: Vec<uniserve_core::Canvas>,
    /// The checkpoint's packed sequence capacity in rows, when it has one.
    #[serde(default)]
    pub max_sequence_rows: Option<u32>,
    /// How the denoiser packs condition rows: in whole tiles for a
    /// multi-region denoiser, or densely, one row per condition token, when
    /// absent.
    #[serde(default)]
    pub condition_tiles: Option<ConditionTiles>,
}

impl VideoDenoiserInfo {
    /// Checks that tasks are named once each, the schedule has an endpoint
    /// and positive finite shifts, every canvas and the capacity are
    /// positive, and condition tiles are video tiles of their row count.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.tasks.is_empty()
                && self.tasks.iter().all(|task| !task.is_empty())
                && self.tasks.iter().collect::<HashSet<_>>().len() == self.tasks.len(),
            "a video denoiser names each of its tasks once"
        );
        ensure_valid!(
            self.schedule_points >= 2
                && [self.video_shift, self.audio_shift]
                    .iter()
                    .all(|shift| shift.is_finite() && *shift > 0.0),
            "a video denoiser schedule needs points and positive shifts"
        );
        ensure_valid!(
            self.canvases
                .iter()
                .all(|canvas| canvas.width > 0 && canvas.height > 0)
                && self.max_sequence_rows != Some(0),
            "a video denoiser declares an empty canvas or sequence capacity"
        );
        ensure_valid!(
            self.condition_tiles.is_none_or(|tiles| {
                tiles.rows > 0
                    && tiles.video.iter().all(|&size| size > 0)
                    && tiles
                        .video
                        .iter()
                        .map(|&size| u64::from(size))
                        .product::<u64>()
                        == u64::from(tiles.rows)
            }),
            "a video denoiser's condition tiles are not video tiles of their rows"
        );
        Ok(())
    }
}

/// Post-load worker geometry, limits, supported work, and model identity.
///
/// [`WorkerInfo::validate`] checks only internal consistency. The engine's
/// `RankProcess::finish_startup` additionally checks the description against
/// the launch: `queue_depth`, rank and world size, component configuration,
/// and non-empty resolved numerical settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Loaded model identity.
    pub model_name: String,
    /// Model-declared finite terminal-media computation, when supported.
    #[serde(default)]
    pub media_components: std::collections::BTreeMap<MediaCall, String>,
    /// Effective number of diffusion predictions advertised by the loaded model.
    pub num_inference_steps: u32,
    /// What the deployment's video denoiser serves, when the model has one.
    #[serde(default)]
    pub video_denoiser: Option<VideoDenoiserInfo>,
    /// Identity of the loaded rank and its host address space.
    pub endpoint: WorkerEndpoint,
    /// Rank-local primary compute device used to bind physical transfer edges.
    pub device: String,
    /// Physical transfer mechanisms initialized by this rank.
    pub transfer_backends: Vec<String>,
    /// Whether this rank's device exports a fabric allocation handle.
    ///
    /// A fabric handle is importable from another host inside the fabric
    /// domain; a descriptor handle reaches only the host that created it, so
    /// this is what decides whether a transfer edge may cross hosts.
    #[serde(default)]
    pub fabric_handles: bool,
    /// Number of physical members in this Worker.
    pub world_size: u32,
    /// Resolved compute dtype, compared per rank on restart.
    #[serde(default)]
    pub model_dtype: String,
    /// Resolved attention selection, compared per rank on restart.
    #[serde(default)]
    pub attention_backend: String,
    /// Numerical weight storage formats present on this rank.
    #[serde(default)]
    pub weight_formats: Vec<String>,
    /// Resolved activation quantization formats on this physical rank.
    #[serde(default)]
    pub activation_formats: Vec<String>,
    /// Identity of the loaded checkpoint files: the lowercase hex SHA-256 the
    /// checkpoint identity rule defines over the checkpoint directory. The
    /// weightless stub model reports none; every other worker must.
    #[serde(default)]
    pub checkpoint_identity: String,
    /// Finalized component membership and logical degrees.
    #[serde(default)]
    pub components: Vec<ComponentInfo>,
    /// Call families accepted by the worker.
    pub supported_calls: Vec<CallKind>,
    /// Maximum unresolved physical runs. It must equal the depth the engine
    /// launched the rank with and connects its channel with.
    pub queue_depth: u32,
    /// Maximum calls in one run.
    pub max_batch_calls: u32,
    /// Maximum text tokens represented in one run.
    pub max_batch_tokens: u32,
    /// Maximum calls in one prefill run: the rows the rank's captured
    /// prefill graphs hold. A prefill run no captured graph holds fails, so
    /// the scheduler never forms one. Zero leaves prefill runs bounded by
    /// `max_batch_calls` alone, as when they run eagerly.
    #[serde(default)]
    pub max_prefill_calls: u32,
    /// Maximum calls in one decode run: the rows the rank's largest captured
    /// decode graph holds, fewer when its KV pool cannot hold a page of
    /// every cache group for more rows. A decode run no captured graph
    /// holds fails, so the scheduler never forms one. Zero leaves decode
    /// runs bounded by `max_batch_calls` alone, as when they run eagerly.
    #[serde(default)]
    pub max_decode_calls: u32,
    /// Number of resident request slots.
    pub request_slots: u32,
    /// Paged KV geometry when autoregressive work is supported.
    pub kv_cache: Option<KvCacheInfo>,
    /// Model-defined units stored in one latent page.
    pub latent_page_units: u32,
    /// Physical latent pages including the reserved sentinel page, which
    /// [`WorkerInfo::latent_capacity_units`] excludes.
    pub latent_pages: u32,
    /// Persistent buffer-pool capacity in bytes.
    pub buffer_pool_bytes: u64,
    /// Maximum simultaneously retained encoder feature products.
    pub encoder_cache_entries: u32,
    /// Maximum bytes in one encoder feature product, independent of pool size.
    pub encoder_entry_bytes: u64,
    /// Maximum unresolved calls per request.
    pub max_unresolved_calls: u32,
    /// Concurrent host-lane tasks this rank's bounded host executor admits.
    /// The scheduler treats zero as one.
    #[serde(default)]
    pub host_lane_capacity: u32,
}

impl WorkerInfo {
    /// Returns the fixed prediction count from the model-declared plan.
    pub fn denoise_steps(&self) -> u32 {
        self.num_inference_steps
    }

    /// Returns the advertised allocatable KV units, or zero when KV is
    /// unsupported.
    pub fn kv_usable_units(&self) -> u32 {
        self.kv_cache.as_ref().map_or(0, KvCacheInfo::usable_units)
    }

    /// Returns total latent capacity in model-defined units.
    pub fn latent_capacity_units(&self) -> u64 {
        u64::from(self.latent_pages.saturating_sub(1))
            .saturating_mul(u64::from(self.latent_page_units))
    }

    /// Returns whether the worker advertises paged KV capacity.
    pub fn uses_kv(&self) -> bool {
        self.kv_cache.is_some()
    }

    /// Validates worker identity, capacity, call, and rank invariants.
    pub fn validate(&self) -> ValidationResult<()> {
        // Identity and physical transfer capabilities.
        self.endpoint.validate()?;
        ensure_valid!(
            !self.device.is_empty()
                && !self.transfer_backends.is_empty()
                && self
                    .transfer_backends
                    .iter()
                    .all(|name| matches!(name.as_str(), "local" | "shm" | "cuda_vmm" | "channel"))
                && self.transfer_backends.iter().collect::<HashSet<_>>().len()
                    == self.transfer_backends.len(),
            "worker physical transfer capabilities are incomplete"
        );
        ensure_valid!(
            self.world_size > 0 && self.endpoint.rank < self.world_size,
            "worker process rank is outside its world"
        );

        // Components: unique names and outputs, membership within the world,
        // and parallel degrees consistent with membership.
        let mut names = HashSet::new();
        for component in &self.components {
            ensure_valid!(
                !component.name.is_empty() && names.insert(&component.name),
                "worker repeats or omits a component name"
            );
            let mut outputs = HashSet::new();
            for output in &component.outputs {
                output.validate()?;
                ensure_valid!(
                    outputs.insert(&output.name),
                    "component repeats a tensor result name"
                );
            }
            let params = &component.config;
            ensure_valid!(
                !params.ranks.is_empty()
                    && params
                        .ranks
                        .iter()
                        .all(|&rank| rank < self.world_size as usize),
                "component membership is outside the process world"
            );
            ensure_valid!(
                params.ranks.iter().collect::<HashSet<_>>().len() == params.ranks.len(),
                "component repeats process ranks"
            );
            let degree = params
                .parallel_config
                .world_size()
                .map_err(|error| invalid_message!("{error}"))?;
            ensure_valid!(
                degree <= u32::MAX as usize,
                "parallel degrees exceed protocol range"
            );
            ensure_valid!(
                params.units_per_rank > 0 && params.units_per_rank <= u32::MAX as usize,
                "component unit capacity is outside protocol range"
            );
            // A distributed component spreads independent units over its member
            // ranks without model parallelism, so its degree is one; otherwise
            // every member rank is one position in the parallel degrees.
            ensure_valid!(
                if params.distribution.is_some() {
                    degree == 1
                } else {
                    degree == params.ranks.len()
                },
                "component membership disagrees with parallel degrees"
            );
        }

        // A collective-only participant owns no request components or KV.
        // It remains addressable for progress, failure and shutdown.
        ensure_valid!(
            !self.supported_calls.is_empty()
                || (self.components.is_empty()
                    && self.kv_cache.is_none()
                    && self.latent_pages == 0
                    && self.media_components.is_empty()),
            "a worker without call kinds must own no request computation"
        );
        ensure_valid!(
            self.supported_calls
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.supported_calls.len(),
            "worker info repeats a call kind"
        );
        // A worker names the component serving each media call it implements.
        // Here every named call must be one it advertises, with a non-empty
        // component name. The video graph may span workers, a model worker
        // decoding and a host worker encoding and muxing, so its completeness
        // and its diffusion step count are the executor's checks over every
        // worker.
        ensure_valid!(
            self.media_components.iter().all(|(call, component)| {
                !component.is_empty() && self.supported_calls.contains(&CallKind::Media(*call))
            }),
            "a media call names no component this worker serves"
        );
        if let Some(video) = &self.video_denoiser {
            video.validate()?;
            ensure_valid!(
                video.schedule_points == self.num_inference_steps + 1,
                "a video denoiser's schedule points disagree with its evaluations"
            );
        }
        ensure_valid!(
            self.max_batch_calls > 0
                && self.max_batch_tokens > 0
                && self.request_slots > 0
                && self.max_unresolved_calls > 0,
            "worker info declare a zero scheduling bound"
        );
        ensure_valid!(self.queue_depth > 0, "worker queue depth must be positive");
        ensure_valid!(
            self.max_prefill_calls <= self.max_batch_calls,
            "worker prefill call bound exceeds its batch call bound"
        );
        ensure_valid!(
            self.max_decode_calls <= self.max_batch_calls,
            "worker decode call bound exceeds its batch call bound"
        );

        // Advertised call families require their corresponding pools; every
        // token-model forward reads or extends a request's KV cache.
        let requires_kv = self
            .supported_calls
            .iter()
            .any(|variant| matches!(variant, CallKind::Forward(_)));
        ensure_valid!(
            !requires_kv || self.kv_cache.is_some(),
            "worker advertises token work without a KV cache"
        );
        if let Some(kv_cache) = &self.kv_cache {
            kv_cache.validate()?;
        }
        // Latent capacity is either absent or at least one usable page beyond
        // the sentinel.
        let has_latent_capacity = self.latent_page_units > 0 || self.latent_pages > 0;
        if has_latent_capacity {
            ensure_valid!(
                self.latent_page_units > 0 && self.latent_pages > 1,
                "worker info declare incomplete latent pool capacity"
            );
        }

        // Model identity remains mandatory independently of enabled resources.
        ensure_valid!(!self.model_name.is_empty(), "worker model name is empty");
        // Only the stub model has no checkpoint behind it; a served checkpoint
        // must be identified so ranks can be held to the same one.
        ensure_valid!(
            if self.checkpoint_identity.is_empty() {
                self.model_name.starts_with(STUB_MODEL_PREFIX)
            } else {
                self.checkpoint_identity.len() == 64
                    && self
                        .checkpoint_identity
                        .bytes()
                        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
            },
            "worker checkpoint identity is missing or malformed"
        );
        Ok(())
    }
}

impl Default for WorkerInfo {
    /// Returns a valid single-rank autoregressive worker description.
    fn default() -> Self {
        Self {
            model_name: "model".to_owned(),
            media_components: Default::default(),
            num_inference_steps: 0,
            video_denoiser: None,
            fabric_handles: false,
            endpoint: WorkerEndpoint {
                worker_id: "worker".into(),
                rank: 0,
                node: "simulation".into(),
                address_space: "simulation".into(),
                incarnation: "simulation".into(),
            },
            world_size: 1,
            device: "cpu".into(),
            transfer_backends: vec!["local".into()],
            model_dtype: String::new(),
            attention_backend: String::new(),
            weight_formats: Vec::new(),
            activation_formats: Vec::new(),
            checkpoint_identity: "0".repeat(64),
            components: Vec::new(),
            supported_calls: vec![
                CallKind::Forward(ForwardMode::Prefill),
                CallKind::Forward(ForwardMode::Decode),
            ],
            queue_depth: 1,
            max_batch_calls: 1,
            max_batch_tokens: 8192,
            max_prefill_calls: 0,
            max_decode_calls: 0,
            request_slots: 128,
            // One full-attention group of 28 layers with 8 BF16 heads of 128:
            // each 64-token page is one unit of 28 columns of 128 KiB K and V
            // planes plus one initialization flag per plane.
            kv_cache: Some(KvCacheInfo {
                num_units: 4096,
                unit_bytes: 28 * 2 * (64 * 8 * 128 * 2 + 1),
                dtype: KvCacheDtype::BFloat16,
                groups: vec![KvCacheGroup {
                    kind: Default::default(),
                    page_tokens: 64,
                    units_per_page: 1,
                    layer_ids: (0..28).collect(),
                    num_kv_heads: 8,
                    total_kv_heads: 8,
                    kv_head_offset: 0,
                    head_dim: 128,
                }],
            }),
            latent_page_units: 0,
            latent_pages: 0,
            buffer_pool_bytes: 0,
            encoder_cache_entries: 0,
            encoder_entry_bytes: 0,
            max_unresolved_calls: 1,
            host_lane_capacity: 1,
        }
    }
}
