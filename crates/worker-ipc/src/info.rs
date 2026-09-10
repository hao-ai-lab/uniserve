//! Worker identity, supported operations, and startup capacity handshake.

use super::*;

/// Fixed physical KV-cache geometry exposed by a worker that executes AR work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Tokens stored in each physical KV page.
    pub block_size: u32,
    /// Total physical pages in the request KV pool.
    pub num_blocks: u32,
    /// Transformer layers represented in the cache.
    pub num_layers: u32,
    /// Transformer layers in the complete logical cache.
    pub total_layers: u32,
    /// First logical layer stored by this rank.
    pub layer_offset: u32,
    /// KV heads stored per layer.
    pub num_kv_heads: u32,
    /// Logical model KV heads across all tensor-parallel members.
    pub total_kv_heads: u32,
    /// First logical KV head stored by this rank.
    pub kv_head_offset: u32,
    /// Elements stored per KV head.
    pub head_dim: u32,
    /// Storage consumed by one token across all layers.
    pub bytes_per_token: u64,
    /// Positional attention groups partitioning the physical pages.
    pub groups: Vec<KvCacheGroup>,
    /// Element data type of KV tensors.
    pub dtype: KvCacheDtype,
}

impl KvCacheConfig {
    /// Bound one token's logical publication independently of the producing TP size.
    /// A partial-page suffix may carry a complete scale for every head group.
    pub fn publication_bytes_per_token(&self) -> u64 {
        let width = match self.dtype {
            KvCacheDtype::Float16 | KvCacheDtype::BFloat16 => 2,
            KvCacheDtype::Float32 => 4,
            KvCacheDtype::Float8E4m3Fn => 1,
        };
        let head_bytes = u64::from(self.head_dim) * width;
        let scale_bytes = if self.dtype == KvCacheDtype::Float8E4m3Fn {
            4
        } else {
            0
        };
        (2 * u64::from(self.total_layers))
            .saturating_mul(u64::from(self.total_kv_heads))
            .saturating_mul(head_bytes + scale_bytes)
    }

    /// Validates positive geometry and a complete non-overlapping group partition.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish the physical dimensions before summing the group partition.
        ensure_valid!(
            self.block_size > 0
                && self.num_blocks > 0
                && self.num_layers > 0
                && u64::from(self.layer_offset) + u64::from(self.num_layers)
                    <= u64::from(self.total_layers)
                && self.num_kv_heads > 0
                && u64::from(self.kv_head_offset) + u64::from(self.num_kv_heads)
                    <= u64::from(self.total_kv_heads)
                && self.head_dim > 0
                && self.bytes_per_token > 0
                && !self.groups.is_empty(),
            "worker info declare incomplete KV geometry"
        );
        let mut total_blocks = 0u64;
        for group in &self.groups {
            ensure_valid!(
                group.num_blocks > 0,
                "worker KV groups are not a canonical physical page partition"
            );
            total_blocks = total_blocks
                .checked_add(u64::from(group.num_blocks))
                .ok_or_else(|| invalid_message!("worker KV group page range overflows"))?;
        }
        ensure_valid!(
            total_blocks == u64::from(self.num_blocks),
            "worker KV groups do not cover the physical request page pool"
        );
        Ok(())
    }
}
/// One component's finalized configuration shared by all worker descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryInfo {
    pub name: String,
    #[serde(flatten)]
    pub config: uniserve_core::EntryConfig,
    /// Named tensor results declared by the loaded computation. Their order
    /// defines product output indices independently of Worker grouping.
    pub outputs: Vec<TensorSpec>,
}

/// Expansion rule for one stage of a finite terminal-media plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaPlanRepeat {
    Once,
    Fixed,
    VideoUnits,
}

/// Bounded operation role implemented by the terminal-media executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaStageRole {
    Encode,
    Prepare,
    Denoise,
    VideoDecode,
    AudioDecode,
    VideoAppend,
    AudioAppend,
    Finalize,
}

impl MediaStageRole {
    const ALL: [Self; 8] = [
        Self::Encode,
        Self::Prepare,
        Self::Denoise,
        Self::VideoDecode,
        Self::AudioDecode,
        Self::VideoAppend,
        Self::AudioAppend,
        Self::Finalize,
    ];
}

/// One model-declared computation stage and its upstream contracts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaPlanStage {
    pub name: String,
    pub operation: OpCode,
    pub entry: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub input_from: Option<String>,
    pub repeat: MediaPlanRepeat,
    pub count: u32,
}

impl MediaPlanStage {
    /// Resolve executor behavior from the operation family and expansion rule.
    pub fn role(&self) -> Option<MediaStageRole> {
        match (self.operation, self.repeat) {
            (OpCode::EncoderText, MediaPlanRepeat::Once) => Some(MediaStageRole::Encode),
            (OpCode::DiffusionPrepare, MediaPlanRepeat::Once) => Some(MediaStageRole::Prepare),
            (OpCode::DiffusionStep, MediaPlanRepeat::Fixed) => Some(MediaStageRole::Denoise),
            (OpCode::DiffusionDecode, MediaPlanRepeat::VideoUnits) => {
                Some(MediaStageRole::VideoDecode)
            }
            (OpCode::DiffusionDecode, MediaPlanRepeat::Once) => Some(MediaStageRole::AudioDecode),
            (OpCode::MediaAppend, MediaPlanRepeat::VideoUnits) => Some(MediaStageRole::VideoAppend),
            (OpCode::MediaAppend, MediaPlanRepeat::Once) => Some(MediaStageRole::AudioAppend),
            (OpCode::DiffusionFinalize, MediaPlanRepeat::Once) => Some(MediaStageRole::Finalize),
            _ => None,
        }
    }
}

/// Ordered finite graph used by the shared terminal-media scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaExecutionPlan {
    pub stages: Vec<MediaPlanStage>,
}

impl MediaExecutionPlan {
    /// Validate the bounded stage vocabulary implemented by the media scheduler.
    pub fn validate(&self) -> ValidationResult<()> {
        let mut declared = HashSet::new();
        let mut roles = HashSet::new();
        for stage in &self.stages {
            ensure_valid!(
                !stage.name.is_empty()
                    && !stage.entry.is_empty()
                    && !declared.contains(stage.name.as_str()),
                "media plan repeats or omits a stage name"
            );
            ensure_valid!(
                stage
                    .dependencies
                    .iter()
                    .all(|dependency| declared.contains(dependency.as_str()))
                    && stage
                        .input_from
                        .as_ref()
                        .is_none_or(|input| declared.contains(input.as_str())),
                "media plan references a later or missing stage"
            );
            ensure_valid!(
                match stage.repeat {
                    MediaPlanRepeat::Fixed => stage.count > 0,
                    MediaPlanRepeat::Once | MediaPlanRepeat::VideoUnits => stage.count == 1,
                },
                "media plan declares an invalid repetition count"
            );
            let role = stage
                .role()
                .ok_or_else(|| invalid_message!("media plan contains an unsupported stage role"))?;
            ensure_valid!(
                roles.insert(role),
                "media plan repeats one terminal-media stage role"
            );
            declared.insert(stage.name.as_str());
        }
        ensure_valid!(
            MediaStageRole::ALL.iter().all(|role| roles.contains(role)),
            "media plan omits a required terminal-media stage role"
        );
        Ok(())
    }

    pub fn stage(&self, name: &str) -> Option<&MediaPlanStage> {
        self.stages.iter().find(|stage| stage.name == name)
    }

    pub fn stage_by_role(&self, role: MediaStageRole) -> Option<&MediaPlanStage> {
        self.stages.iter().find(|stage| stage.role() == Some(role))
    }

    pub fn denoise_steps(&self) -> u32 {
        self.stage_by_role(MediaStageRole::Denoise)
            .map_or(0, |stage| stage.count)
    }
}

/// A computation's bounded tensor result, before request and storage binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape_bound: ShapeBound,
}

impl TensorSpec {
    /// Validate the logical representation advertised to allocation and routing.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(!self.name.is_empty(), "tensor result must have a name");
        self.shape_bound.validate()?;
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

/// Post-load worker geometry, limits, supported work, and model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Loaded model identity.
    pub model_name: String,
    /// Model-declared finite terminal-media computation, when supported.
    #[serde(default)]
    pub media_plan: Option<MediaExecutionPlan>,
    /// Model-weight revision used to reject cross-version products.
    /// Identity of the loaded rank and its host address space.
    pub endpoint: WorkerEndpoint,
    /// Rank-local primary compute device used to bind physical transfer edges.
    pub device: String,
    /// Physical transfer mechanisms initialized by this rank.
    pub transfer_backends: Vec<String>,
    /// Number of physical members in this Worker.
    pub world_size: u32,
    /// Stable identity of the expanded component configuration.
    #[serde(default)]
    pub configuration_id: String,
    /// Finalized component membership and logical degrees.
    #[serde(default)]
    pub components: Vec<EntryInfo>,
    /// Operation families accepted by the worker.
    pub supported_ops: Vec<OpCode>,
    /// Maximum unresolved physical runs.
    pub queue_depth: u32,
    /// Maximum operations in one run.
    pub max_batch_ops: u32,
    /// Maximum text tokens represented in one run.
    pub max_batch_tokens: u32,
    /// Number of resident request slots.
    pub request_slots: u32,
    /// Paged KV geometry when autoregressive work is supported.
    pub kv_cache: Option<KvCacheConfig>,
    /// Model-defined units stored in one latent page.
    pub latent_page_units: u32,
    /// Physical latent pages including the reserved sentinel page.
    pub latent_pages: u32,
    /// Persistent buffer-pool capacity in bytes.
    pub buffer_pool_bytes: u64,
    /// Maximum unresolved operations per request lineage.
    pub max_unresolved_ops: u32,
}

impl WorkerInfo {
    /// Returns the fixed prediction count from the model-declared plan.
    pub fn denoise_steps(&self) -> u32 {
        self.media_plan
            .as_ref()
            .map_or(0, MediaExecutionPlan::denoise_steps)
    }

    /// Returns the advertised KV page size, or zero when KV is unsupported.
    pub fn kv_block_size(&self) -> u32 {
        self.kv_cache.as_ref().map_or(0, |config| config.block_size)
    }

    /// Returns the advertised KV page count, or zero when KV is unsupported.
    pub fn kv_num_blocks(&self) -> u32 {
        self.kv_cache.as_ref().map_or(0, |config| config.num_blocks)
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

    /// Validates worker identity, capacity, operation, and rank invariants.
    pub fn validate(&self) -> ValidationResult<()> {
        self.endpoint.validate()?;
        ensure_valid!(
            !self.device.is_empty()
                && !self.transfer_backends.is_empty()
                && self
                    .transfer_backends
                    .iter()
                    .all(|name| matches!(name.as_str(), "local" | "shm" | "cuda_ipc"))
                && self.transfer_backends.iter().collect::<HashSet<_>>().len()
                    == self.transfer_backends.len(),
            "worker physical transfer capabilities are incomplete"
        );
        ensure_valid!(
            self.world_size > 0 && self.endpoint.rank < self.world_size,
            "worker process rank is outside its world"
        );
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
                    "entry repeats a tensor result name"
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
            ensure_valid!(
                if params.distribution.is_some() {
                    degree == 1
                } else {
                    degree == params.ranks.len()
                },
                "component membership disagrees with parallel degrees"
            );
        }
        // Capability and scheduling limits must describe a usable worker.
        ensure_valid!(
            !self.supported_ops.is_empty(),
            "worker info declare no work variants"
        );
        ensure_valid!(
            self.supported_ops
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.supported_ops.len(),
            "worker info repeat a work variant"
        );
        if let Some(plan) = &self.media_plan {
            plan.validate()?;
            ensure_valid!(
                plan.stages
                    .iter()
                    .all(|stage| self.supported_ops.contains(&stage.operation)),
                "media plan uses an operation the worker does not support"
            );
        }
        ensure_valid!(
            self.max_batch_ops > 0
                && self.max_batch_tokens > 0
                && self.request_slots > 0
                && self.max_unresolved_ops > 0,
            "worker info declare a zero scheduling bound"
        );
        ensure_valid!(self.queue_depth > 0, "worker queue depth must be positive");
        // Advertised operation families require their corresponding pools.
        let requires_kv = self.supported_ops.iter().any(|variant| {
            matches!(
                variant,
                OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify
            )
        });
        ensure_valid!(
            !requires_kv || self.kv_cache.is_some(),
            "worker advertises AR work without a KV cache"
        );
        if let Some(kv_cache) = &self.kv_cache {
            kv_cache.validate()?;
        }
        let has_latent_capacity = self.latent_page_units > 0 || self.latent_pages > 0;
        if has_latent_capacity {
            ensure_valid!(
                self.latent_page_units > 0 && self.latent_pages > 1,
                "worker info declare incomplete latent pool capacity"
            );
        }
        // Model identity remains mandatory independently of enabled resources.
        ensure_valid!(!self.model_name.is_empty(), "worker model name is empty");
        Ok(())
    }
}

impl Default for WorkerInfo {
    /// Returns a valid single-rank autoregressive worker description.
    fn default() -> Self {
        Self {
            model_name: "model".to_owned(),
            media_plan: None,
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
            configuration_id: String::new(),
            components: Vec::new(),
            supported_ops: vec![OpCode::ArExtend, OpCode::ArDecode],
            queue_depth: 1,
            max_batch_ops: 1,
            max_batch_tokens: 8192,
            request_slots: 128,
            kv_cache: Some(KvCacheConfig {
                block_size: 64,
                num_blocks: 4096,
                num_layers: 28,
                total_layers: 28,
                layer_offset: 0,
                num_kv_heads: 8,
                total_kv_heads: 8,
                kv_head_offset: 0,
                head_dim: 128,
                bytes_per_token: 57_344,
                groups: vec![KvCacheGroup {
                    num_blocks: 4096,
                    kind: Default::default(),
                }],
                dtype: KvCacheDtype::BFloat16,
            }),
            latent_page_units: 0,
            latent_pages: 0,
            buffer_pool_bytes: 0,
            max_unresolved_ops: 1,
        }
    }
}
