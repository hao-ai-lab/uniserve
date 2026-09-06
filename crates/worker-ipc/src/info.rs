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
    /// KV heads stored per layer.
    pub num_kv_heads: u32,
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
    /// Validates positive geometry and a complete non-overlapping group partition.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish the physical dimensions before summing the group partition.
        ensure_valid!(
            self.block_size > 0
                && self.num_blocks > 0
                && self.num_layers > 0
                && self.num_kv_heads > 0
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
/// One component's finalized deployment shared by all worker descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub name: String,
    #[serde(flatten)]
    pub deployment: uniserve_core::ComponentDeployConfig,
}

/// Post-load worker geometry, limits, supported work, and model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Loaded model identity.
    pub model_name: String,
    /// Model-weight revision used to reject cross-version products.
    pub weight_version: u64,
    /// Physical process rank topology.
    pub rank: RankInfo,
    /// Stable identity of the expanded component configuration.
    #[serde(default)]
    pub configuration_id: String,
    /// Finalized component membership and logical degrees.
    #[serde(default)]
    pub components: Vec<ComponentInfo>,
    /// Operation families accepted by the worker.
    pub supported_ops: Vec<OpKind>,
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
        ensure_valid!(
            self.rank.world_size > 0 && self.rank.rank < self.rank.world_size,
            "worker process rank is outside its world"
        );
        let mut names = HashSet::new();
        for component in &self.components {
            ensure_valid!(
                !component.name.is_empty() && names.insert(&component.name),
                "worker repeats or omits a component name"
            );
            let placement = &component.deployment;
            ensure_valid!(
                !placement.ranks.is_empty()
                    && placement
                        .ranks
                        .iter()
                        .all(|&rank| rank < self.rank.world_size as usize),
                "component membership is outside the process world"
            );
            ensure_valid!(
                placement.ranks.iter().collect::<HashSet<_>>().len() == placement.ranks.len(),
                "component repeats process ranks"
            );
            let degree = placement
                .parallel_config
                .world_size()
                .map_err(|error| invalid_message!("{error}"))?;
            ensure_valid!(
                degree <= u32::MAX as usize,
                "parallel degrees exceed protocol range"
            );
            ensure_valid!(
                placement.units_per_rank > 0 && placement.units_per_rank <= u32::MAX as usize,
                "component unit capacity is outside protocol range"
            );
            ensure_valid!(
                if placement.distribution.is_some() {
                    degree == 1
                } else {
                    degree == placement.ranks.len()
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
                OpKind::ArExtend | OpKind::ArDecode | OpKind::ArVerify
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
        let addresses_latent = self
            .supported_ops
            .iter()
            .any(|variant| matches!(variant, OpKind::DiffusionPrepare | OpKind::DiffusionStep));
        ensure_valid!(
            !addresses_latent || has_latent_capacity,
            "worker info advertise latent work without a latent page pool"
        );
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
            weight_version: 0,
            rank: RankInfo::default(),
            configuration_id: String::new(),
            components: Vec::new(),
            supported_ops: vec![OpKind::ArExtend, OpKind::ArDecode],
            queue_depth: 1,
            max_batch_ops: 1,
            max_batch_tokens: 8192,
            request_slots: 128,
            kv_cache: Some(KvCacheConfig {
                block_size: 64,
                num_blocks: 4096,
                num_layers: 28,
                num_kv_heads: 8,
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
