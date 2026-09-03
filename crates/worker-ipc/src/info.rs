//! Worker startup capacities, identity, and supported operations.

use super::*;

/// Fixed physical KV-cache geometry exposed by a worker that executes AR work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub bytes_per_token: u64,
    pub groups: Vec<KvCacheGroup>,
    pub dtype: KvCacheDtype,
}

impl KvCacheConfig {
    pub fn validate(&self) -> ValidationResult<()> {
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
/// Post-load worker geometry, limits, supported work, and model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub model_name: String,
    pub weight_version: u64,
    pub rank: RankInfo,
    pub supported_ops: Vec<OpKind>,
    pub queue_depth: u32,
    pub max_batch_ops: u32,
    pub max_batch_tokens: u32,
    pub request_slots: u32,
    pub kv_cache: Option<KvCacheConfig>,
    pub latent_page_units: u32,
    pub latent_pages: u32,
    pub buffer_pool_bytes: u64,
    pub max_unresolved_ops: u32,
}

impl WorkerInfo {
    pub fn kv_block_size(&self) -> u32 {
        self.kv_cache.as_ref().map_or(0, |config| config.block_size)
    }

    pub fn kv_num_blocks(&self) -> u32 {
        self.kv_cache.as_ref().map_or(0, |config| config.num_blocks)
    }

    pub fn latent_capacity_units(&self) -> u64 {
        u64::from(self.latent_pages.saturating_sub(1))
            .saturating_mul(u64::from(self.latent_page_units))
    }

    pub fn uses_kv(&self) -> bool {
        self.kv_cache.is_some()
    }

    pub fn validate(&self) -> ValidationResult<()> {
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
        ensure_valid!(!self.model_name.is_empty(), "worker model name is empty");
        Ok(())
    }
}

impl Default for WorkerInfo {
    fn default() -> Self {
        Self {
            model_name: "model".to_owned(),
            weight_version: 0,
            rank: RankInfo::default(),
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
