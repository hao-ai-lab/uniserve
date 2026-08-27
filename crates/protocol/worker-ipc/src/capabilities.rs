use super::*;

// ---------------------------------------------------------------------------
// Capabilities and startup agreement
// ---------------------------------------------------------------------------

/// One exact decode-and-flow row combination qualified for a single physical call.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MixedExecutionCapability {
    pub decode_rows: u32,
    pub flow_rows: u32,
    pub height: u32,
    pub width: u32,
    pub cfg_branches: u32,
}
/// A worker's advertised capabilities. Admission requires every rank, worker,
/// and frontend to agree on the protocol layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub supported_work: Vec<ForwardMode>,
    pub latent_page_units: u32,
    pub num_latent_pages: u32,
    pub latent_width: u32,
    pub latent_dtype: Option<ModelDtype>,
    pub latent_downsample: u32,
    pub max_vae_grid_tokens: u32,
    pub max_vit_grid_tokens: u32,
    pub max_latent_feature_bytes: u64,
    pub max_vision_feature_bytes: u64,
    pub commit_marker_tokens: u32,
    pub gen_rope_advance: u32,
    pub max_cfg_branches: u32,
    pub bytes_per_token: u64,
    pub groups: Vec<KvCacheGroupSpec>,
    pub kv_dtype: Option<KvCacheDtype>,
    pub model_dtype: ModelDtype,
    pub rank: RankInfo,
    pub pipeline_depth: u32,
    pub encoder_cache_budget: u32,
    pub supported_controls: Vec<RequestKind>,
    pub max_batch_operations: u32,
    pub max_batch_tokens: u32,
    pub max_request_pool_size: u32,
    pub max_unresolved_window: u32,
    pub incremental_kv_publication: bool,
    pub mixed_buckets: Vec<MixedExecutionCapability>,
    pub sampling_ownership: SamplingOwnership,
    pub resource_classes: Vec<ResourceClass>,
    pub model_identity: Option<Digest>,
    pub weight_digest: Option<Digest>,
}

impl WorkerCapabilities {
    pub fn latent_capacity_units(&self) -> u64 {
        u64::from(self.num_latent_pages.saturating_sub(1))
            .saturating_mul(u64::from(self.latent_page_units))
    }

    pub fn uses_kv(&self) -> bool {
        self.supported_work.iter().any(|variant| {
            matches!(
                variant,
                ForwardMode::TokenExtend
                    | ForwardMode::TokenDecode
                    | ForwardMode::TokenVerify
                    | ForwardMode::Draft
                    | ForwardMode::TransferKvPublish
                    | ForwardMode::TransferKvInstall
            )
        }) || self.resource_classes.contains(&ResourceClass::KvBlock)
    }

    pub fn validate(&self) -> ProtocolResult<()> {
        protocol_ensure!(
            !self.supported_work.is_empty(),
            "worker capabilities declare no work variants"
        );
        protocol_ensure!(
            self.supported_work
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.supported_work.len(),
            "worker capabilities repeat a work variant"
        );
        protocol_ensure!(
            self.supported_controls
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.supported_controls.len(),
            "worker capabilities repeat a control"
        );
        protocol_ensure!(
            self.resource_classes
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.resource_classes.len(),
            "worker capabilities repeat a resource class"
        );
        protocol_ensure!(
            self.max_batch_operations > 0
                && self.max_batch_tokens > 0
                && self.max_request_pool_size > 0
                && self.max_unresolved_window > 0,
            "worker capabilities declare a zero scheduling bound"
        );
        protocol_ensure!(
            self.mixed_buckets.iter().collect::<HashSet<_>>().len() == self.mixed_buckets.len(),
            "worker capabilities repeat a mixed-execution bucket"
        );
        for bucket in &self.mixed_buckets {
            protocol_ensure!(
                bucket.decode_rows > 0
                    && bucket.flow_rows > 0
                    && bucket.height > 0
                    && bucket.width > 0
                    && bucket.cfg_branches > 0
                    && bucket.decode_rows.saturating_add(bucket.flow_rows)
                        <= self.max_batch_operations,
                "worker capabilities declare an invalid mixed-execution bucket"
            );
        }
        protocol_ensure!(
            self.pipeline_depth > 0,
            "worker pipeline depth must be positive"
        );
        if self.uses_kv() {
            protocol_ensure!(
                self.block_size > 0
                    && self.num_blocks > 0
                    && self.num_layers > 0
                    && self.num_kv_heads > 0
                    && self.head_dim > 0
                    && self.bytes_per_token > 0
                    && !self.groups.is_empty()
                    && self.kv_dtype.is_some(),
                "worker capabilities declare incomplete KV geometry"
            );
            let mut next_offset = 0u64;
            for (index, group) in self.groups.iter().enumerate() {
                protocol_ensure!(
                    group.group_id == index as u32
                        && u64::from(group.block_offset) == next_offset
                        && group.num_blocks > 0,
                    "worker KV groups are not a canonical physical page partition"
                );
                next_offset = next_offset
                    .checked_add(u64::from(group.num_blocks))
                    .ok_or_else(|| protocol_error!("worker KV group page range overflows"))?;
            }
            protocol_ensure!(
                next_offset == u64::from(self.num_blocks),
                "worker KV groups do not cover the physical request page pool"
            );
        } else {
            protocol_ensure!(
                self.block_size == 0
                    && self.num_blocks == 0
                    && self.num_layers == 0
                    && self.num_kv_heads == 0
                    && self.head_dim == 0
                    && self.bytes_per_token == 0
                    && self.groups.is_empty()
                    && self.kv_dtype.is_none(),
                "KV-free worker capabilities must carry zero KV geometry"
            );
        }
        let has_latent_geometry = self.latent_page_units > 0
            || self.num_latent_pages > 0
            || self.latent_width > 0
            || self.latent_dtype.is_some();
        if has_latent_geometry || self.resource_classes.contains(&ResourceClass::ImageLatent) {
            protocol_ensure!(
                self.latent_page_units > 0
                    && self.num_latent_pages > 1
                    && self.latent_width > 0
                    && self.latent_dtype.is_some(),
                "worker capabilities declare incomplete latent pool geometry"
            );
        }
        let addresses_latent = self
            .supported_work
            .iter()
            .any(|variant| matches!(variant, ForwardMode::GenTransition | ForwardMode::GenFlow));
        protocol_ensure!(
            !addresses_latent || self.resource_classes.contains(&ResourceClass::ImageLatent),
            "worker capabilities advertise latent work without a latent page pool"
        );
        protocol_ensure!(
            self.model_identity.is_some() == self.weight_digest.is_some(),
            "worker capability model and weight identities are incomplete"
        );
        Ok(())
    }

    pub fn generation_runtime_capabilities(&self) -> GenerationRuntimeCapabilities {
        let supports = |variant: ForwardMode| self.supported_work.contains(&variant);
        GenerationRuntimeCapabilities {
            supports_understanding: supports(ForwardMode::TokenExtend)
                && supports(ForwardMode::TokenDecode),
            supports_vision_encode: supports(ForwardMode::EncodeVision),
            supports_latent_encode: supports(ForwardMode::EncodeLatent),
            supports_image_generation: supports(ForwardMode::GenFlow)
                && supports(ForwardMode::Materialize)
                && supports(ForwardMode::TransferKvPublish)
                && self.incremental_kv_publication,
            max_latent_units: self.latent_capacity_units(),
            latent_downsample: self.latent_downsample,
            max_vae_grid_tokens: if self.max_vae_grid_tokens > 0 {
                self.max_vae_grid_tokens
            } else {
                self.latent_capacity_units().min(u64::from(u32::MAX)) as u32
            },
            max_vit_grid_tokens: self.max_vit_grid_tokens,
            max_latent_feature_bytes: self.max_latent_feature_bytes,
            max_vision_feature_bytes: self.max_vision_feature_bytes,
            commit_marker_tokens: self.commit_marker_tokens,
            max_cfg_branches: self.max_cfg_branches,
            encoder_cache_entries: self.encoder_cache_budget,
        }
    }
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_blocks: 4096,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            supported_work: vec![ForwardMode::TokenExtend, ForwardMode::TokenDecode],
            latent_page_units: 0,
            num_latent_pages: 0,
            latent_width: 0,
            latent_dtype: None,
            latent_downsample: 1,
            max_vae_grid_tokens: 0,
            max_vit_grid_tokens: 0,
            max_latent_feature_bytes: 0,
            max_vision_feature_bytes: 0,
            commit_marker_tokens: 2,
            gen_rope_advance: 2,
            max_cfg_branches: 3,
            bytes_per_token: 57_344,
            groups: vec![KvCacheGroupSpec {
                group_id: 0,
                block_offset: 0,
                num_blocks: 4096,
                kind: Default::default(),
            }],
            kv_dtype: Some(KvCacheDtype::BFloat16),
            model_dtype: ModelDtype::BFloat16,
            rank: RankInfo::default(),
            pipeline_depth: 1,
            encoder_cache_budget: 0,
            supported_controls: Vec::new(),
            max_batch_operations: 1,
            max_batch_tokens: 8192,
            max_request_pool_size: 128,
            max_unresolved_window: 1,
            incremental_kv_publication: true,
            mixed_buckets: Vec::new(),
            sampling_ownership: SamplingOwnership::DesignatedRank,
            resource_classes: Vec::new(),
            model_identity: None,
            weight_digest: None,
        }
    }
}
