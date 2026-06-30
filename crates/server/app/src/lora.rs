use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, RwLock};
use uniserve_engine_client::EngineCoreClient;
use uniserve_engine_client::protocol::lora::LoraRequest;
pub use uniserve_openai_api::LoraModelResolution;

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Dynamically loaded LoRA adapters keyed by public model name.
    requests: RwLock<BTreeMap<String, LoraRequest>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
}

#[derive(Debug)]
pub enum LoadLoraError {
    AlreadyLoaded {
        lora_name: String,
    },
    BaseModelName {
        lora_name: String,
    },
    Engine(uniserve_engine_client::Error),
    NotLoaded {
        lora_name: String,
    },
    /// the engine/worker LoRA contract carries `lora_id` as `u32`
    /// (worker-wire `WorkerRequest::lora_id`, FB `lora_id:uint`). The registry
    /// allocates ids from a `u64` counter, so guard the single allocation site
    /// rather than silently truncating with `as u32` at each engine boundary.
    IdSpaceExhausted,
}

#[derive(Debug)]
pub enum UnloadLoraError {
    NotFound {
        lora_name: String,
    },
    IntIdMismatch {
        lora_name: String,
        expected: u64,
        actual: u64,
    },
    Engine(uniserve_engine_client::Error),
    NotRemoved {
        lora_name: String,
        lora_int_id: u64,
    },
}

impl LoraManager {
    pub(crate) fn new() -> Self {
        Self {
            requests: RwLock::new(BTreeMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    /// Return base served model names plus dynamically loaded LoRA adapter
    /// names.
    pub(crate) async fn served_model_names(&self, base_model_names: &[String]) -> Vec<String> {
        let mut names = base_model_names.to_vec();
        names.extend(self.requests.read().await.keys().cloned());
        names
    }

    /// Resolve the requested model against one consistent LoRA registry
    /// snapshot.
    pub(crate) async fn resolve_model(
        &self,
        base_model_names: &[String],
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        let requests = self.requests.read().await;
        let mut model_names = base_model_names.to_vec();
        model_names.extend(requests.keys().cloned());
        let lora_request = model_name.and_then(|name| requests.get(name).cloned());

        LoraModelResolution {
            model_names,
            lora_request,
        }
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub(crate) async fn load_lora(
        &self,
        uniserve_engine_client: &EngineCoreClient,
        base_model_names: &[String],
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        let _guard = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let resident = self.requests.read().await.get(&lora_name).cloned();
        if !load_inplace && resident.is_some() {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        // The worker merges one adapter into the weights on load and rejects an
        // `add_lora` over an already-resident adapter (the `load_inplace` flag
        // is not threaded to the worker). To honor an in-place reload, evict the
        // resident adapter from the engine first so the subsequent `add_lora`
        // sees no resident deltas.
        if load_inplace && let Some(resident) = resident.as_ref() {
            uniserve_engine_client
                .remove_lora(resident.lora_int_id)
                .await
                .map_err(LoadLoraError::Engine)?;
        }

        let lora_int_id = resident
            .as_ref()
            .map(|request| request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        // the engine/worker contract is u32; refuse to allocate an id
        // the wire cannot represent instead of silently truncating downstream.
        if lora_int_id > u64::from(u32::MAX) {
            return Err(LoadLoraError::IdSpaceExhausted);
        }
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let loaded = uniserve_engine_client
            .add_lora(&lora_request)
            .await
            .map_err(LoadLoraError::Engine)?;
        if !loaded {
            return Err(LoadLoraError::NotLoaded { lora_name });
        }
        self.requests
            .write()
            .await
            .insert(lora_name, lora_request.clone());
        Ok(lora_request)
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub(crate) async fn unload_lora(
        &self,
        uniserve_engine_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        let _guard = self.update_lock.lock().await;
        let lora_request = self
            .requests
            .read()
            .await
            .get(lora_name)
            .cloned()
            .ok_or_else(|| UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            })?;

        if let Some(actual) = requested_lora_int_id
            && actual != lora_request.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: lora_request.lora_int_id,
                actual,
            });
        }

        let removed = uniserve_engine_client
            .remove_lora(lora_request.lora_int_id)
            .await
            .map_err(UnloadLoraError::Engine)?;
        if !removed {
            return Err(UnloadLoraError::NotRemoved {
                lora_name: lora_request.lora_name,
                lora_int_id: lora_request.lora_int_id,
            });
        }

        Ok(self
            .requests
            .write()
            .await
            .remove(lora_name)
            .unwrap_or(lora_request))
    }
}
