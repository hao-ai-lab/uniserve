//! Request backing and retained encoder publications owned by the scheduler.

use super::*;
use uniserve_worker_ipc::BufferId;

/// Logical storage reservations and retained publications owned by the engine.
pub(super) struct Storage {
    pub(super) cache: Option<KVCacheManager>,
    pub(super) encoder_cache: crate::kv::EncoderCacheManager,
    pub(super) reserved_encoder_entries: usize,
    pub(super) request_pool: RequestPool,
    pub(super) latent_pool: LatentPool,
    pub(super) reserved_blocks: usize,
    pub(super) buffer_pool: BufferPool,
    pub(super) media_storage: HashMap<crate::WorkerId, MediaStorage>,
    pub(super) encoder_buffers: HashMap<uniserve_worker_ipc::BufferId, BufferSpan>,
    pub(super) pending_buffer_frees: HashMap<BufferId, BufferSpan>,
}

impl Storage {
    pub(super) fn free_media_request(&mut self, worker: &crate::WorkerId, allocation: RequestSlot) {
        self.media_storage
            .get_mut(worker)
            .expect("media allocation names a loaded worker")
            .requests
            .free(allocation);
    }

    pub(super) fn free_media_buffer(&mut self, worker: &crate::WorkerId, allocation: BufferSpan) {
        self.media_storage
            .get_mut(worker)
            .expect("media allocation names a loaded worker")
            .buffers
            .free(allocation);
    }

    pub(super) fn cache(&self) -> &KVCacheManager {
        self.cache
            .as_ref()
            .expect("generation requires loaded KV cache")
    }

    pub(super) fn free_blocks(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |cache| cache.block_pool.free_request_pages())
    }

    pub(super) fn usable_blocks(&self) -> usize {
        self.cache.as_ref().map_or(0, |cache| cache.usable_blocks)
    }

    /// Transfer the existing Tensor allocation to encoder-cache retention.
    pub(super) fn retain_encoder_buffer(
        &mut self,
        buffer: BufferId,
        allocation: BufferSpan,
    ) -> Result<(), BufferSpan> {
        if allocation.owner != buffer.owner || self.encoder_buffers.contains_key(&buffer) {
            return Err(allocation);
        }
        self.encoder_buffers.insert(buffer, allocation);
        Ok(())
    }

    pub(super) fn take_encoder_buffer(&mut self, buffer: BufferId) -> Option<BufferSpan> {
        self.encoder_buffers.remove(&buffer)
    }

    /// Buffers whose cache/reader lifetime extends beyond their producing request.
    pub(super) fn retained_buffers(&self, request: RequestKey) -> Vec<BufferId> {
        let mut buffers: Vec<_> = self
            .encoder_buffers
            .keys()
            .copied()
            .filter(|buffer| buffer.owner == request)
            .collect();
        buffers.sort_unstable_by_key(|buffer| {
            (
                buffer.producer_call_id,
                buffer.output_index,
                buffer.generation,
            )
        });
        buffers
    }
}

/// Allocation ownership retained until the exact request epoch closes on every rank.
pub(super) struct RetiringRequest {
    pub(super) request_key: RequestKey,
    pub(super) allocations: Vec<RequestAllocations>,
    pub(super) media_allocations: Option<MediaAllocations>,
    pub(super) buffers: HashMap<BufferId, BufferSpan>,
}

pub(super) struct RequestAllocations {
    pub(super) request_slot: RequestSlot,
    pub(super) kv: KvAllocation,
    pub(super) latent: Option<LatentPages>,
    pub(super) buffers: HashMap<BufferId, BufferSpan>,
}

impl RequestAllocations {
    /// Returns the request-slot identifier.
    pub(super) fn request_slot(&self) -> u32 {
        self.request_slot.index
    }

    /// Returns shared access to the request block tables.
    pub(super) fn block_tables(&self) -> &[BlockTable] {
        &self.kv.tables
    }

    /// Returns mutable access to the request block tables.
    pub(super) fn block_tables_mut(&mut self) -> &mut Vec<BlockTable> {
        &mut self.kv.tables
    }

    /// Takes ownership of the request buffer allocation.
    pub(super) fn take_buffer(&mut self, id: BufferId) -> Option<BufferSpan> {
        self.buffers.remove(&id)
    }

    /// Releases each resource through its owning pool.
    pub(super) fn free(self, storage: &mut Storage) {
        for buffer in self.buffers.into_values() {
            storage.buffer_pool.free(buffer);
        }
        if let Some(latent) = self.latent {
            storage.latent_pool.free(latent);
        }
        storage.request_pool.free(self.request_slot);
    }
}

pub(super) struct MediaAllocations {
    pub(super) tensors: HashMap<(String, u32), MediaTensorAllocation>,
    pub(super) request_slots: HashMap<crate::WorkerId, RequestSlot>,
}

/// A request reserves each declared result in every Worker address space on
/// its route. Video ranges occupy disjoint slices of the temporal result,
/// independent of decoder Worker width.
pub(super) struct MediaTensorAllocation {
    pub(super) allocations: HashMap<crate::WorkerId, BufferSpan>,
    pub(super) dtype: DType,
    pub(super) shape_bound: ShapeBound,
}

/// Independent scheduler address space for one physical media WorkerGroup.
/// Replicas intentionally reuse row numbers and byte offsets in their own
/// processes; the worker identity keeps those physical addresses distinct.
pub(super) struct MediaStorage {
    pub(super) requests: RequestPool,
    pub(super) buffers: BufferPool,
}

impl MediaTensorAllocation {
    pub(super) fn bind(
        &self,
        product: &TensorRef,
        start_unit: u32,
        worker: &crate::WorkerId,
    ) -> BufferAllocation {
        let offset = self.allocations[worker].offset;
        let unit_bytes = match product.shape_bound.dims.first() {
            Some(DimBound::Static(units)) if start_unit > 0 => {
                product.max_bytes() / u64::from(*units)
            }
            _ => 0,
        };
        BufferAllocation {
            buffer: product.buffer_id(),
            offset: offset + u64::from(start_unit) * unit_bytes,
            bytes: product.max_bytes(),
        }
    }
}

impl MediaAllocations {
    /// Returns the request row assigned in one physical worker address space.
    pub(super) fn request_slot(&self, worker: &crate::WorkerId) -> u32 {
        self.request_slots[worker].index
    }

    /// Releases the owned request allocation.
    pub(super) fn free(self, storage: &mut Storage) {
        for tensor in self.tensors.into_values() {
            for (worker, allocation) in tensor.allocations {
                storage.free_media_buffer(&worker, allocation);
            }
        }
        for (worker, allocation) in self.request_slots {
            storage.free_media_request(&worker, allocation);
        }
    }
}
