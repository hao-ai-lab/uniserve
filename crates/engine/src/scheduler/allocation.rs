//! Request backing and retained encoder exports owned by the scheduler.
//!
//! The scheduler owns the logical allocation of worker-side storage: it
//! chooses request rows, KV pages, latent pages, and buffer byte spans, and
//! workers use the addresses they are handed. Allocations follow a request
//! from admission (`RequestAllocations`, `MediaAllocations`) through
//! retirement: a finished request registered with its workers parks its
//! allocations in a `RetiringRequest` until its `Finish` command is
//! acknowledged, so storage is reused only after the workers' physical readers
//! have finished.

use super::*;
use uniserve_worker_ipc::BufferId;

/// Logical storage reservations and retained exports owned by the engine.
pub(super) struct Storage {
    /// Text KV pages and prefix cache; `None` when the runtime has no KV cache.
    pub(super) cache: Option<KVCacheManager>,
    pub(super) encoder_cache: crate::kv::EncoderCacheManager,
    /// Encoder-cache entries promised to admitted requests. `admit` checks it
    /// against `EncoderCacheManager::budget` only for requests that reserve
    /// their worst case.
    pub(super) reserved_encoder_entries: usize,
    /// Request-state rows of token requests and their flow prefixes, sized by
    /// the runtime's `request_slots`.
    pub(super) request_pool: RequestPool,
    pub(super) latent_pool: LatentPool,
    /// KV units reserved by admitted worst-case requests.
    pub(super) reserved_units: usize,
    /// Byte spans of persistent products in the runtime's buffer pool.
    pub(super) buffer_pool: BufferPool,
    /// Per-worker row and buffer address spaces of the media workers.
    pub(super) media_storage: HashMap<crate::WorkerId, MediaStorage>,
    /// Spans whose ownership moved from their producing request to
    /// encoder-cache retention, keyed by the cached product's buffer.
    pub(super) encoder_buffers: HashMap<uniserve_worker_ipc::BufferId, BufferSpan>,
    /// Spans that left encoder-cache retention (or failed to enter it) and
    /// whose `Free` command is queued or in flight; the span returns to
    /// `buffer_pool` when `Scheduler::acknowledge_commands` sees that `Free`.
    /// A request-owned span stays with its request until then instead.
    pub(super) pending_buffer_frees: HashMap<BufferId, BufferSpan>,
}

/// A media allocation named a worker the scheduler holds no media storage for.
///
/// Media storage exists for every loaded worker that serves a media call and
/// every media allocation is taken from it, so this reports a scheduler defect.
#[derive(Debug)]
pub(super) struct UnknownMediaWorker(pub(super) crate::WorkerId);

impl std::fmt::Display for UnknownMediaWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "media allocation names worker {:?} without media storage",
            self.0
        )
    }
}

impl Storage {
    pub(super) fn free_media_request(
        &mut self,
        worker: &crate::WorkerId,
        allocation: RequestSlot,
    ) -> Result<(), UnknownMediaWorker> {
        self.media_storage
            .get_mut(worker)
            .ok_or_else(|| UnknownMediaWorker(worker.clone()))?
            .requests
            .free(allocation);
        Ok(())
    }

    pub(super) fn free_media_buffer(
        &mut self,
        worker: &crate::WorkerId,
        allocation: BufferSpan,
    ) -> Result<(), UnknownMediaWorker> {
        self.media_storage
            .get_mut(worker)
            .ok_or_else(|| UnknownMediaWorker(worker.clone()))?
            .buffers
            .free(allocation);
        Ok(())
    }

    /// Returns the text KV cache, which exists when the worker serves token
    /// generation; token requests are rejected at submission without it.
    pub(super) fn cache(&self) -> Option<&KVCacheManager> {
        self.cache.as_ref()
    }

    /// Returns the queued KV units, including cached ones allocation evicts.
    pub(super) fn free_units(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |cache| cache.block_pool.free_units())
    }

    /// Returns the KV units the scheduler may allocate.
    pub(super) fn usable_units(&self) -> usize {
        self.cache.as_ref().map_or(0, KVCacheManager::usable_units)
    }

    /// Transfer the existing Tensor allocation to encoder-cache retention.
    ///
    /// Hands `allocation` back, retaining nothing, when it is not owned by the
    /// buffer's producing request or the buffer is already retained.
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
    ///
    /// The `Finish` command for `request` lists them so its workers keep them
    /// readable until their own `Free`. The list is sorted by producer call,
    /// output index, and generation.
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
///
/// `Scheduler::acknowledge_commands` releases the record when a `Finish`
/// acknowledgement names the same `request_key`.
pub(super) struct RetiringRequest {
    pub(super) request_key: RequestKey,
    /// The token request's own allocations and those of its flow prefix;
    /// empty for a media request.
    pub(super) allocations: Vec<RequestAllocations>,
    pub(super) media_allocations: Option<MediaAllocations>,
    /// Persistent product spans still held by the request. An earlier `Free`
    /// acknowledgement may release one of them before the `Finish` does.
    pub(super) buffers: HashMap<BufferId, BufferSpan>,
}

/// Storage one token request, or its flow prefix, holds in the runtime's
/// shared pools.
pub(super) struct RequestAllocations {
    pub(super) request_slot: RequestSlot,
    /// One block table per KV cache group.
    pub(super) kv: KvAllocation,
    /// Latent pages of the current image trajectory, reserved by its latent
    /// preparation or denoising call on a worker that tracks image latents
    /// and released when its image decoding completes or with the request's
    /// other allocations.
    pub(super) latent: Option<LatentPages>,
    /// Persistent product spans keyed by the product's buffer identity.
    pub(super) buffers: HashMap<BufferId, BufferSpan>,
}

impl RequestAllocations {
    /// Returns the request row, sent to workers as `request_pool_idx`.
    pub(super) fn request_slot(&self) -> u32 {
        self.request_slot.index()
    }

    /// Returns shared access to the request block tables.
    pub(super) fn block_tables(&self) -> &[BlockTable] {
        &self.kv.tables
    }

    /// Takes ownership of the request buffer allocation.
    pub(super) fn take_buffer(&mut self, id: BufferId) -> Option<BufferSpan> {
        self.buffers.remove(&id)
    }

    /// Releases each resource through its owning pool.
    ///
    /// KV units need no explicit release: each page in `kv` is a counted
    /// reference, dropped with `self`.
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

/// Storage one media request reserves at admission in the per-worker address
/// spaces of `Storage::media_storage`.
pub(super) struct MediaAllocations {
    /// Declared results keyed by component name and output index.
    pub(super) tensors: HashMap<(String, u32), MediaTensorAllocation>,
    /// The request's row on each worker of its route.
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
    /// Returns the span `product` occupies in `worker`'s reservation.
    ///
    /// A product covering part of the result starts `start_unit` units into
    /// it, where one unit is the product's bytes divided by its static leading
    /// dimension. With `start_unit` zero, or a leading dimension that is not
    /// static, the product binds at the start of the reservation.
    ///
    /// Panics when `worker` holds no reservation for this result.
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
        self.request_slots[worker].index()
    }

    /// Releases the owned request allocation.
    ///
    /// Every allocation on a worker with media storage is released even when
    /// another names an unknown worker; the first unknown worker is reported.
    pub(super) fn free(self, storage: &mut Storage) -> Result<(), UnknownMediaWorker> {
        let mut released = Ok(());
        for tensor in self.tensors.into_values() {
            for (worker, allocation) in tensor.allocations {
                released = released.and(storage.free_media_buffer(&worker, allocation));
            }
        }
        for (worker, allocation) in self.request_slots {
            released = released.and(storage.free_media_request(&worker, allocation));
        }
        released
    }
}
