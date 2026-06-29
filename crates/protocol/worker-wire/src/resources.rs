//! Resource plane types.
//!
//! Makes resources first-class without moving tensors across the boundary. The
//! host owns logical resource identity + policy (it issues [`ResourceLease`]s and
//! asserts [`ResourceInvariant`]s via the host-side `uniserve_kv::ResourceLedger`); the
//! worker owns physical device storage and reports [`ResourcePressure`]. Every
//! field here is an id / class / count / scalar — never a tensor.

use serde::{Deserialize, Serialize};
use uniserve_core::RequestId;

/// The kinds of resource the engine accounts for. A worker declares
/// which classes it manages in `EngineCaps::resource_classes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
 /// Logical KV-cache blocks (paged or model-native).
    KvBlock,
 /// Cached encoder outputs (ViT/VAE embeddings), referenced by handle.
    EncoderOutput,
 /// Denoise image latents resident during generation.
    ImageLatent,
 /// Transient denoise/CFG scratch.
    Scratch,
 /// Resident LoRA adapter weights.
    Adapter,
}

impl ResourceClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceClass::KvBlock => "kv_block",
            ResourceClass::EncoderOutput => "encoder_output",
            ResourceClass::ImageLatent => "image_latent",
            ResourceClass::Scratch => "scratch",
            ResourceClass::Adapter => "adapter",
        }
    }

 /// the single authoritative accounting **unit** for each class.
 /// Both sides MUST account this class in this unit: the host
 /// [`ResourceLease::capacity`] it issues and the worker's
 /// `ResourceRuntime` used/total it reports for the same class must be the
    /// same magnitude. KvBlock leases and reports blocks; Scratch leases and
    /// reports CFG branch slots against the same unit on both sides.
    pub fn unit(&self) -> &'static str {
        match self {
 // Logical KV-cache blocks (pages), not tokens.
            ResourceClass::KvBlock => "blocks",
 // Cached encoder outputs, referenced by handle.
            ResourceClass::EncoderOutput => "handles",
 // Image latent residency, counted in latent tokens.
            ResourceClass::ImageLatent => "latent_tokens",
 // Transient denoise/CFG scratch, counted in CFG branch slots (one
 // per active CFG branch). The physical scratch pool is sized in
 // latent tokens worker-side, but the cross-side *ledger* accounts
 // branch slots on both halves.
            ResourceClass::Scratch => "branch_slots",
 // Resident LoRA adapter slots.
            ResourceClass::Adapter => "adapters",
        }
    }
}

/// An opaque logical handle to a resource instance. `id` is host-assigned; for a
/// KV span it is the lease id, for an encoder output it mirrors the worker's
/// `encoder_handle`, etc. Physical storage stays worker-private.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceHandle {
    pub class: ResourceClass,
    pub id: u64,
}

/// The reuse / release / preemption policy attached to a lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeasePolicy {
 /// Released when the owning request completes or is dropped (default).
    #[default]
    PerRequest,
 /// Eligible for eviction/reuse under pressure (e.g. prefix-cache blocks).
    Evictable,
 /// Pinned until explicitly released (e.g. an in-flight denoise latent —
 /// evicting it discards expensive diffusion work).
    Pinned,
}

/// A host-issued resource lease: who owns it, how much, and its lifetime policy
///. Auditable ownership — the ledger asserts every lease is
/// released after the owner finishes / is dropped / is preempted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLease {
    pub handle: ResourceHandle,
    pub owner_request: RequestId,
 /// Capacity in the class's authoritative unit ([`ResourceClass::unit`]).
 /// KvBlock = blocks, EncoderOutput = handles, ImageLatent = latent
 /// tokens, Scratch = CFG branch slots, Adapter = adapter slots. The host
 /// issues this and the worker's pressure for the same class must report the
 /// same magnitude — see [`ResourceClass::unit`] for the full contract.
    pub capacity: u64,
    pub policy: LeasePolicy,
}

/// Worker-reported pressure for one resource class. Used == in use,
/// evictable == reusable under pressure, free == headroom. All counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePressure {
    pub class: ResourceClass,
    pub total: u64,
    pub used: u64,
    pub evictable: u64,
    pub free: u64,
}

/// A lifecycle event for a lease — the audit trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceEventKind {
    Issued,
    Released,
    Evicted,
    InvariantViolation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceEvent {
    pub kind: ResourceEventKind,
    pub class: ResourceClass,
    pub owner_request: RequestId,
    pub capacity: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_class_strings_roundtrip() {
        for c in [
            ResourceClass::KvBlock,
            ResourceClass::EncoderOutput,
            ResourceClass::ImageLatent,
            ResourceClass::Scratch,
            ResourceClass::Adapter,
        ] {
 // snake_case serde matches as_str (the worker-declared wire form).
            let json = serde_json::to_string(&c).unwrap();
            assert_eq!(json.trim_matches('"'), c.as_str());
        }
    }

 /// pin the authoritative accounting unit per class so a future
 /// change can't silently re-introduce a host/worker unit divergence (e.g.
 /// leasing KvBlock in tokens again, or Scratch in branch slots). This is the
 /// single source the host lease and the worker `ResourceRuntime` both follow.
    #[test]
    fn resource_class_units_are_pinned() {
        assert_eq!(ResourceClass::KvBlock.unit(), "blocks");
        assert_eq!(ResourceClass::EncoderOutput.unit(), "handles");
        assert_eq!(ResourceClass::ImageLatent.unit(), "latent_tokens");
 // Scratch is the observe-only ledger's CFG branch-slot count (the host
 // lease and the worker `_scratch_units` both use branch slots), distinct
 // from the physical token-sized scratch pool.
        assert_eq!(ResourceClass::Scratch.unit(), "branch_slots");
        assert_eq!(ResourceClass::Adapter.unit(), "adapters");
    }

    #[test]
    fn lease_and_pressure_are_plain_descriptors() {
        let lease = ResourceLease {
            handle: ResourceHandle {
                class: ResourceClass::KvBlock,
                id: 7,
            },
            owner_request: RequestId(3),
            capacity: 1024,
            policy: LeasePolicy::Pinned,
        };
        assert_eq!(lease.handle.id, 7);
        assert_eq!(lease.policy, LeasePolicy::Pinned);

        let p = ResourcePressure {
            class: ResourceClass::Scratch,
            total: 100,
            used: 30,
            evictable: 0,
            free: 70,
        };
        assert_eq!((p.used, p.free), (30, 70));
    }
}
