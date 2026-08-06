//! Resource classes and the canonical finite-credit vector.
//!
//! Physical stores continue to report class-specific pressure, while admission
//! and operation registration use one atomic vector. Every component is a hard
//! maximum in the unit named by [`CreditDimension`].

use serde::{Deserialize, Serialize};

/// The kinds of physical resource exposed by worker store telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    KvBlock,
    EncoderOutput,
    ImageLatent,
    Scratch,
}

impl ResourceClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceClass::KvBlock => "kv_block",
            ResourceClass::EncoderOutput => "encoder_output",
            ResourceClass::ImageLatent => "image_latent",
            ResourceClass::Scratch => "scratch",
        }
    }

    pub fn unit(&self) -> &'static str {
        match self {
            ResourceClass::KvBlock => "pages",
            ResourceClass::EncoderOutput => "handles",
            ResourceClass::ImageLatent => "bytes",
            ResourceClass::Scratch => "bytes",
        }
    }
}

/// One dimension of the admission and registration credit vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CreditDimension {
    RegisteredOperations,
    ExecutionSlots,
    CompletionSlots,
    DeviceProducts,
    KvPages,
    RollbackDeltas,
    LatentArtifactBytes,
    PinnedCompletionStagingBytes,
    TransferBytes,
    TransferTickets,
    CpuTasks,
    OutputJournalBytes,
}

impl CreditDimension {
    pub const ALL: [Self; 12] = [
        Self::RegisteredOperations,
        Self::ExecutionSlots,
        Self::CompletionSlots,
        Self::DeviceProducts,
        Self::KvPages,
        Self::RollbackDeltas,
        Self::LatentArtifactBytes,
        Self::PinnedCompletionStagingBytes,
        Self::TransferBytes,
        Self::TransferTickets,
        Self::CpuTasks,
        Self::OutputJournalBytes,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RegisteredOperations => "registered_operations",
            Self::ExecutionSlots => "execution_slots",
            Self::CompletionSlots => "completion_slots",
            Self::DeviceProducts => "device_products",
            Self::KvPages => "kv_pages",
            Self::RollbackDeltas => "rollback_deltas",
            Self::LatentArtifactBytes => "latent_artifact_bytes",
            Self::PinnedCompletionStagingBytes => "pinned_completion_staging_bytes",
            Self::TransferBytes => "transfer_bytes",
            Self::TransferTickets => "transfer_tickets",
            Self::CpuTasks => "cpu_tasks",
            Self::OutputJournalBytes => "output_journal_bytes",
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Self::LatentArtifactBytes
            | Self::PinnedCompletionStagingBytes
            | Self::TransferBytes
            | Self::OutputJournalBytes => "bytes",
            Self::KvPages => "pages",
            Self::RegisteredOperations
            | Self::ExecutionSlots
            | Self::CompletionSlots
            | Self::DeviceProducts
            | Self::RollbackDeltas
            | Self::TransferTickets
            | Self::CpuTasks => "count",
        }
    }
}

/// Complete finite resource reservation. Arithmetic is checked so accounting
/// faults cannot be hidden by saturation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CreditVector {
    pub registered_operations: u64,
    pub execution_slots: u64,
    pub completion_slots: u64,
    pub device_products: u64,
    pub kv_pages: u64,
    pub rollback_deltas: u64,
    pub latent_artifact_bytes: u64,
    pub pinned_completion_staging_bytes: u64,
    pub transfer_bytes: u64,
    pub transfer_tickets: u64,
    pub cpu_tasks: u64,
    pub output_journal_bytes: u64,
}

impl CreditVector {
    pub const ZERO: Self = Self {
        registered_operations: 0,
        execution_slots: 0,
        completion_slots: 0,
        device_products: 0,
        kv_pages: 0,
        rollback_deltas: 0,
        latent_artifact_bytes: 0,
        pinned_completion_staging_bytes: 0,
        transfer_bytes: 0,
        transfer_tickets: 0,
        cpu_tasks: 0,
        output_journal_bytes: 0,
    };

    pub fn get(self, dimension: CreditDimension) -> u64 {
        match dimension {
            CreditDimension::RegisteredOperations => self.registered_operations,
            CreditDimension::ExecutionSlots => self.execution_slots,
            CreditDimension::CompletionSlots => self.completion_slots,
            CreditDimension::DeviceProducts => self.device_products,
            CreditDimension::KvPages => self.kv_pages,
            CreditDimension::RollbackDeltas => self.rollback_deltas,
            CreditDimension::LatentArtifactBytes => self.latent_artifact_bytes,
            CreditDimension::PinnedCompletionStagingBytes => self.pinned_completion_staging_bytes,
            CreditDimension::TransferBytes => self.transfer_bytes,
            CreditDimension::TransferTickets => self.transfer_tickets,
            CreditDimension::CpuTasks => self.cpu_tasks,
            CreditDimension::OutputJournalBytes => self.output_journal_bytes,
        }
    }

    pub fn set(&mut self, dimension: CreditDimension, value: u64) {
        match dimension {
            CreditDimension::RegisteredOperations => self.registered_operations = value,
            CreditDimension::ExecutionSlots => self.execution_slots = value,
            CreditDimension::CompletionSlots => self.completion_slots = value,
            CreditDimension::DeviceProducts => self.device_products = value,
            CreditDimension::KvPages => self.kv_pages = value,
            CreditDimension::RollbackDeltas => self.rollback_deltas = value,
            CreditDimension::LatentArtifactBytes => self.latent_artifact_bytes = value,
            CreditDimension::PinnedCompletionStagingBytes => {
                self.pinned_completion_staging_bytes = value;
            }
            CreditDimension::TransferBytes => self.transfer_bytes = value,
            CreditDimension::TransferTickets => self.transfer_tickets = value,
            CreditDimension::CpuTasks => self.cpu_tasks = value,
            CreditDimension::OutputJournalBytes => self.output_journal_bytes = value,
        }
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        let mut result = Self::ZERO;
        for dimension in CreditDimension::ALL {
            result.set(
                dimension,
                self.get(dimension).checked_add(other.get(dimension))?,
            );
        }
        Some(result)
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        let mut result = Self::ZERO;
        for dimension in CreditDimension::ALL {
            result.set(
                dimension,
                self.get(dimension).checked_sub(other.get(dimension))?,
            );
        }
        Some(result)
    }

    pub fn contains(self, requested: Self) -> bool {
        CreditDimension::ALL
            .into_iter()
            .all(|dimension| requested.get(dimension) <= self.get(dimension))
    }

    pub fn first_exhausted(self, used: Self, requested: Self) -> Option<CreditDimension> {
        CreditDimension::ALL.into_iter().find(|dimension| {
            used.get(*dimension)
                .checked_add(requested.get(*dimension))
                .is_none_or(|projected| projected > self.get(*dimension))
        })
    }

    pub fn is_zero(self) -> bool {
        CreditDimension::ALL
            .into_iter()
            .all(|dimension| self.get(dimension) == 0)
    }
}

/// Per-route request maximum and the worker-wide capacity that backs it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteCreditLimits {
    pub per_request: CreditVector,
    pub worker: CreditVector,
}

impl RouteCreditLimits {
    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.worker.contains(self.per_request),
            "route per-request credits exceed worker-wide credits"
        );
        Ok(())
    }
}

/// Worker-reported pressure for one physical resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePressure {
    pub class: ResourceClass,
    pub total: u64,
    pub used: u64,
    pub evictable: u64,
    pub free: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_vector_arithmetic_and_exhaustion_are_dimension_exact() {
        let capacity = CreditVector {
            registered_operations: 4,
            execution_slots: 2,
            completion_slots: 2,
            transfer_bytes: 1024,
            ..CreditVector::ZERO
        };
        let first = CreditVector {
            registered_operations: 2,
            execution_slots: 1,
            completion_slots: 1,
            transfer_bytes: 768,
            ..CreditVector::ZERO
        };
        let used = CreditVector::ZERO.checked_add(first).unwrap();
        assert_eq!(used.checked_sub(first), Some(CreditVector::ZERO));
        let second = CreditVector {
            transfer_bytes: 512,
            ..CreditVector::ZERO
        };
        assert_eq!(
            capacity.first_exhausted(used, second),
            Some(CreditDimension::TransferBytes)
        );
    }

    #[test]
    fn route_limits_cover_every_per_request_dimension() {
        let per_request = CreditVector {
            registered_operations: 2,
            kv_pages: 16,
            output_journal_bytes: 4096,
            ..CreditVector::ZERO
        };
        RouteCreditLimits {
            per_request,
            worker: CreditVector {
                registered_operations: 8,
                kv_pages: 64,
                output_journal_bytes: 16_384,
                ..CreditVector::ZERO
            },
        }
        .validate()
        .unwrap();
    }
}
